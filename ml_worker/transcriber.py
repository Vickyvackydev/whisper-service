import gc
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from faster_whisper import WhisperModel
try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:
    BatchedInferencePipeline = None
import torch
from ml_worker.config import WorkerConfig
from ml_worker.normalizer import normalize_segment, normalize_case_numbers_and_slashes

logger = logging.getLogger("ml_worker.transcriber")

class Transcriber:
    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self.model_name = WorkerConfig.WHISPER_MODEL
        self.device = WorkerConfig.WHISPER_DEVICE
        self.compute_type = WorkerConfig.WHISPER_COMPUTE_TYPE
        # Strictly 1 CTranslate2 worker thread to prevent parallel_for cudaErrorInvalidDevice
        self.num_workers = 1

    def load_model(self):
        logger.info(
            f"Loading Whisper model: {self.model_name} "
            f"(device={self.device}, compute_type={self.compute_type}, num_workers={self.num_workers})"
        )
        
        # Check CUDA availability
        actual_device = self.device
        device_index = 0
        if actual_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            actual_device = "cpu"
            self.compute_type = "int8"
        elif actual_device == "auto":
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            if actual_device == "cpu":
                self.compute_type = "int8"

        if actual_device == "cuda":
            try:
                device_index = torch.cuda.current_device() if (torch and torch.cuda.is_available()) else 0
                if torch and torch.cuda.is_available():
                    torch.cuda.set_device(device_index)
                logger.info(f"Pinned CTranslate2 / Whisper device to CUDA ordinal: {device_index}")
            except Exception as dev_err:
                logger.warning(f"Note verifying CUDA device index: {dev_err}")
                device_index = 0

        self.model = WhisperModel(
            self.model_name,
            device=actual_device,
            device_index=device_index,
            compute_type=self.compute_type,
            cpu_threads=WorkerConfig.WHISPER_CPU_THREADS,
            num_workers=self.num_workers,
            download_root=str(WorkerConfig.SCRATCH_DIR / "models" / "whisper")
        )

        self.device = actual_device
        logger.info(f"Whisper model loaded successfully on {actual_device}:{device_index} (num_workers={self.num_workers}).")

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        target_language: Optional[str] = None,
        enable_translation: bool = False,
        mode: str = "fast",
        progress_callback = None
    ) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        # Configure decoding based on transcription mode.
        # Avoid 6-step temperature cascade on noisy courtroom silence which causes 6x-30x slowdowns,
        # while preserving 5-beam search and [0.0, 0.2] fallback for full word recall and segment boundary accuracy.
        beam_size = 1
        temperature: Any = 0.0
        if mode == "accurate":
            beam_size = 5
            temperature = [0.0, 0.2]
        elif mode == "balanced":
            beam_size = 3
            temperature = 0.0

        lang_arg = language.lower().strip() if (language and language.lower().strip() not in ("auto", "none")) else None
        target_lang_arg = target_language.lower().strip() if target_language else None

        # If lang_arg is 'en', preserve it so Whisper transcribes cleanly without language auto-detection misclassifications
        if lang_arg in ("auto", "none", ""):
            lang_arg = None

        # Determine task ('translate' vs 'transcribe')
        # Only run 'translate' when translation to English is explicitly enabled and source is not already English.
        if enable_translation and (target_lang_arg is None or target_lang_arg == "en") and lang_arg != "en":
            task = "translate"
        else:
            task = "transcribe"

        logger.info(
            f"Running Whisper inference: task={task}, source_language={lang_arg}, "
            f"target_language={target_lang_arg or ('en' if task == 'translate' else lang_arg)}, "
            f"beam_size={beam_size}, temperature={temperature}"
        )

        # Optimal VAD and decoding parameters for high accuracy & preserving courtroom speech
        vad_params = dict(
            threshold=0.20,                # High sensitivity: captures quick remarks & courtroom interjections
            min_speech_duration_ms=100,    # Catch rapid interjections (e.g. 100ms+)
            min_silence_duration_ms=250,   # Isolate rapid micro-pauses between speaker interchanges
            speech_pad_ms=300              # Pad 300ms to preserve leading/trailing word boundaries
        )

        if self.device == "cuda" and torch and torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass

        prompt = WorkerConfig.WHISPER_INITIAL_PROMPT
        if not prompt or str(prompt).strip().lower() in ("none", "false", "0", ""):
            prompt = None
        else:
            prompt = prompt.strip()

        raw_segments = None
        info = None

        try:
            segments_iter, info = self.model.transcribe(
                str(audio_path),
                language=lang_arg,
                task=task,
                beam_size=beam_size,
                temperature=temperature,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=vad_params,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                initial_prompt=prompt
            )
            temp_segments = []
            for seg in segments_iter:
                temp_segments.append(seg)
                if progress_callback and info and info.duration:
                    progress_callback(seg.end, info.duration)
            raw_segments = temp_segments
        except Exception as unbatched_err:
            logger.warning(f"Standard transcribe encountered error ({unbatched_err}); reloading WhisperModel and retrying...")
            gc.collect()
            if torch and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(0)
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            self.load_model()
            segments_iter, info = self.model.transcribe(
                str(audio_path),
                language=lang_arg,
                task=task,
                beam_size=3,
                temperature=0.0,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=vad_params,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                initial_prompt=prompt
            )
            temp_segments = []
            for seg in segments_iter:
                temp_segments.append(seg)
                if progress_callback and info and info.duration:
                    progress_callback(seg.end, info.duration)
            raw_segments = temp_segments

        detected_language = info.language
        language_probability = info.language_probability
        duration = info.duration

        segments_list = []
        full_text_parts = []
        total_words = 0

        for seg in raw_segments:
            text = seg.text.strip()

            words_data = []
            if seg.words:
                for w in seg.words:
                    word_clean = w.word.strip()
                    if word_clean:
                        total_words += 1
                        words_data.append({
                            "word": word_clean,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "score": round(w.probability, 3) if w.probability is not None else None,
                            "speaker": "SPEAKER_00", # default, updated by diarizer
                            "source_word": None,
                            "mapping_type": None
                        })

            seg_dict = {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": text,
                "source_text": None,
                "speaker": "SPEAKER_00", # default, updated by diarizer
                "words": words_data
            }
            # Normalize legal symbols (e.g. E-slash-21-slash-2025 -> E/21/2025, ill-health, My Lord)
            seg_dict = normalize_segment(seg_dict)
            if seg_dict["text"]:
                full_text_parts.append(seg_dict["text"])
            segments_list.append(seg_dict)

        full_text = " ".join(full_text_parts)

        actual_target_lang = "en" if task == "translate" else (target_lang_arg or detected_language)
        trans_method = "whisper_native" if task == "translate" else None

        return {
            "segments": segments_list,
            "transcribed_text": full_text,
            "source_transcribed_text": None,
            "language": detected_language,
            "language_probability": round(language_probability, 3),
            "duration": round(duration, 3),
            "word_count": total_words,
            "model_used": self.model_name,
            "transcription_mode": mode,
            "target_language": actual_target_lang,
            "translation_method": trans_method
        }
