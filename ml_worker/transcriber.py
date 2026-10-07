import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from faster_whisper import WhisperModel
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

    def load_model(self):
        logger.info(f"Loading Whisper model: {self.model_name} (device={self.device}, compute_type={self.compute_type})")
        
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
            download_root=str(WorkerConfig.SCRATCH_DIR / "models" / "whisper")
        )
        self.device = actual_device
        logger.info(f"Whisper model loaded successfully on {actual_device}:{device_index}.")

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

        # Configure decoding based on transcription mode
        beam_size = 1
        temperature = 0.0
        if mode == "accurate":
            beam_size = 5
            temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        elif mode == "balanced":
            beam_size = 3
            temperature = 0.0

        lang_arg = language.lower().strip() if (language and language.lower().strip() not in ("auto", "none")) else None
        target_lang_arg = target_language.lower().strip() if target_language else None

        # If lang_arg is 'en', preserve it so Whisper transcribes cleanly without language auto-detection misclassifications
        if lang_arg in ("auto", "none", ""):
            lang_arg = None

        # Determine task ('translate' vs 'transcribe')
        # Whisper's native GPU model ONLY translates any spoken language -> English ('en').
        # If target language is explicitly English ('en') or translation to English is requested:
        if target_lang_arg == "en" or (enable_translation and (target_lang_arg is None or target_lang_arg == "en")):
            task = "translate"
        else:
            # For non-English target languages (e.g., target='fr', 'es', 'de') or auto-detect matching target,
            # run 'transcribe' so Whisper outputs text in the detected spoken language (e.g. French audio -> French text).
            task = "transcribe"

        logger.info(f"Running Whisper inference: task={task}, source_language={lang_arg}, target_language={target_lang_arg or ('en' if task == 'translate' else lang_arg)}")

        # Optimal VAD and decoding parameters for high accuracy & preserving music/vocals
        vad_params = dict(
            threshold=0.20,                # High sensitivity: captures quick remarks & courtroom interjections
            min_speech_duration_ms=100,    # Catch rapid interjections (e.g. 100ms+)
            min_silence_duration_ms=250,    # Isolate rapid micro-pauses between speaker interchanges
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

        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=lang_arg,
            task=task,
            beam_size=beam_size,
            temperature=temperature,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=vad_params,
            condition_on_previous_text=False, # Prevents hallucinations / skipping words
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            initial_prompt=prompt
        )

        detected_language = info.language
        language_probability = info.language_probability
        duration = info.duration

        segments_list = []
        full_text_parts = []
        total_words = 0

        for seg in segments_iter:
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

            if progress_callback:
                progress_callback(seg.end, duration)

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
