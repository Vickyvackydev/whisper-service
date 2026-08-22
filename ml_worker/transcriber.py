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
        if actual_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            actual_device = "cpu"
            self.compute_type = "int8"
        elif actual_device == "auto":
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            if actual_device == "cpu":
                self.compute_type = "int8"

        self.model = WhisperModel(
            self.model_name,
            device=actual_device,
            compute_type=self.compute_type,
            cpu_threads=WorkerConfig.WHISPER_CPU_THREADS,
            download_root=str(WorkerConfig.SCRATCH_DIR / "models" / "whisper")
        )
        self.device = actual_device
        logger.info("Whisper model loaded successfully.")

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

        # Determine task ('translate' vs 'transcribe')
        # 1. If explicit non-English source language equals target language (e.g. fr -> fr, es -> es), run pure transcription in that language.
        if lang_arg and target_lang_arg and lang_arg == target_lang_arg and lang_arg != "en":
            task = "transcribe"
        # 2. If target language is explicitly English ('en') or translation is requested to English
        elif target_lang_arg == "en" or (enable_translation and (target_lang_arg is None or target_lang_arg == "en")):
            task = "translate"
            if lang_arg == "en":
                # exscriptai-backend maps 'auto' to 'en'. Clear lang_arg to allow auto-detection of true audio language (e.g. French 'fr').
                logger.info("Target English requested with source 'en'. Auto-detecting spoken audio language.")
                lang_arg = None
        # 3. If exscriptai-backend sent language="en" & target_language="en" (UI default for auto-detect + target english)
        elif lang_arg == "en" and (target_lang_arg is None or target_lang_arg == "en"):
            task = "translate"
            lang_arg = None
        else:
            task = "transcribe"

        logger.info(f"Running Whisper inference: task={task}, source_language={lang_arg}, target_language={target_lang_arg or ('en' if task == 'translate' else lang_arg)}")

        # Optimal VAD and decoding parameters for high accuracy & preserving music/vocals
        vad_params = dict(
            threshold=0.35,                # Sensitive to low volume / vocal singing
            min_speech_duration_ms=250,
            min_silence_duration_ms=1000,  # Prevent cutting during music pauses
            speech_pad_ms=500              # Pad 500ms to avoid clipping start/end words
        )

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
            initial_prompt="Suit No., /, Court, Plaintiff, Defendant, Counsel, Your Lordship, Milord."
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
