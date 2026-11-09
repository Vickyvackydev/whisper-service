import gc
import re
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, Any, List
import numpy as np
from faster_whisper import WhisperModel
try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:
    BatchedInferencePipeline = None
import torch
from ml_worker.config import WorkerConfig
from ml_worker.normalizer import normalize_segment, normalize_case_numbers_and_slashes

logger = logging.getLogger("ml_worker.transcriber")

COURTROOM_MARKER_RE = re.compile(
    r"\b(?:my\s+lord|your\s+lordship|your\s+honou?r|court\s+pleases|learned\s+counsel|"
    r"learned\s+silk|statement\s+of\s+claim|statement\s+of\s+defence|motion\s+on\s+notice|"
    r"affidavit|exhibit|cross[\s-]?examination|claimant|defendant|plaintiff|respondent|"
    r"applicant|petitioner|adjourn(?:ed)?|ruling|judg(?:e)?ment)\b",
    re.IGNORECASE,
)

class Transcriber:
    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self.batched_model: Optional[Any] = None
        self.model_name = WorkerConfig.WHISPER_MODEL
        self.device = WorkerConfig.WHISPER_DEVICE
        self.compute_type = WorkerConfig.WHISPER_COMPUTE_TYPE
        # Strictly 1 CTranslate2 worker thread to prevent parallel_for cudaErrorInvalidDevice
        self.num_workers = 1

    @staticmethod
    def _has_active_vocal_energy(audio_slice: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Determines whether an untranscribed audio window contains active human vocal/singing
        energy (250 Hz - 3800 Hz) with syllable modulation, versus flat courtroom recess hum or silence.
        """
        if audio_slice is None or len(audio_slice) < sample_rate * 2:
            return False
        rms = float(np.sqrt(np.mean(audio_slice ** 2)))
        if rms < 0.006:
            return False

        frame_len = sample_rate // 2  # 0.5s frames
        n_frames = len(audio_slice) // frame_len
        if n_frames < 4:
            return rms >= 0.012

        frames = audio_slice[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        p90 = float(np.percentile(frame_rms, 90))
        p15 = float(np.percentile(frame_rms, 15))
        if p90 < 0.012:
            return False

        # Steady fan/AC hum has flat envelope; speech and music vocals have dynamic modulation or strong energy
        modulation_ratio = p90 / max(1e-5, p15)
        if modulation_ratio < 1.35 and rms < 0.025:
            return False

        # Check mid-frequency vocal band energy (250 Hz - 3800 Hz) on active frames
        active_idx = np.argsort(frame_rms)[-min(8, n_frames):]
        active_frames = frames[active_idx]
        spec = np.abs(np.fft.rfft(active_frames, axis=1)) ** 2
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
        total_power = float(np.sum(spec))
        if total_power <= 1e-9:
            return False
        vocal_mask = (freqs >= 250.0) & (freqs <= 3800.0)
        vocal_power = float(np.sum(spec[:, vocal_mask]))
        vocal_ratio = vocal_power / total_power
        return vocal_ratio >= 0.12 or p90 >= 0.035

    def _recover_skipped_audio_regions(
        self,
        raw_segments: List[Any],
        audio_data: np.ndarray,
        duration: float,
        lang_arg: Optional[str],
        task: str,
        beam_size: int,
        need_word_timestamps: bool,
        prompt: Optional[str],
        sample_rate: int = 16000,
    ) -> List[Any]:
        """
        Inspects transcribed segments against the waveform.
        1. If the recording has 0 courtroom markers and Silero VAD / courtroom prompt suppressed > 45% of
           an active vocal recording (or censored lyrics with '***'), re-decodes cleanly without courtroom prompt bias.
        2. For any remaining untranscribed gaps >= 5.5s that contain active vocal energy (e.g., singing over
           heavy drums/bass or a soft-spoken witness filtered out by Silero VAD), decodes those windows with
           vad_filter=False and merges recovered segments chronologically.
        """
        if audio_data is None or len(audio_data) < sample_rate * 4 or duration <= 4.0:
            return raw_segments

        covered_dur = sum(max(0.0, float(s.end) - float(s.start)) for s in raw_segments)
        coverage_ratio = covered_dur / max(1.0, duration)
        combined_initial_text = " ".join(getattr(s, "text", "") for s in raw_segments)
        is_courtroom = bool(COURTROOM_MARKER_RE.search(combined_initial_text))
        has_asterisk_censorship = "*" in combined_initial_text

        HALLUCINATION_PHRASES = (
            "thank you for watching",
            "thanks for watching",
            "subtitles by",
            "amara.org",
            "please subscribe",
            "translated by",
        )

        # Case A: Non-courtroom audio (e.g. music / song / general media) where courtroom prompt or VAD
        # either censored words with '***' or dropped > 45% of active vocal audio on a file <= 15 mins
        if (not is_courtroom) and duration <= 900.0 and (has_asterisk_censorship or coverage_ratio < 0.55):
            if self._has_active_vocal_energy(audio_data, sample_rate):
                logger.info(
                    f"Non-courtroom vocal audio detected with low VAD coverage ({coverage_ratio*100:.1f}%) "
                    f"or prompt censorship (asterisks={has_asterisk_censorship}). "
                    f"Running full-coverage unprompted pass (vad_filter=False)..."
                )
                try:
                    rec_iter, _ = self.model.transcribe(
                        audio_data,
                        language=lang_arg,
                        task=task,
                        beam_size=max(2, beam_size),
                        temperature=0.0,
                        word_timestamps=need_word_timestamps,
                        vad_filter=False,
                        condition_on_previous_text=False,
                        repetition_penalty=1.15,
                        no_speech_threshold=0.90,
                        log_prob_threshold=-1.35,
                        compression_ratio_threshold=2.4,
                        initial_prompt=None,
                    )
                    full_rec = [s for s in rec_iter if getattr(s, "text", "").strip()]
                    rec_dur = sum(max(0.0, float(s.end) - float(s.start)) for s in full_rec)
                    if len(full_rec) >= len(raw_segments) or rec_dur > covered_dur * 1.15 or (has_asterisk_censorship and "*" not in " ".join(s.text for s in full_rec)):
                        raw_segments = full_rec
                        logger.info(f"Full-coverage unprompted pass recovered {len(raw_segments)} segments ({rec_dur:.1f}s / {duration:.1f}s).")
                except Exception as rec_err:
                    logger.warning(f"Full-coverage unprompted recovery note: {rec_err}")

        # Case B: Targeted Gap Recovery for any remaining untranscribed vocal windows >= 5.5s
        sorted_segs = sorted(raw_segments, key=lambda s: float(s.start))
        gaps = []
        cursor = 0.0
        for s in sorted_segs:
            st = float(s.start)
            et = float(s.end)
            if st - cursor >= 5.5:
                gaps.append((cursor, st))
            cursor = max(cursor, et)
        if duration - cursor >= 5.5:
            gaps.append((cursor, duration))

        if not gaps:
            return sorted_segs

        recovered_any = False
        all_segments = list(sorted_segs)
        # Only cap gap recovery count so multi-hour silent recesses don't trigger excess work
        checked_gaps = 0
        for (g_start, g_end) in gaps:
            if checked_gaps >= 24:
                break
            s_idx = max(0, int(g_start * sample_rate))
            e_idx = min(len(audio_data), int(g_end * sample_rate))
            gap_audio = audio_data[s_idx:e_idx]
            if not self._has_active_vocal_energy(gap_audio, sample_rate):
                continue

            checked_gaps += 1
            gap_dur = g_end - g_start
            logger.info(f"Recovering skipped active vocal region [{g_start:.1f}s -> {g_end:.1f}s] ({gap_dur:.1f}s) with vad_filter=False...")
            try:
                gap_iter, _ = self.model.transcribe(
                    gap_audio,
                    language=lang_arg,
                    task=task,
                    beam_size=max(2, beam_size),
                    temperature=0.0,
                    word_timestamps=need_word_timestamps,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    repetition_penalty=1.18,
                    no_speech_threshold=0.92,
                    log_prob_threshold=-1.35,
                    compression_ratio_threshold=2.4,
                    initial_prompt=prompt if is_courtroom else None,
                )
                for gseg in gap_iter:
                    gtxt = self._collapse_repeated_clauses(getattr(gseg, "text", "").strip())
                    if not gtxt:
                        continue
                    gtxt_low = gtxt.lower()
                    if any(hp in gtxt_low for hp in HALLUCINATION_PHRASES):
                        continue
                    avg_lp = getattr(gseg, "avg_logprob", 0.0) or 0.0
                    nsp = getattr(gseg, "no_speech_prob", 0.0) or 0.0
                    if avg_lp < -1.30 and nsp > 0.85:
                        continue

                    new_start = round(float(gseg.start) + g_start, 3)
                    new_end = round(min(g_end, float(gseg.end) + g_start), 3)
                    if new_end <= new_start:
                        continue

                    new_words = []
                    if getattr(gseg, "words", None):
                        for gw in gseg.words:
                            new_words.append(
                                SimpleNamespace(
                                    word=gw.word,
                                    start=round(float(gw.start) + g_start, 3),
                                    end=round(min(g_end, float(gw.end) + g_start), 3),
                                    probability=getattr(gw, "probability", None),
                                )
                            )
                    all_segments.append(
                        SimpleNamespace(
                            start=new_start,
                            end=new_end,
                            text=gtxt,
                            words=new_words,
                            no_speech_prob=nsp,
                            avg_logprob=avg_lp,
                        )
                    )
                    recovered_any = True
            except Exception as gap_err:
                logger.debug(f"Gap recovery note for [{g_start:.1f}s -> {g_end:.1f}s]: {gap_err}")

        if recovered_any:
            all_segments.sort(key=lambda s: float(s.start))
            logger.info(f"Audio completeness recovery finished: total segments now {len(all_segments)} (was {len(sorted_segs)}).")
        return all_segments

    @staticmethod
    def _collapse_repeated_clauses(text: str) -> str:
        """
        Collapses exact intra-segment repeated sentences/clauses caused by Whisper decoder loops
        on noisy audio (e.g. 'I would like to go through the process. I would like to go through the process.'
        -> 'I would like to go through the process.').
        """
        cleaned = text.strip()
        if not cleaned or len(cleaned) < 12:
            return cleaned
        # Split on sentence-ending punctuation while keeping delimiters
        parts = re.split(r'(?<=[.!?])\s+', cleaned)
        if len(parts) >= 2:
            deduped = [parts[0]]
            for p in parts[1:]:
                if p.strip().lower() != deduped[-1].strip().lower():
                    deduped.append(p)
            cleaned = " ".join(deduped).strip()
        # Also check exact half-string repetition without punctuation ('phrase phrase')
        words = cleaned.split()
        n = len(words)
        if n >= 6 and n % 2 == 0:
            half = n // 2
            if [w.lower().strip(".,!?") for w in words[:half]] == [w.lower().strip(".,!?") for w in words[half:]]:
                cleaned = " ".join(words[:half])
        return cleaned

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

        if BatchedInferencePipeline is not None and actual_device == "cuda":
            try:
                self.batched_model = BatchedInferencePipeline(model=self.model)
                logger.info(f"Initialized BatchedInferencePipeline (batch_size={WorkerConfig.WHISPER_BATCH_SIZE}) for high-speed multi-hour GPU transcription.")
            except Exception as bp_err:
                logger.warning(f"Could not initialize BatchedInferencePipeline ({bp_err}); using standard WhisperModel.")
                self.batched_model = None
        else:
            self.batched_model = None

        self.device = actual_device
        logger.info(f"Whisper model loaded successfully on {actual_device}:{device_index} (num_workers={self.num_workers}).")

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        target_language: Optional[str] = None,
        enable_translation: bool = False,
        mode: str = "fast",
        progress_callback = None,
        audio_data: Optional[Any] = None,
        need_word_timestamps: bool = True
    ) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()

        # Configure decoding based on transcription mode.
        # Avoid multi-step temperature cascade on noisy courtroom silence which causes 6x-30x slowdowns,
        # while preserving beam search for full word recall and segment boundary accuracy.
        beam_size = 1
        temperature: Any = 0.0
        if mode == "accurate":
            beam_size = 3
            temperature = 0.0
        elif mode == "balanced":
            beam_size = 2
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
            f"beam_size={beam_size}, temperature={temperature}, word_timestamps={need_word_timestamps}"
        )

        # Optimal VAD and decoding parameters for high accuracy & preserving courtroom speech + music/low-SNR vocals
        # threshold=0.18 captures quiet remarks ("Yes, My Lord"), distant witnesses, and vocals over music
        # while post-decoding repetition filters eliminate empty courtroom recess hallucinations
        vad_params = dict(
            threshold=0.18,
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
        audio_input = audio_data if audio_data is not None else str(audio_path)
        est_duration = (len(audio_data) / 16000.0) if (audio_data is not None and hasattr(audio_data, "__len__")) else 0.0
        use_batched = (self.batched_model is not None) and (est_duration >= 900.0)  # Use GPU batching for 15m - 10h recordings

        try:
            if use_batched:
                logger.info(
                    f"Using BatchedInferencePipeline (batch_size={WorkerConfig.WHISPER_BATCH_SIZE}) "
                    f"for long audio ({est_duration / 60.0:.1f} mins)..."
                )
                segments_iter, info = self.batched_model.transcribe(
                    audio_input,
                    language=lang_arg,
                    task=task,
                    beam_size=max(2, beam_size - 1) if est_duration >= 5400.0 else beam_size,
                    temperature=temperature,
                    word_timestamps=need_word_timestamps,
                    vad_filter=True,
                    vad_parameters=vad_params,
                    condition_on_previous_text=False,
                    repetition_penalty=1.15,
                    no_speech_threshold=0.75,
                    compression_ratio_threshold=2.4,
                    initial_prompt=prompt,
                    batch_size=WorkerConfig.WHISPER_BATCH_SIZE,
                )
            else:
                segments_iter, info = self.model.transcribe(
                    audio_input,
                    language=lang_arg,
                    task=task,
                    beam_size=beam_size,
                    temperature=temperature,
                    word_timestamps=need_word_timestamps,
                    vad_filter=True,
                    vad_parameters=vad_params,
                    condition_on_previous_text=False,
                    repetition_penalty=1.15,
                    no_speech_threshold=0.75,
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
            logger.warning(f"Whisper transcribe encountered error ({unbatched_err}); reloading WhisperModel and retrying...")
            gc.collect()
            if torch and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(0)
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            self.load_model()
            segments_iter, info = self.model.transcribe(
                audio_input,
                language=lang_arg,
                task=task,
                beam_size=2,
                temperature=0.0,
                word_timestamps=need_word_timestamps,
                vad_filter=True,
                vad_parameters=vad_params,
                condition_on_previous_text=False,
                repetition_penalty=1.15,
                no_speech_threshold=0.75,
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

        # Recover any vocal/music/speech regions skipped by Silero VAD or suppressed by courtroom prompt
        try:
            waveform_for_recovery = audio_data
            if waveform_for_recovery is None and audio_path and Path(audio_path).exists():
                import soundfile as sf
                w_np, _ = sf.read(str(audio_path), dtype="float32")
                if len(w_np.shape) > 1:
                    w_np = w_np.mean(axis=1)
                waveform_for_recovery = np.ascontiguousarray(w_np, dtype=np.float32)
            if waveform_for_recovery is not None:
                raw_segments = self._recover_skipped_audio_regions(
                    raw_segments=raw_segments,
                    audio_data=waveform_for_recovery,
                    duration=duration,
                    lang_arg=lang_arg or detected_language,
                    task=task,
                    beam_size=beam_size,
                    need_word_timestamps=need_word_timestamps,
                    prompt=prompt,
                )
        except Exception as rec_outer_err:
            logger.debug(f"Audio completeness recovery skipped: {rec_outer_err}")

        segments_list = []
        full_text_parts = []
        total_words = 0
        prev_norm_key = None
        consecutive_dup_count = 0
        dropped_hallucinations = 0

        for seg in raw_segments:
            text = self._collapse_repeated_clauses(seg.text.strip())
            if not text:
                continue

            # Detect consecutive hallucinated recess loops (e.g. "Can you read?" repeated 24x on courtroom background noise)
            norm_key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
            no_speech_prob = getattr(seg, "no_speech_prob", 0.0) or 0.0
            avg_logprob = getattr(seg, "avg_logprob", 0.0) or 0.0

            if norm_key and norm_key == prev_norm_key:
                consecutive_dup_count += 1
                # Drop repeated identical segments when repeated >= 2 times in a row (or immediately if low confidence / high no_speech_prob)
                if consecutive_dup_count >= 2 or no_speech_prob > 0.45 or avg_logprob < -0.75:
                    dropped_hallucinations += 1
                    continue
            else:
                prev_norm_key = norm_key
                consecutive_dup_count = 0

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
                if not words_data:
                    total_words += len(seg_dict["text"].split())
            segments_list.append(seg_dict)

        if dropped_hallucinations > 0:
            logger.info(f"Filtered {dropped_hallucinations} consecutive hallucinated/repeated recess segments prior to alignment.")

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
