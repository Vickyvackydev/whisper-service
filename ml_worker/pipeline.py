import gc
import re
import time
import logging
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
try:
    import torch
except ImportError:
    torch = None
from ml_worker.config import WorkerConfig
from ml_worker.audio import safe_download_audio, convert_to_wav_16k_mono, cleanup_file
from ml_worker.transcriber import Transcriber
from ml_worker.diarizer import SpeakerDiarizer
from ml_worker.normalizer import normalize_segment, normalize_word_stream

logger = logging.getLogger("ml_worker.pipeline")

def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

class InferencePipeline:
    def __init__(self, transcriber: Transcriber, diarizer: SpeakerDiarizer):
        self.transcriber = transcriber
        self.diarizer = diarizer
        self._whisper_lock = threading.Lock()
        self._align_lock = threading.Lock()
        self._diarize_lock = threading.Lock()
        self._align_models: Dict[str, Tuple[Any, Any]] = {}
        WorkerConfig.ensure_scratch_dir()

    def _get_cached_align_model(self, lang_code: str, align_device: str):
        import whisperx
        key = f"{lang_code}:{align_device}"
        if key not in self._align_models:
            logger.info(f"Loading & caching WhisperX Wav2Vec2 alignment model for language={lang_code} on {align_device}...")
            align_model, align_metadata = whisperx.load_align_model(
                language_code=lang_code,
                device=align_device
            )
            self._align_models[key] = (align_model, align_metadata)
        return self._align_models[key]

    def process(
        self,
        job_id: str,
        audio_url: str,
        enable_diarization: bool = True,
        enable_translation: bool = False,
        language: Optional[str] = None,
        target_language: Optional[str] = None,
        transcription_mode: str = "fast",
        progress_updater: Optional[Callable[[str, int], None]] = None,
        prefetched_wav: Optional[Tuple[Path, float]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        raw_download_path: Optional[Path] = None
        wav_path: Optional[Path] = None
        audio_arr: Optional[Any] = None

        try:
            if prefetched_wav is not None and prefetched_wav[0].exists():
                wav_path, audio_duration = prefetched_wav
                if progress_updater:
                    progress_updater("preprocessing", 30)
                logger.info(f"[{job_id}] Using pre-downloaded & normalized 16kHz WAV ({audio_duration:.2f}s): {wav_path}")
            else:
                # 1. Downloading
                if progress_updater:
                    progress_updater("downloading", 10)
                logger.info(f"[{job_id}] Downloading audio from: {audio_url}")
                raw_download_path = safe_download_audio(audio_url, WorkerConfig.SCRATCH_DIR)

                # 2. Preprocessing / ffmpeg normalization
                if progress_updater:
                    progress_updater("preprocessing", 25)
                logger.info(f"[{job_id}] Normalizing audio to 16kHz mono WAV...")
                wav_path, audio_duration = convert_to_wav_16k_mono(raw_download_path, WorkerConfig.SCRATCH_DIR)

                # Clean raw download immediately to save disk
                cleanup_file(raw_download_path)
                raw_download_path = None

            # Load 16kHz mono float32 waveform into RAM once so Whisper, WhisperX, and NeMo all reuse it without extra disk reads or FFmpeg subprocesses
            try:
                import soundfile as sf
                import numpy as np
                data_np, sr = sf.read(str(wav_path), dtype="float32")
                if len(data_np.shape) > 1:
                    data_np = data_np.mean(axis=1)
                audio_arr = np.ascontiguousarray(data_np, dtype=np.float32)
            except Exception as load_err:
                logger.debug(f"[{job_id}] In-memory WAV load note ({load_err}); stages will read from {wav_path}")
                audio_arr = None

            # 3. Whisper Transcription (guarded by _whisper_lock so Stage 1 can run concurrently while another job is in Stage 2 Alignment/Diarization)
            if progress_updater:
                progress_updater("transcribing", 40)
            logger.info(f"[{job_id}] Running Whisper inference (mode={transcription_mode}, enable_translation={enable_translation}, lang={language}, target={target_language})...")

            def transcribe_progress_cb(current_sec, total_sec):
                if total_sec > 0 and progress_updater:
                    pct = 40 + int((current_sec / total_sec) * 35) # 40% to 75%
                    progress_updater("transcribing", min(75, pct))

            # Skip redundant CTranslate2 DTW word_timestamps pass when WhisperX Wav2Vec2 phoneme alignment will overwrite word timestamps
            use_whisperx_align = getattr(WorkerConfig, "FORCE_WHISPERX_ALIGNMENT", True)
            with self._whisper_lock:
                whisper_result = self.transcriber.transcribe(
                    wav_path,
                    language=language,
                    target_language=target_language,
                    enable_translation=enable_translation,
                    mode=transcription_mode,
                    progress_callback=transcribe_progress_cb,
                    audio_data=audio_arr,
                    need_word_timestamps=not use_whisperx_align
                )

            segments = whisper_result["segments"]
            num_speakers = 1

            # 4. Stage 3 & 4: WhisperX Phoneme Forced Alignment + Speaker Diarization Stitching
            align_device = "cuda" if (torch and torch.cuda.is_available() and WorkerConfig.WHISPER_DEVICE != "cpu") else "cpu"
            aligned_result = {"segments": segments}

            if segments and len(segments) > 0:
                lang_code = whisper_result.get("language") or "en"
                try:
                    if progress_updater:
                        progress_updater("aligning", 78)
                    logger.info(f"[{job_id}] Running WhisperX Wav2Vec2 phoneme forced alignment (language={lang_code}, device={align_device})...")
                    import whisperx
                    align_audio = audio_arr if audio_arr is not None else whisperx.load_audio(str(wav_path))
                    with self._align_lock:
                        if align_device == "cuda" and torch and torch.cuda.is_available():
                            try:
                                torch.cuda.set_device(0)
                            except Exception:
                                pass
                        align_model, align_metadata = self._get_cached_align_model(lang_code, align_device)
                        aligned_result = whisperx.align(
                            segments,
                            align_model,
                            align_metadata,
                            align_audio,
                            align_device,
                            return_char_alignments=False
                        )

                    logger.info(f"[{job_id}] Phoneme forced alignment successful across {len(aligned_result.get('segments', []))} segments.")
                except Exception as align_err:
                    logger.warning(f"[{job_id}] WhisperX phoneme alignment skipped or failed ({align_err}). Retaining Whisper timestamps.")
                    aligned_result = {"segments": segments}

            # Diarization & Word Midpoint Stitching
            # Pre-populate proportional word timestamps for any segment where WhisperX Wav2Vec2 alignment logged
            # 'backtrack failed, resorting to original' BEFORE calling whisperx.assign_word_speakers() so every
            # fallback segment still receives full word-level speaker diarization!
            align_segs = aligned_result.get("segments", [])
            total_align_segs = len(align_segs)
            failed_align_segs = 0
            for s in align_segs:
                if not s.get("words"):
                    failed_align_segs += 1
                    raw_txt = str(s.get("text", "")).strip()
                    if raw_txt:
                        toks = raw_txt.split()
                        s_start = round(float(s.get("start", 0.0)), 3)
                        s_end = round(float(s.get("end", s_start + 0.1)), 3)
                        dur = max(0.1, s_end - s_start)
                        step = dur / max(1, len(toks))
                        s["words"] = [
                            {
                                "word": tok,
                                "start": round(s_start + idx * step, 3),
                                "end": round(s_start + (idx + 1) * step, 3),
                                "score": None,
                            }
                            for idx, tok in enumerate(toks)
                        ]
            logger.info(f"[{job_id}] [DIAGNOSTIC] Alignment status: {total_align_segs - failed_align_segs}/{total_align_segs} segments aligned via Wav2Vec2 ({failed_align_segs} synthesized pre-diarization fallback words).")

            logger.info(f"[{job_id}] Diarization check: requested={enable_diarization}, model_loaded={self.diarizer.is_loaded}")
            if self.diarizer.is_loaded and enable_diarization:
                if progress_updater:
                    progress_updater("diarizing", 85)
                logger.info(f"[{job_id}] Running {self.diarizer.backend.upper()} diarization on {wav_path}...")
                with self._diarize_lock:
                    diarize_df = self.diarizer.diarize_dataframe(
                        wav_path,
                        audio_data=audio_arr,
                        sample_rate=16000
                    )
                logger.info(f"[{job_id}] Diarization generated {len(diarize_df)} intervals.")

                if len(diarize_df) > 0:
                    try:
                        # 1. Dump raw diarization intervals for the first 35 seconds
                        if hasattr(diarize_df, "iterrows"):
                            early_df = diarize_df[diarize_df["start"] <= 35.0] if "start" in diarize_df.columns else diarize_df.head(15)
                            diar_intervals = "\n".join([
                                f"    [{row.get('start', 0.0):.2f}s -> {row.get('end', 0.0):.2f}s] {row.get('speaker', 'UNKNOWN')}"
                                for _, row in early_df.iterrows()
                            ])
                            logger.info(f"[{job_id}] [DIAGNOSTIC 1: Raw {self.diarizer.backend.upper()} intervals (first 35s)]:\n{diar_intervals}")

                        import whisperx
                        stitched = whisperx.assign_word_speakers(diarize_df, aligned_result, fill_nearest=True)

                        # 2. Dump raw word speakers directly after whisperx.assign_word_speakers
                        raw_early_words = []
                        for s in stitched.get("segments", []):
                            for w in s.get("words", []):
                                if float(w.get("start", 0.0)) <= 35.0:
                                    raw_early_words.append(f"{w.get('word', '')}[{w.get('speaker', 'NONE')}]")
                        logger.info(f"[{job_id}] [DIAGNOSTIC 2: Raw Word Speakers from WhisperX (first 35s)]:\n" + " ".join(raw_early_words[:80]))

                        segments, num_speakers = self.reconstruct_speaker_turns(
                            stitched,
                            audio_data=audio_arr,
                            sample_rate=16000
                        )

                        # 3. Dump final reconstructed turns for first 35 seconds
                        final_early_turns = []
                        for s in segments:
                            if float(s.get("start", 0.0)) <= 35.0:
                                final_early_turns.append(f"    [{s.get('start', 0.0):.2f}s -> {s.get('end', 0.0):.2f}s] {s.get('speaker')}: \"{s.get('text', '')}\"")
                        logger.info(f"[{job_id}] [DIAGNOSTIC 3: Final Reconstructed Turns (first 35s)]:\n" + "\n".join(final_early_turns))

                        logger.info(f"[{job_id}] Speaker alignment complete: {num_speakers} unique speakers, {len(segments)} turns.")
                    except ImportError as ix_err:
                        logger.warning(f"[{job_id}] whisperx is not installed in the worker's environment ({ix_err}). Falling back to standard turn assignment.")
                        turns = diarize_df.to_dict(orient="records") if hasattr(diarize_df, "to_dict") else []
                        segments, num_speakers = self.diarizer.assign_speakers(aligned_result.get("segments", segments), turns)
                else:
                    logger.warning(f"[{job_id}] Diarization produced 0 turns. Defaulting to SPEAKER_00.")
                    segments = aligned_result.get("segments", segments)
                    num_speakers = 1
            else:
                segments = aligned_result.get("segments", segments)
                num_speakers = 1

            # 5. Finalizing Result
            if progress_updater:
                progress_updater("finalizing", 95)

            end_time = time.time()
            processing_time = end_time - start_time

            final_text = " ".join(seg["text"] for seg in segments if seg.get("text")).strip() if segments else whisper_result["transcribed_text"]
            final_word_count = sum(len(seg.get("words", [])) for seg in segments) if segments else whisper_result["word_count"]

            formatted_result = {
                "segments": segments,
                "language": whisper_result["language"],
                "transcribed_text": final_text,
                "source_transcribed_text": whisper_result.get("source_transcribed_text"),
                "transcription_mode": transcription_mode,
                "model_used": whisper_result["model_used"],
                "duration": round(audio_duration, 3),
                "duration_formatted": format_duration(audio_duration),
                "num_speakers": num_speakers,
                "processing_time": round(processing_time, 4),
                "processing_time_formatted": f"{processing_time:.1f}s",
                "word_count": final_word_count,
                "target_language": whisper_result.get("target_language"),
                "translation_method": whisper_result.get("translation_method")
            }

            logger.info(f"[{job_id}] Transcription completed successfully in {processing_time:.2f}s (Audio duration: {audio_duration:.2f}s, RTF: {processing_time/audio_duration:.3f})")
            return formatted_result

        finally:
            # Guarantee scratch file cleanup and release pipeline audio buffer
            cleanup_file(raw_download_path)
            cleanup_file(wav_path)
            if audio_arr is not None:
                del audio_arr
            gc.collect()

    @staticmethod
    def _split_segment_runs(clean_words: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Splits contiguous words inside a single Whisper segment into distinct speaker sub-chunks
        when a genuine interjection/handoff occurs (e.g. cross-talk without pause).
        Filters out single-word acoustic jitter (< 0.35s) so natural sentences aren't fractured.
        """
        if not clean_words:
            return []

        def _w_clean(w_dict: Dict[str, Any]) -> str:
            return str(w_dict.get("word", "")).lower().strip(".,!?;:\"'()[]{} ")

        def _ends_punct(w_dict: Dict[str, Any]) -> bool:
            txt = str(w_dict.get("word", "")).strip()
            return any(txt.endswith(p) for p in (".", "?", "!"))

        FORMULA_END_WORDS = {"pleases", "grateful", "obliged", "lord", "lordship", "honour", "honor", "sir"}
        PURE_BC_WORDS = {
            "yes", "no", "yeah", "nope", "sure", "correct", "sir", "true",
            "milord", "pleases", "grateful", "okay", "ok", "sorry", "pardon"
        }

        def _find_bc_span(word_list: List[Dict[str, Any]]) -> List[int]:
            toks = [_w_clean(w) for w in word_list]
            n_t = len(toks)
            bc_idx = set()
            for idx_t, t in enumerate(toks):
                if t in PURE_BC_WORDS:
                    bc_idx.add(idx_t)
                elif t == "my" and idx_t + 1 < n_t and toks[idx_t + 1] in ("lord", "lordship", "ladyship"):
                    bc_idx.update((idx_t, idx_t + 1))
                elif t in ("lord", "lordship", "ladyship") and idx_t > 0 and toks[idx_t - 1] in ("my", "your"):
                    bc_idx.update((idx_t - 1, idx_t))
                elif t in ("i'm", "am") and idx_t + 1 < n_t and toks[idx_t + 1] == "grateful":
                    bc_idx.update((idx_t, idx_t + 1))
                elif t == "as" and idx_t + 3 < n_t and toks[idx_t : idx_t + 4] == ["as", "the", "court", "pleases"]:
                    bc_idx.update((idx_t, idx_t + 1, idx_t + 2, idx_t + 3))
            return sorted(bc_idx)

        # 0. Pre-snap speaker transitions inside clean_words to nearest preceding sentence/formula boundary (1-6 words back)
        #    Fixes NeMo Sortformer's ~250ms onset latency when a new speaker cuts in immediately after a period.
        n_w = len(clean_words)
        for k in range(1, n_w):
            spk_prev = clean_words[k - 1].get("speaker", "SPEAKER_00")
            spk_curr = clean_words[k].get("speaker", "SPEAKER_00")
            if spk_prev != spk_curr and not _ends_punct(clean_words[k - 1]):
                snap_idx = None
                for lookback in range(1, min(7, k + 1)):
                    cand_prev = clean_words[k - lookback - 1] if (k - lookback - 1) >= 0 else None
                    if cand_prev is not None:
                        cand_txt = _w_clean(cand_prev)
                        if _ends_punct(cand_prev) or cand_txt in FORMULA_END_WORDS:
                            mid_slice = clean_words[k - lookback : k]
                            if not any(_ends_punct(mw) for mw in mid_slice):
                                snap_idx = k - lookback
                                break
                if snap_idx is not None:
                    for m_idx in range(snap_idx, k):
                        clean_words[m_idx]["speaker"] = spk_curr

        # 1. Group contiguous words into initial speaker runs
        runs: List[Dict[str, Any]] = []
        for w in clean_words:
            w_spk = w.get("speaker", "SPEAKER_00")
            if not runs or runs[-1]["speaker"] != w_spk:
                runs.append({"speaker": w_spk, "words": [w]})
            else:
                runs[-1]["words"].append(w)

        # 1b. Isolate sandwiched backchannels (e.g. "the order you asked" | "for yes sir does" | "not talk about...")
        if len(runs) >= 3:
            for r_i in range(1, len(runs) - 1):
                prev_r = runs[r_i - 1]
                curr_r = runs[r_i]
                next_r = runs[r_i + 1]
                if prev_r["speaker"] == next_r["speaker"] and curr_r["speaker"] != prev_r["speaker"]:
                    c_words = curr_r["words"]
                    if 1 <= len(c_words) <= 5:
                        bc_indices = _find_bc_span(c_words)
                        if bc_indices:
                            first_bc = bc_indices[0]
                            last_bc = bc_indices[-1]
                            lead_non_bc = c_words[:first_bc]
                            core_bc = c_words[first_bc : last_bc + 1]
                            trail_non_bc = c_words[last_bc + 1 :]
                            if lead_non_bc:
                                for lw in lead_non_bc:
                                    lw["speaker"] = prev_r["speaker"]
                                prev_r["words"].extend(lead_non_bc)
                            if trail_non_bc:
                                for tw in trail_non_bc:
                                    tw["speaker"] = next_r["speaker"]
                                next_r["words"] = trail_non_bc + next_r["words"]
                            curr_r["words"] = core_bc
            runs = [r for r in runs if r["words"]]

        STANDALONE_AFFIRMATIONS = {"yes", "no", "yeah", "nope", "sure", "correct", "sir", "true"}

        # 2. Smooth single-word acoustic jitter (< 0.35s)
        changed = True
        while changed and len(runs) > 1:
            changed = False
            new_runs = []
            i = 0
            while i < len(runs):
                r = runs[i]
                r_dur = r["words"][-1]["end"] - r["words"][0]["start"]
                r_len = len(r["words"])
                first_w_txt = r["words"][0]["word"].lower().strip(".,!?;:\"' ")

                # Intentional short response / affirmation
                is_valid_standalone = False
                if r_len == 1 and first_w_txt in STANDALONE_AFFIRMATIONS:
                    is_valid_standalone = True

                # Preceded by a natural pause or closing punctuation
                if i > 0:
                    prev_last_w = runs[i-1]["words"][-1]
                    gap = r["words"][0]["start"] - prev_last_w["end"]
                    ends_clause = any(prev_last_w["word"].endswith(p) for p in (".", "?", "!"))
                    if ends_clause or gap >= 0.30:
                        is_valid_standalone = True

                # If 1-word island and NOT an intentional standalone, absorb as jitter
                if r_len == 1 and r_dur < 0.35 and not is_valid_standalone:
                    if i > 0 and i < len(runs) - 1 and runs[i-1]["speaker"] == runs[i+1]["speaker"]:
                        # Middle jitter: absorb into surrounding speaker
                        prev_spk = runs[i-1]["speaker"]
                        for w in r["words"]:
                            w["speaker"] = prev_spk
                        new_runs[-1]["words"].extend(r["words"])
                        changed = True
                        i += 1
                        continue
                    elif i == 0 and len(runs) > 1:
                        # Leading jitter
                        next_spk = runs[1]["speaker"]
                        for w in r["words"]:
                            w["speaker"] = next_spk
                        runs[1]["words"] = r["words"] + runs[1]["words"]
                        changed = True
                        i += 1
                        continue
                    elif i == len(runs) - 1 and len(new_runs) > 0:
                        # Trailing jitter
                        prev_spk = new_runs[-1]["speaker"]
                        for w in r["words"]:
                            w["speaker"] = prev_spk
                        new_runs[-1]["words"].extend(r["words"])
                        changed = True
                        i += 1
                        continue

                new_runs.append(r)
                i += 1

            # Merge consecutive identical speakers in new_runs
            merged = []
            for r in new_runs:
                if not merged or merged[-1]["speaker"] != r["speaker"]:
                    merged.append(r)
                else:
                    merged[-1]["words"].extend(r["words"])
            runs = merged

        return [(r["speaker"], r["words"]) for r in runs]

    def reconstruct_speaker_turns(
        self,
        stitched_result: Dict[str, Any],
        audio_data: Optional[Any] = None,
        sample_rate: int = 16000
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Takes WhisperX stitched word-level speaker assignments and builds
        clean, conversational speaker turns while strictly preserving Whisper's
        ground-truth segment boundaries.

        Why this architecture:
        - Whisper's Voice Activity Detector (VAD) already produces near-flawless
          utterance segmentation, cleanly separating turns between speakers.
        - Previous word-flattening heuristics destroyed these boundaries, gluing
          distinct speakers together into 12-second blobs while splitting single
          sentences in half over natural 1-second pauses.
        - By treating Whisper's segments as ground-truth speech units and performing
          robust segment-level speaker attribution + conversational legal smoothing,
          we achieve clean, readable transcripts with accurate speaker attribution.
        """
        raw_segments = stitched_result.get("segments", [])
        if not raw_segments:
            return [], 0

        import re

        # Universal Commonwealth / Common-Law Courtroom Honorifics (spoken BY Counsel / Litigants TO the Bench)
        HONORIFIC_LEGAL_COUNSEL_PATTERNS = [
            r"\bmy\s+lord\b", r"\bmilord\b", r"\bme\s+lord\b", r"\bmy\s+noble\s+lord\b",
            r"\byour\s+lordship\b", r"\byour\s+lordships\b", r"\byour\s+honou?r\b",
            r"\byour\s+ladyship\b", r"\byour\s+worship\b",
            r"\bas\s+the\s+court\s+pleases\b", r"\bas\s+your\s+lordship\s+pleases\b",
            r"\bif\s+your\s+lordship\s+pleases\b", r"\bcourt\s+pleases\b",
            r"\bas\s+the\s+court\s+deems\s+fit\b", r"\bmay\s+it\s+please\s+the\s+court\b",
            r"\bmost\s+obliged\b", r"\bmost\s+grateful\b"
        ]
        re_counsel_honorifics = [re.compile(p, re.IGNORECASE) for p in HONORIFIC_LEGAL_COUNSEL_PATTERNS]

        # Universal Judicial Authority / Bench Language (spoken BY the Judge / Magistrate)
        JUDGE_AUTHORITY_PATTERNS = [
            r"\blearned\s+counsel\b", r"\blearned\s+silk\b", r"\blearned\s+friend\b",
            r"\bthis\s+court\s+(?:holds?|orders?|rules?|finds?|hereby)\b",
            r"\bthe\s+order\s+of\s+(?:the\s+)?court\b",
            r"\bthe\s+court\s+will\b",
            r"\bcase\s+is\s+(?:adjourned|struck\s+out|dismissed)\b",
            r"\bstand\s+(?:down|over)\b",
            r"\bwhat\s+is\s+your\s+(?:application|submission|response|prayer)\b",
            r"\baddress\s+(?:the\s+court|me)\s+on\b",
            r"\bwhat\s+did\s+(?:he|she|they|the\s+judge)\s+say\b",
            r"\bcan\s+i\s+see\s+the\b", r"\blet\s+me\s+see\s+the\b",
            r"\bwhat\s+are\s+you\s+(?:saying|talking\s+about|asking\s+for)\b",
            r"\bi\s+don't\s+like\s+this\s+idea\s+of\b",
            r"\bis\s+that\s+what\s+(?:you|was)\s+granted\b",
            r"\byou\s+may\s+call\s+the\b",
            r"\bcall\s+the\s+(?:first\s+|next\s+)?(?:matter|case|suit|cause)\b",
            r"\byou\s+may\s+(?:go\s+ahead|proceed)\b",
            r"\bchoose\s+the\s+one\s+you\s+want\b",
            r"\bwhat\s+you\s+asked\s+for\s+is\s+what\b",
            r"\bthe\s+order\s+you\s+asked\s+for\b",
            r"\b(?:there'?s|is\s+there)\s+(?:no|a)\s+subsequent\s+process\b",
            r"\bwhy\s+did\s+you\s+say\b",
            r"\bcounsel\s+saying\s+the\s+judge\s+said\b",
            r"\bit'?s\s+(?:not\s+fair|a\s+new\s+legal\s+year)\b",
            r"\bplease,\s+mr\.?\s+[a-z]+\b",
            r"\bmr\.?\s+[a-z]+,\s+it'?s\b",
            r"\b(?:does\s+not|doesn'?t)\s+talk\s+about\b",
            r"\bit\s+identifies\s+the\s+processes\b",
            r"\bgranted\s+as\s+prayed\b",
            r"\bmotion\s+(?:is\s+)?granted\b",
            r"\bfile\s+a\s+motion\b",
            r"\bput\s+it\s+in(?:to)?\s+writing\b",
            r"\bmake\s+your\s+order\b",
            r"\b(?:this\s+|the\s+)?(?:suit|matter|case)\s+is\s+adjourned\b",
            r"\badjourned\s+to\s+the\b",
            r"\bwe\s+shall\s+go\s+for\s+a\s+short\s+break\b",
            r"\byou\s+have\s+only\s+\d+\s+minutes\b",
            r"\bcontinue\s+with\s+the\s+cross[\s-]?examination\b",
            r"\badmitted\s+and\s+marked\s+as\b",
            r"\bthe\s+objection\s+is\s+noted\b",
            r"\bthe\s+question\s+calls\s+for\s+speculation\b",
            r"\bconfine\s+yourself\s+to\b",
            r"\byou\s+have\s+asked\s+that\s+question\s+already\b",
            r"\bavoid\s+repetition\b",
            r"\bwitness\s+is\s+discharged\b",
            r"\bphotograph\s+of\s+the\s+bailiff\b"
        ]
        re_judge_authority = [re.compile(p, re.IGNORECASE) for p in JUDGE_AUTHORITY_PATTERNS]

        # Universal Judicial Directives & Courtroom Management (exclusively spoken by the Presiding Judge)
        RE_JUDGE_DIRECTIVES = re.compile(
            r"\b(?:"
            r"(?:he|she|it|they)\s+will\s+not\s+pay\s+(?:for\s+your\s+rent|school\s+fees|medical|maintenance|upkeep|you)|"
            r"not\s+to\s+touch\s+(?:school\s+fees|medical|maintenance|upkeep)|"
            r"(?:have|did)\s+i\s+given?\s+my\s+(?:judgement|ruling|decision|order)|"
            r"give\s+me\s+judgement|"
            r"for\s+upkeep\s+you\s+don't\s+need|"
            r"why\s+are\s+you\s+here\??|"
            r"(?:just\s+)?keep\s+quiet|"
            r"i\s+said\s+keep\s+quiet|"
            r"watch\s+what\s+you\s+are\s+saying|"
            r"let's\s+resolve\s+this|"
            r"let's\s+get\s+to\s+the\s+root\s+of\s+this|"
            r"stop\s+interjecting(?:\s+(?:him|her|them))?|"
            r"let\s+(?:him|her|them)\s+(?:finish|speak|tell\s+me)|"
            r"let\s+me\s+listen\s+to\s+(?:him|her|them)|"
            r"send\s+you\s+out\s+of\s+the\s+courtroom|"
            r"who\s+was\s+paying\s+(?:the\s+)?(?:school\s+fees|rent|medical|upkeep)|"
            r"that\s+is\s+not\s+your\s+problem|"
            r"let\s+(?:her|him|them)\s+round\s+up\s+this\s+session|"
            r"start\s+the\s+next\s+session\s+from\s+that\s+school|"
            r"judge\s+in\s+your\s+own\s+(?:cause|course)"
            r")\b",
            re.IGNORECASE
        )

        RE_SUBMISSION = re.compile(r"\b(?:as\s+(?:the|your\s+lordship's?)\s+court\s+pleases|as\s+the\s+court\s+pleases|court\s+pleases)\b", re.IGNORECASE)
        RE_CLARIFICATION = re.compile(r"^(?:sir\??|my\s+lord\??|pardon\??|sorry\??)$", re.IGNORECASE)
        RE_BENCH_INQUIRY = re.compile(r"\b(?:can\s+i\s+see\s+the|let\s+me\s+see\s+the|show\s+me\s+the|where\s+is\s+the\s+(?:order|prayer|writ|suit|process))\b", re.IGNORECASE)
        RE_COUNSEL_ARGUMENT = re.compile(r"\b(?:what\s+i'm\s+saying|what\s+i\s+am\s+saying|what\s+we\s+are\s+saying|our\s+submission|our\s+prayer|our\s+application|we\s+are\s+asking|we\s+asked\s+for|we\s+prayed|we\s+filed|we\s+served|we\s+said)\b", re.IGNORECASE)
        DANGLING_CONJUNCTIONS = {"and", "or", "because", "which", "that", "but", "whether"}

        processed_segments: List[Dict[str, Any]] = []

        # -------------------------------------------------------------------
        # STAGE 1: Process Each Segment Intact (Preserving Natural Utterances)
        # -------------------------------------------------------------------
        for seg in raw_segments:
            seg_start = round(float(seg.get("start", 0.0)), 3)
            seg_end = round(float(seg.get("end", seg_start + 0.1)), 3)
            raw_text = str(seg.get("text", "")).strip()
            words = seg.get("words", [])

            # Fallback if WhisperX alignment did not produce word timestamps
            if not words and raw_text:
                tokens = raw_text.split()
                dur = max(0.1, seg_end - seg_start)
                step = dur / max(1, len(tokens))
                words = [
                    {
                        "word": tok,
                        "start": round(seg_start + idx * step, 3),
                        "end": round(seg_start + (idx + 1) * step, 3),
                        "speaker": seg.get("speaker") or "SPEAKER_00",
                        "score": None,
                    }
                    for idx, tok in enumerate(tokens)
                ]

            clean_words = []
            for w in words:
                w_txt = str(w.get("word", "")).strip()
                if not w_txt:
                    continue
                w_start = round(float(w.get("start", seg_start)), 3)
                w_end = round(float(w.get("end", w_start + 0.05)), 3)
                if w_end < w_start:
                    w_end = w_start + 0.05
                clean_words.append({
                    "word": w_txt,
                    "start": w_start,
                    "end": w_end,
                    "speaker": str(w.get("speaker") or seg.get("speaker") or "SPEAKER_00"),
                    "score": round(float(w.get("score")), 3) if w.get("score") is not None else None,
                })

            if not clean_words and not raw_text:
                continue

            # Pre-normalize phonetic courtroom phrases ("as a couple places" -> "As the Court pleases.", etc.)
            # BEFORE _split_segment_runs so formula end-tokens ("pleases.") are recognized during boundary snapping!
            if clean_words:
                clean_words = normalize_word_stream(clean_words)

            # Calculate dominant speaker for this segment based on word durations
            spk_durations: Dict[str, float] = {}
            for w in clean_words:
                spk = w["speaker"]
                dur = max(0.01, w["end"] - w["start"])
                spk_durations[spk] = spk_durations.get(spk, 0.0) + dur

            dominant_spk = max(spk_durations, key=spk_durations.get) if spk_durations else (seg.get("speaker") or "SPEAKER_00")

            # Intra-segment speaker handoffs (split only when genuine multi-word turns occur)
            sub_chunks = self._split_segment_runs(clean_words)

            if len(sub_chunks) <= 1:
                # Intact segment: assign dominant speaker to all words
                seg_spk = sub_chunks[0][0] if sub_chunks else dominant_spk
                for w in clean_words:
                    w["speaker"] = seg_spk
                processed_segments.append({
                    "start": clean_words[0]["start"] if clean_words else seg_start,
                    "end": clean_words[-1]["end"] if clean_words else seg_end,
                    "text": " ".join(cw["word"] for cw in clean_words).strip() if clean_words else raw_text,
                    "source_text": seg.get("source_text"),
                    "speaker": seg_spk,
                    "words": clean_words,
                })
            else:
                for chunk_spk, chunk_words in sub_chunks:
                    chunk_text = " ".join(cw["word"] for cw in chunk_words).strip()
                    processed_segments.append({
                        "start": chunk_words[0]["start"],
                        "end": chunk_words[-1]["end"],
                        "text": chunk_text,
                        "source_text": None,
                        "speaker": chunk_spk,
                        "words": chunk_words,
                    })

        if not processed_segments:
            return [], 0

        # Pre-normalize multi-word phonetic phrases across segment boundaries first
        # (e.g. "as" [SPEAKER_00] | "a copy" [SPEAKER_01] -> "As the Court pleases." [SPEAKER_00],
        #  "Ankom" | "proceedings" -> "ongoing" | "proceedings")
        flat_stream: List[Dict[str, Any]] = []
        for s_idx, seg in enumerate(processed_segments):
            for w in seg.get("words", []):
                w_copy = dict(w)
                w_copy["_seg_idx"] = s_idx
                flat_stream.append(w_copy)
        if flat_stream:
            flat_stream = normalize_word_stream(flat_stream)
            regrouped_segments: List[Dict[str, Any]] = []
            for w in flat_stream:
                w_spk = w.get("speaker", "SPEAKER_00")
                w_sidx = w.pop("_seg_idx", None)
                if not regrouped_segments:
                    regrouped_segments.append({
                        "start": w["start"],
                        "end": w["end"],
                        "speaker": w_spk,
                        "text": w["word"],
                        "source_text": None,
                        "words": [w],
                        "_orig_sidx": w_sidx,
                    })
                else:
                    prev_rg = regrouped_segments[-1]
                    if prev_rg["speaker"] == w_spk and (prev_rg.get("_orig_sidx") == w_sidx or w_sidx is None):
                        prev_rg["words"].append(w)
                        prev_rg["end"] = w["end"]
                        prev_rg["text"] = " ".join(x["word"] for x in prev_rg["words"]).strip()
                    else:
                        regrouped_segments.append({
                            "start": w["start"],
                            "end": w["end"],
                            "speaker": w_spk,
                            "text": w["word"],
                            "source_text": None,
                            "words": [w],
                            "_orig_sidx": w_sidx,
                        })
            for rg in regrouped_segments:
                rg.pop("_orig_sidx", None)
            processed_segments = regrouped_segments

        # Pre-normalize each segment so Stage 2 role discovery and boundary healing operate on canonical legal phrases
        processed_segments = [normalize_segment(seg) for seg in processed_segments]

        # -------------------------------------------------------------------
        # STAGE 2: General Courtroom Conversational Smoothing
        # -------------------------------------------------------------------
        # Analyze speaker roles across the entire recording:
        # - Identifies who is the Bench (Judge) vs Counsel / Litigants
        # - Resolves honorific misattributions (e.g. Judge mistakenly tagged saying "My Lord")
        # - Supports any number of participants (Judge, Multiple Counsels, Registrar, Witnesses)
        speaker_counsel_votes: Dict[str, int] = {}
        speaker_judge_votes: Dict[str, int] = {}
        speaker_total_dur: Dict[str, float] = {}
        speaker_word_counts: Dict[str, int] = {}
        speaker_seg_counts: Dict[str, int] = {}

        RE_LEGAL_VOCAB_GATE = re.compile(
            r"\b(?:my\s+lord|your\s+lordship|your\s+honou?r|court\s+pleases|learned\s+counsel|"
            r"learned\s+silk|statement\s+of\s+claim|statement\s+of\s+defence|motion\s+on\s+notice|"
            r"affidavit|exhibit|cross[\s-]?examination|claimant|defendant|plaintiff|respondent|"
            r"applicant|petitioner|adjourn(?:ed)?|ruling|judg(?:e)?ment|counsel|magistrate|tribunal)\b",
            re.IGNORECASE,
        )

        total_audio_speech = 0.0
        legal_vocab_hits = 0
        for seg in processed_segments:
            spk = seg["speaker"]
            dur = max(0.05, seg["end"] - seg["start"])
            speaker_total_dur[spk] = speaker_total_dur.get(spk, 0.0) + dur
            speaker_seg_counts[spk] = speaker_seg_counts.get(spk, 0) + 1
            speaker_word_counts[spk] = speaker_word_counts.get(spk, 0) + len(seg.get("words", []))
            total_audio_speech += dur
            txt = seg["text"].lower()

            if any(p.search(txt) for p in re_counsel_honorifics):
                speaker_counsel_votes[spk] = speaker_counsel_votes.get(spk, 0) + 1
            if any(p.search(txt) for p in re_judge_authority) or RE_JUDGE_DIRECTIVES.search(txt):
                speaker_judge_votes[spk] = speaker_judge_votes.get(spk, 0) + 1
            if RE_LEGAL_VOCAB_GATE.search(txt):
                legal_vocab_hits += 1

        total_role_votes = sum(speaker_counsel_votes.values()) + sum(speaker_judge_votes.values())
        is_courtroom_context = (total_role_votes >= 1) or (legal_vocab_hits >= 2)

        counsel_spk = None
        judge_spk = None
        all_counsel_spks = set()

        if is_courtroom_context:
            # Identify Primary Counsel (most honorifics addressed to bench)
            counsel_candidates = [s for s in speaker_counsel_votes if speaker_counsel_votes[s] > 0]
            if counsel_candidates:
                counsel_spk = max(counsel_candidates, key=lambda s: speaker_counsel_votes[s])
                all_counsel_spks = set(counsel_candidates)
            else:
                counsel_spk = processed_segments[0]["speaker"]
                all_counsel_spks = {counsel_spk}

            # Identify Primary Judge (The Bench)
            # Only select a speaker as judge_spk if they have positive judicial authority votes and are not primarily a Counsel speaker
            judge_candidates = [
                s for s in speaker_judge_votes
                if speaker_judge_votes[s] > 0
                and s != counsel_spk
                and speaker_judge_votes[s] >= speaker_counsel_votes.get(s, 0)
            ]
            if judge_candidates:
                judge_spk = max(judge_candidates, key=lambda s: (speaker_judge_votes[s], -speaker_counsel_votes.get(s, 0)))
            elif len(speaker_total_dur) == 2:
                # Strictly 2-speaker recording fallback: the non-counsel speaker with 0 counsel honorifics is the Bench
                other_speakers = [
                    s for s in speaker_total_dur
                    if s != counsel_spk and speaker_total_dur[s] >= 1.0 and speaker_counsel_votes.get(s, 0) == 0
                ]
                if other_speakers:
                    judge_spk = other_speakers[0]
            else:
                # Multi-speaker courtroom where the Judge's mic was merged with a Counsel cluster:
                # use a dedicated virtual cluster for Bench rulings so other speakers (e.g. 2nd Counsel, Witness, Registrar) are never overwritten!
                if sum(speaker_judge_votes.values()) > 0:
                    judge_spk = "SPEAKER_COURT"
            if judge_spk in all_counsel_spks and len(all_counsel_spks) > 1:
                all_counsel_spks.discard(judge_spk)

            logger.info(
                f"Courtroom Role Discovery: Counsel={counsel_spk} (all_counsels={sorted(all_counsel_spks)}, votes={speaker_counsel_votes}), "
                f"Judge={judge_spk} (votes={speaker_judge_votes})"
            )
        else:
            logger.info(
                f"Courtroom Role Discovery skipped (non-courtroom audio detected: role_votes={total_role_votes}, "
                f"legal_vocab_hits={legal_vocab_hits}); preserving pure neural voiceprint diarization."
            )

        # Micro-cluster pruning: only prune true acoustic glitches/clicks (< 0.8s, <= 1 word, isolated turn)
        # Legitimate 2nd / 3rd participants (e.g. Father, Registrar, 2nd Counsel, Witness) are strictly preserved!
        cluster_remap: Dict[str, str] = {}
        target_max_speakers = getattr(WorkerConfig, "MAX_SPEAKERS", None)
        dominant_overall_spk = max(speaker_total_dur, key=speaker_total_dur.get) if speaker_total_dur else "SPEAKER_00"

        for spk, dur in speaker_total_dur.items():
            if spk in (counsel_spk, judge_spk):
                continue
            word_cnt = speaker_word_counts.get(spk, 0)
            seg_cnt = speaker_seg_counts.get(spk, 0)

            is_micro_phantom = (dur < 0.8 and word_cnt <= 1 and seg_cnt <= 1)
            is_enforced_2spk = (is_courtroom_context and target_max_speakers == 2 and len(speaker_total_dur) > 2)

            if is_micro_phantom or is_enforced_2spk:
                if is_courtroom_context:
                    c_votes = speaker_counsel_votes.get(spk, 0)
                    j_votes = speaker_judge_votes.get(spk, 0)
                    if c_votes > j_votes and counsel_spk is not None:
                        cluster_remap[spk] = counsel_spk
                    elif j_votes > c_votes and judge_spk is not None:
                        cluster_remap[spk] = judge_spk
                    elif judge_spk is not None and spk != counsel_spk:
                        cluster_remap[spk] = judge_spk
                elif spk != dominant_overall_spk:
                    cluster_remap[spk] = dominant_overall_spk

        if cluster_remap:
            for seg in processed_segments:
                orig = seg["speaker"]
                if orig in cluster_remap:
                    target = cluster_remap[orig]
                    seg["speaker"] = target
                    for w in seg.get("words", []):
                        w["speaker"] = target

        def _clean_w(w_obj: Dict[str, Any]) -> str:
            return str(w_obj.get("word", "")).lower().strip(".,!?;:\"'()[]{} ")

        def _has_term(w_obj: Dict[str, Any]) -> bool:
            t = str(w_obj.get("word", "")).strip()
            return any(t.endswith(p) for p in (".", "?", "!"))

        def _rebuild_seg(seg_obj: Dict[str, Any]):
            w_list = seg_obj.get("words", [])
            if w_list:
                seg_obj["start"] = w_list[0]["start"]
                seg_obj["end"] = w_list[-1]["end"]
                seg_obj["text"] = " ".join(str(w.get("word", "")).strip() for w in w_list if str(w.get("word", "")).strip()).strip()
                for w in w_list:
                    w["speaker"] = seg_obj["speaker"]

        FORMULA_END_TOKENS = {"pleases", "grateful", "obliged", "lord", "lordship", "honour", "honor", "sir"}
        PURE_BC_SET = {
            "yes", "no", "yeah", "nope", "sure", "correct", "sir", "true",
            "milord", "pleases", "grateful", "okay", "ok", "sorry", "pardon"
        }

        def _find_bc_indices(word_list: List[Dict[str, Any]]) -> List[int]:
            toks = [_clean_w(w) for w in word_list]
            n_t = len(toks)
            bc_idx = set()
            for idx_t, t in enumerate(toks):
                if t in PURE_BC_SET:
                    bc_idx.add(idx_t)
                elif t == "my" and idx_t + 1 < n_t and toks[idx_t + 1] in ("lord", "lordship", "ladyship"):
                    bc_idx.update((idx_t, idx_t + 1))
                elif t in ("lord", "lordship", "ladyship") and idx_t > 0 and toks[idx_t - 1] in ("my", "your"):
                    bc_idx.update((idx_t - 1, idx_t))
                elif t in ("i'm", "am") and idx_t + 1 < n_t and toks[idx_t + 1] == "grateful":
                    bc_idx.update((idx_t, idx_t + 1))
                elif t == "as" and idx_t + 3 < n_t and toks[idx_t : idx_t + 4] == ["as", "the", "court", "pleases"]:
                    bc_idx.update((idx_t, idx_t + 1, idx_t + 2, idx_t + 3))
            return sorted(bc_idx)

        # Pass 1C (Run BEFORE Pass 1 whole-segment flipping!):
        # Split embedded or trailing Counsel formulas inside a Judge segment, or Judge imperatives around Counsel formulas:
        # e.g. "processes no no no My Lord so why did you say..." ->
        #      ["processes" (Judge), "No, no, no, My Lord." (Counsel), "so why did you say..." (Judge)]
        # e.g. "file a motion for that purpose As the Court pleases. file a motion" ->
        #      ["file a motion for that purpose" (Judge), "As the Court pleases." (Counsel), "file a motion" (Judge)]
        # e.g. "prayer? Sir? Can I see the order? Yes, My Lord. It was stated..." ->
        #      ["prayer?" (Judge), "Sir?" (Counsel), "Can I see the order?" (Judge), "Yes, My Lord. It was stated..." (Counsel)]
        # e.g. "the Judge it's not fair now My Lord" ->
        #      ["the Judge it's not fair now" (Judge), "My Lord" (Counsel)]
        if judge_spk is not None and counsel_spk is not None:
            EMBEDDED_COUNSEL_SPANS = [
                ["no", "no", "no", "my", "lord"],
                ["no", "no", "my", "lord"],
                ["yes", "my", "lord", "lord"],
                ["yes", "my", "lord"],
                ["no", "my", "lord"],
                ["as", "the", "court", "pleases"],
                ["i'm", "grateful", "my", "lord"],
                ["am", "grateful", "my", "lord"],
                ["i'm", "grateful"],
                ["am", "grateful"],
                ["my", "lord"],
            ]
            # Iteratively split up to 3 times per segment
            for _pass_iter in range(3):
                expanded_segments: List[Dict[str, Any]] = []
                any_split = False
                for seg in processed_segments:
                    w_list = seg.get("words", [])
                    if len(w_list) < 4:
                        expanded_segments.append(seg)
                        continue
                    c_tokens = [_clean_w(w) for w in w_list]
                    split_done = False

                    # Case 1C-1: Split at "Sir?" inside or at the end of a Judge segment
                    # e.g. "prayer? Sir? Can I see the order?" or "...Can I see the prayer? Sir?"
                    if seg["speaker"] == judge_spk:
                        for idx_w in range(1, len(c_tokens)):
                            if c_tokens[idx_w] == "sir" and _has_term(w_list[idx_w]) and _has_term(w_list[idx_w - 1]):
                                s1 = dict(seg, words=w_list[:idx_w], speaker=judge_spk)
                                s2 = dict(seg, words=[w_list[idx_w]], speaker=counsel_spk)
                                _rebuild_seg(s1)
                                _rebuild_seg(s2)
                                if idx_w + 1 < len(w_list):
                                    s3 = dict(seg, words=w_list[idx_w + 1:], speaker=judge_spk)
                                    _rebuild_seg(s3)
                                    expanded_segments.extend([s1, s2, s3])
                                else:
                                    expanded_segments.extend([s1, s2])
                                split_done = True
                                any_split = True
                                break
                    if split_done:
                        continue

                    # Case 1C-2: Split embedded or trailing Counsel formula inside a segment containing Judge speech
                    for span in EMBEDDED_COUNSEL_SPANS:
                        s_len = len(span)
                        for idx_w in range(1, len(c_tokens) - s_len + 1):
                            if c_tokens[idx_w : idx_w + s_len] == span:
                                lead_w = w_list[:idx_w]
                                rest_w = w_list[idx_w:]
                                lead_txt = " ".join(_clean_w(x) for x in lead_w)

                                # Case A: Trailing "My Lord" at the very end of a Judge segment
                                if span == ["my", "lord"] and idx_w + s_len == len(c_tokens) and seg["speaker"] == judge_spk:
                                    # Check that lead_w is not ending with "yes" or "no" or "grateful" (handled by longer spans)
                                    if c_tokens[idx_w - 1] not in ("yes", "no", "grateful", "sorry", "objection"):
                                        s1 = dict(seg, words=lead_w, speaker=judge_spk)
                                        s2 = dict(seg, words=rest_w, speaker=counsel_spk)
                                        _rebuild_seg(s1)
                                        _rebuild_seg(s2)
                                        expanded_segments.extend([s1, s2])
                                        split_done = True
                                        any_split = True
                                        break

                                # Case B: Embedded formula ("as the court pleases", "no no no my lord", "yes my lord")
                                if span != ["my", "lord"]:
                                    mid_w = w_list[idx_w : idx_w + s_len]
                                    tail_w = w_list[idx_w + s_len :]
                                    # If tail_w starts a Counsel explanation ("it was stated...") after "Yes, My Lord.",
                                    # and lead_w is a Judge question ("Can I see the order?"), split into [lead (Judge), mid+tail (Counsel)]!
                                    if span in (["yes", "my", "lord", "lord"], ["yes", "my", "lord"]) and RE_BENCH_INQUIRY.search(lead_txt):
                                        # Dedup "Yes, My Lord. Lord" -> "Yes, My Lord."
                                        if span == ["yes", "my", "lord", "lord"]:
                                            rest_w = w_list[idx_w : idx_w + 3] + w_list[idx_w + 4 :]
                                        s1 = dict(seg, words=lead_w, speaker=judge_spk)
                                        s2 = dict(seg, words=rest_w, speaker=counsel_spk)
                                        _rebuild_seg(s1)
                                        _rebuild_seg(s2)
                                        expanded_segments.extend([s1, s2])
                                        split_done = True
                                        any_split = True
                                        break
                                    elif tail_w and (seg["speaker"] == judge_spk or "file a motion" in lead_txt):
                                        s1 = dict(seg, words=lead_w, speaker=judge_spk)
                                        s2 = dict(seg, words=mid_w, speaker=counsel_spk)
                                        s3 = dict(seg, words=tail_w, speaker=judge_spk)
                                        _rebuild_seg(s1)
                                        _rebuild_seg(s2)
                                        _rebuild_seg(s3)
                                        expanded_segments.extend([s1, s2, s3])
                                        split_done = True
                                        any_split = True
                                        break
                                    elif not tail_w and (seg["speaker"] == judge_spk or any(p.search(lead_txt) for p in re_judge_authority) or RE_JUDGE_DIRECTIVES.search(lead_txt)):
                                        s1 = dict(seg, words=lead_w, speaker=judge_spk)
                                        s2 = dict(seg, words=mid_w, speaker=counsel_spk)
                                        _rebuild_seg(s1)
                                        _rebuild_seg(s2)
                                        expanded_segments.extend([s1, s2])
                                        split_done = True
                                        any_split = True
                                        break
                        if split_done:
                            break
                    if not split_done:
                        expanded_segments.append(seg)
                processed_segments = expanded_segments
                if not any_split:
                    break

        # Pass 1: Universal Courtroom Rule: The Judge NEVER addresses themselves as "My Lord" or "As the Court pleases"
        RE_REGISTRAR_PROTECT = re.compile(
            r"\b(?:suit\s+no|charge\s+no|appeal\s+no|petitioner\s+is\s+present|respondent\s+is\s+absent|"
            r"reminded\s+of\s+(?:his|her)\s+oath|witnesses\s+are\s+ordered\s+out|take\s+the\s+witness\s+stand|"
            r"do\s+you\s+swear\s+that\s+the\s+evidence|nothing\s+but\s+the\s+truth)\b",
            re.IGNORECASE,
        )
        if counsel_spk is not None and judge_spk is not None and judge_spk != "SPEAKER_COURT":
            for seg in processed_segments:
                if seg["speaker"] == judge_spk:
                    txt = seg["text"].lower()
                    if RE_REGISTRAR_PROTECT.search(txt):
                        continue
                    if any(p.search(txt) for p in re_counsel_honorifics) and not RE_BENCH_INQUIRY.search(txt):
                        orig_speakers = [w.get("speaker") for w in seg.get("words", []) if w.get("speaker") and w.get("speaker") != judge_spk]
                        if orig_speakers:
                            chosen_spk = max(set(orig_speakers), key=orig_speakers.count)
                            seg["speaker"] = chosen_spk
                        else:
                            seg["speaker"] = counsel_spk
                        for w in seg.get("words", []):
                            w["speaker"] = seg["speaker"]

        # Pass 1B: Universal Courtroom Rule: Only the Bench issues judicial orders, maintenance directives, and contempt warnings
        if judge_spk is not None:
            for seg in processed_segments:
                if seg["speaker"] != judge_spk:
                    txt = seg["text"].lower()
                    if RE_JUDGE_DIRECTIVES.search(txt):
                        seg["speaker"] = judge_spk
                        for w in seg.get("words", []):
                            w["speaker"] = judge_spk

        def _run_pass_1e(seg_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            for i in range(1, len(seg_list)):
                prev_seg = seg_list[i - 1]
                curr_seg = seg_list[i]
                prev_words = prev_seg.get("words", [])
                curr_words = curr_seg.get("words", [])

                if not prev_words or not curr_words or prev_seg["speaker"] == curr_seg["speaker"]:
                    continue

                prev_last_clean = _clean_w(prev_words[-1])
                curr_first_clean = _clean_w(curr_words[0])

                # 0. Split Honorific Completion across segments ("...asking for further dates, my" | "Lord. What is the court date?")
                if prev_last_clean in ("my", "your") and curr_first_clean in ("lord", "lords", "lordship", "ladyship", "honour", "honor", "worship"):
                    moved_w = curr_words[0]
                    prev_seg["words"][-1]["word"] = "My" if prev_last_clean == "my" else "Your"
                    if not _has_term(moved_w):
                        moved_w["word"] = "Lord." if curr_first_clean == "lord" else moved_w["word"]
                    prev_seg["words"].append(moved_w)
                    curr_seg["words"] = curr_words[1:]
                    _rebuild_seg(prev_seg)
                    _rebuild_seg(curr_seg)
                    prev_words = prev_seg["words"]
                    curr_words = curr_seg["words"]
                    if not prev_words or not curr_words:
                        continue
                    prev_last_clean = _clean_w(prev_words[-1])
                    curr_first_clean = _clean_w(curr_words[0])

                # 1. Leading preposition completion ("...that you asked" + "for yes" / "...last known address" + "of file a motion")
                if not _has_term(prev_words[-1]) and len(curr_words) >= 2:
                    curr_second_clean = _clean_w(curr_words[1])
                    is_dangling_prep = (
                        (prev_last_clean in {"asked", "asking", "applied", "apply", "prayed", "pray", "looking", "waiting"} and curr_first_clean == "for") or
                        (prev_last_clean in {"address", "service", "notice", "copy", "provisions", "requirements", "mode"} and curr_first_clean == "of")
                    )
                    if is_dangling_prep and (curr_second_clean in PURE_BC_SET or curr_second_clean in {"my", "as", "i'm", "file", "serve", "make", "put"} or _has_term(curr_words[0])):
                        moved_w = curr_words[0]
                        prev_seg["words"].append(moved_w)
                        curr_seg["words"] = curr_words[1:]
                        _rebuild_seg(prev_seg)
                        _rebuild_seg(curr_seg)
                        prev_words = prev_seg["words"]
                        curr_words = curr_seg["words"]
                        if not prev_words or not curr_words:
                            continue

                # 2. Trailing bleed after terminal punctuation OR after a courtroom formula
                for cut_len in (7, 6, 5, 4, 3, 2, 1):
                    if len(prev_words) > cut_len:
                        trail_words = prev_words[-cut_len:]
                        lead_words = prev_words[:-cut_len]

                        last_lead_w = lead_words[-1]
                        last_lead_clean = _clean_w(last_lead_w)
                        has_terminal = _has_term(last_lead_w)
                        ends_formula = last_lead_clean in FORMULA_END_TOKENS or (
                            len(lead_words) >= 2 and [_clean_w(lead_words[-2]), last_lead_clean] in (
                                ["court", "pleases"], ["my", "lord"], ["i'm", "grateful"], ["am", "grateful"], ["yes", "sir"], ["no", "sir"]
                            )
                        )

                        trail_clean_list = [_clean_w(w) for w in trail_words]
                        trail_txt = " ".join(trail_clean_list)
                        curr_head_list = [_clean_w(w) for w in curr_words[:3]]
                        curr_head = " ".join(curr_head_list)
                        combined = f"{trail_txt} {curr_head}"

                        is_severed_phrase = (
                            ("as the court pleases" in combined) or
                            ("as your lordship pleases" in combined) or
                            ("may it please the court" in combined) or
                            ("with due respect" in combined) or
                            ("objection" in trail_txt and "my lord" in combined) or
                            (trail_txt in ("as the", "as") and "court" in curr_head) or
                            (trail_clean_list[:2] == ["my", "lord"] and prev_seg["speaker"] == judge_spk) or
                            (trail_clean_list[:2] == ["may", "i"] and prev_seg["speaker"] == judge_spk)
                        )

                        trail_has_end_punct = _has_term(trail_words[-1])
                        trail_is_judge_phrase = (
                            RE_BENCH_INQUIRY.search(trail_txt)
                            or RE_JUDGE_DIRECTIVES.search(trail_txt)
                            or any(p.search(trail_txt) for p in re_judge_authority)
                        )
                        is_onset_bleed = False
                        if (has_terminal or ends_formula) and not trail_has_end_punct and not (prev_seg["speaker"] == judge_spk and trail_is_judge_phrase):
                            if ends_formula and prev_seg["speaker"] == counsel_spk and curr_seg["speaker"] == judge_spk:
                                if len(lead_words) <= 6:
                                    is_onset_bleed = True
                            if has_terminal and cut_len <= 5:
                                if len(curr_words) <= 4 or not str(curr_words[0].get("word", ""))[:1].isupper() or trail_clean_list[0] in {
                                    "can", "let", "so", "and", "but", "is", "are", "was", "were", "what", "where", "why", "how",
                                    "my", "may", "i", "we", "you", "put", "file", "tell", "no", "yes"
                                }:
                                    is_onset_bleed = True

                        if is_severed_phrase or is_onset_bleed:
                            logger.info(f"Boundary Healing: moving '{trail_txt}' from seg {i-1} ({prev_seg['speaker']}) to seg {i} ({curr_seg['speaker']})")
                            curr_seg["words"] = trail_words + curr_words
                            prev_seg["words"] = lead_words
                            _rebuild_seg(prev_seg)
                            _rebuild_seg(curr_seg)
                            break
            return [s for s in seg_list if s.get("words")]

        # Pass 1E (First pass): Heal cross-segment bleeds before sandwich trimming
        processed_segments = _run_pass_1e(processed_segments)

        # Pass 1D: Sandwiched Backchannel Trimming across 3 consecutive segments (SPEAKER_A | short SPEAKER_B | SPEAKER_A)
        for i in range(1, len(processed_segments) - 1):
            prev_seg = processed_segments[i - 1]
            curr_seg = processed_segments[i]
            next_seg = processed_segments[i + 1]
            c_words = curr_seg.get("words", [])
            if prev_seg["speaker"] == next_seg["speaker"] and curr_seg["speaker"] != prev_seg["speaker"] and 1 <= len(c_words) <= 9:
                c_clean = [_clean_w(w) for w in c_words]
                # Special case: "says put it in writing as you make your" (between two Judge turns ending in "...and all that" and "order...")
                if c_clean and c_clean[0] in ("says", "yes", "yeah") and len(c_words) >= 4 and not _has_term(c_words[-1]) and not _find_bc_indices(c_words[1:]):
                    if c_clean[0] == "says":
                        c_words[0]["word"] = "Yes."
                    trail_bleed = c_words[1:]
                    next_seg["words"] = trail_bleed + next_seg["words"]
                    curr_seg["words"] = [c_words[0]]
                    _rebuild_seg(curr_seg)
                    _rebuild_seg(next_seg)
                    continue

                if len(c_words) <= 5:
                    bc_pos = _find_bc_indices(c_words)
                    if bc_pos:
                        f_bc, l_bc = bc_pos[0], bc_pos[-1]
                        lead_bleed = c_words[:f_bc]
                        core_bc = c_words[f_bc : l_bc + 1]
                        trail_bleed = c_words[l_bc + 1 :]
                        if lead_bleed and (lead_bleed[0]["start"] - prev_seg["end"]) <= 1.2:
                            prev_seg["words"].extend(lead_bleed)
                            _rebuild_seg(prev_seg)
                        elif lead_bleed:
                            core_bc = lead_bleed + core_bc
                        if trail_bleed and (next_seg["start"] - trail_bleed[-1]["end"]) <= 1.2:
                            next_seg["words"] = trail_bleed + next_seg["words"]
                            _rebuild_seg(next_seg)
                        elif trail_bleed:
                            core_bc = core_bc + trail_bleed
                        curr_seg["words"] = core_bc
                        _rebuild_seg(curr_seg)
                    elif len(c_words) <= 3 and not _has_term(prev_seg["words"][-1]) and (curr_seg["start"] - prev_seg["end"]) <= 0.6:
                        # Pure mid-sentence fragment ("as", "as not in", "for") with no valid backchannel: return to SPEAKER_A
                        curr_seg["speaker"] = prev_seg["speaker"]
                        _rebuild_seg(curr_seg)

        processed_segments = [s for s in processed_segments if s.get("words")]

        # Pass 1E (Second pass): Clean up any newly adjacent boundaries
        processed_segments = _run_pass_1e(processed_segments)

        # Pass 2: Conversational Alternation & Response Sandwiches
        SHORT_AFFIRMATIONS = {"sir?", "sir", "yes sir", "yes my lord", "no my lord", "as the court pleases"}
        if judge_spk is not None and counsel_spk is not None:
            for i in range(len(processed_segments)):
                seg = processed_segments[i]
                txt_clean = seg["text"].lower().strip(".,!?;:\"' ")

                # Case A: Bench Authority & Document Inquiries ("Can I see the prayer?", "Well, choose the one you want.")
                if (
                    (RE_BENCH_INQUIRY.search(txt_clean) or any(p.search(txt_clean) for p in re_judge_authority))
                    and not any(p.search(txt_clean) for p in re_counsel_honorifics)
                    and not RE_REGISTRAR_PROTECT.search(txt_clean)
                ):
                    seg["speaker"] = judge_spk
                    continue

                # Case B: Submission Sandwich ("As the Court pleases", "Yes, My Lord")
                # When Counsel submits to the Bench in a 2-speaker exchange, the preceding segment was the Judge
                if i > 0 and len(speaker_total_dur) <= 2 and (RE_SUBMISSION.search(txt_clean) or txt_clean in {"yes, my lord", "no, my lord", "yes sir", "no sir"}):
                    prev_seg = processed_segments[i - 1]
                    if prev_seg["speaker"] == counsel_spk and not any(p.search(prev_seg["text"].lower()) for p in re_counsel_honorifics) and not RE_REGISTRAR_PROTECT.search(prev_seg["text"]):
                        prev_seg["speaker"] = judge_spk

                # Case C: Clarification Sandwich ("Sir?", "My Lord?", "Pardon?")
                # When Counsel asks for clarification, the preceding segment was the Judge
                if i > 0 and RE_CLARIFICATION.search(txt_clean):
                    prev_seg = processed_segments[i - 1]
                    if prev_seg["speaker"] == counsel_spk:
                        prev_seg["speaker"] = judge_spk

                # Case D: Counsel Case Argumentation ("what I'm saying", "our submission", "we said...")
                if RE_COUNSEL_ARGUMENT.search(txt_clean):
                    if seg["speaker"] == judge_spk:
                        seg["speaker"] = counsel_spk

                # Case E: Safe Continuation for Judicial Directives
                # Only joins across a brief pause (<= 0.4s) when the Judge was reading an active directive with a dangling conjunction
                if i > 0:
                    prev_seg = processed_segments[i - 1]
                    gap = seg["start"] - prev_seg["end"]
                    prev_tokens = prev_seg["text"].lower().strip(".,!?;:\"' ").split()
                    prev_last_word = prev_tokens[-1] if prev_tokens else ""
                    if gap <= 0.4 and prev_last_word in DANGLING_CONJUNCTIONS:
                        if prev_seg["speaker"] == judge_spk and RE_JUDGE_DIRECTIVES.search(prev_seg["text"]):
                            seg["speaker"] = judge_spk

                # Case F: Short Affirmations & Submissions ("Sir?", "Yes sir", "As the Court pleases")
                if (txt_clean in SHORT_AFFIRMATIONS or (any(p.search(txt_clean) for p in re_counsel_honorifics) and not RE_BENCH_INQUIRY.search(txt_clean))) and seg["speaker"] == judge_spk:
                    seg["speaker"] = counsel_spk

        # -------------------------------------------------------------------
        # STAGE 2B: Neural Voiceprint Turn Verification & Continuity Smoothing
        # Eliminates false speaker splits mid-monologue when a speaker's volume,
        # pitch, or background noise shifts, while respecting verified legal roles.
        # -------------------------------------------------------------------
        if len(set(s["speaker"] for s in processed_segments)) > 1:
            import numpy as np

            def _has_explicit_legal_marker(seg_obj: Dict[str, Any]) -> bool:
                if not is_courtroom_context:
                    return False
                t_low = seg_obj.get("text", "").lower().strip(".,!?;:\"' ")
                if t_low in SHORT_AFFIRMATIONS or RE_SUBMISSION.search(t_low) or RE_CLARIFICATION.search(t_low):
                    return True
                if RE_BENCH_INQUIRY.search(t_low) or RE_JUDGE_DIRECTIVES.search(t_low) or RE_COUNSEL_ARGUMENT.search(t_low):
                    return True
                if any(p.search(t_low) for p in re_counsel_honorifics) or any(p.search(t_low) for p in re_judge_authority):
                    return True
                return False

            # Build per-speaker neural voiceprint centroids across processed_segments
            seg_spk_centroids: Dict[str, np.ndarray] = {}
            seg_embeddings_cache: Dict[int, np.ndarray] = {}
            if audio_data is not None and len(audio_data) > 0 and self.diarizer is not None:
                spk_to_seg_indices: Dict[str, List[int]] = {}
                for idx_s, s_obj in enumerate(processed_segments):
                    spk_to_seg_indices.setdefault(s_obj["speaker"], []).append(idx_s)
                    s_st = max(0, int(float(s_obj["start"]) * sample_rate))
                    s_et = min(len(audio_data), int(float(s_obj["end"]) * sample_rate))
                    if s_et - s_st >= int(0.32 * sample_rate):
                        emb = self.diarizer._extract_audio_feature(audio_data[s_st:s_et], sample_rate)
                        if emb is not None:
                            seg_embeddings_cache[idx_s] = emb

                for spk_k, idx_list in spk_to_seg_indices.items():
                    # Sort by segment duration descending to build clean centroid from top 8 longest turns
                    sorted_idx = sorted(
                        idx_list,
                        key=lambda ix: float(processed_segments[ix]["end"]) - float(processed_segments[ix]["start"]),
                        reverse=True,
                    )
                    embs_k = [seg_embeddings_cache[ix] for ix in sorted_idx[:8] if ix in seg_embeddings_cache]
                    if embs_k:
                        avg_e = np.mean(embs_k, axis=0)
                        nrm_e = float(np.linalg.norm(avg_e))
                        if nrm_e > 1e-6:
                            seg_spk_centroids[spk_k] = (avg_e / nrm_e).astype(np.float32)

                # 1. Merge any remaining duplicate speaker clusters whose TitaNet voiceprints match
                #    (unless they are the primary opposing Counsel vs Judge roles in a courtroom recording)
                curr_spk_durs: Dict[str, float] = {}
                for s_obj in processed_segments:
                    sk = s_obj["speaker"]
                    curr_spk_durs[sk] = curr_spk_durs.get(sk, 0.0) + max(0.05, float(s_obj["end"]) - float(s_obj["start"]))
                ranked_seg_spks = sorted(curr_spk_durs.keys(), key=lambda x: curr_spk_durs[x], reverse=True)
                uses_neural = getattr(self.diarizer, "speaker_emb_model", None) is not None

                seg_cluster_merge: Dict[str, str] = {}
                for i_min in range(1, len(ranked_seg_spks)):
                    s_min = ranked_seg_spks[i_min]
                    if s_min not in seg_spk_centroids:
                        continue
                    for i_maj in range(i_min):
                        s_maj = ranked_seg_spks[i_maj]
                        while s_maj in seg_cluster_merge:
                            s_maj = seg_cluster_merge[s_maj]
                        if s_maj not in seg_spk_centroids or s_maj == s_min:
                            continue
                        # Never merge Primary Judge and Primary Counsel together if both were identified by legal phrases
                        if (
                            is_courtroom_context
                            and judge_spk is not None
                            and counsel_spk is not None
                            and {s_maj, s_min} == {judge_spk, counsel_spk}
                            and total_role_votes >= 2
                        ):
                            continue
                        sim_val = float(np.dot(seg_spk_centroids[s_maj], seg_spk_centroids[s_min]))
                        # In courtroom audio, two distinct speakers on the same room mic have sim ~ 0.50-0.68,
                        # whereas duplicate chunk clusters of the same person have sim >= 0.90.
                        # In non-courtroom audio (e.g. solo vocalist music track over beat drops), sim >= 0.50 merges the artist.
                        if is_courtroom_context:
                            merge_thresh = 0.90 if uses_neural else 0.94
                        else:
                            merge_thresh = 0.50 if uses_neural else 0.76
                        if sim_val >= merge_thresh:
                            logger.info(
                                f"[PIPELINE VOICEPRINT MERGE] Merging duplicate speaker bank {s_min} -> {s_maj} "
                                f"(voiceprint_sim={sim_val:.3f}, courtroom={is_courtroom_context})"
                            )
                            seg_cluster_merge[s_min] = s_maj
                            break

                if seg_cluster_merge:
                    for s_obj in processed_segments:
                        cur = s_obj["speaker"]
                        while cur in seg_cluster_merge:
                            cur = seg_cluster_merge[cur]
                        s_obj["speaker"] = cur
                        for w in s_obj.get("words", []):
                            w["speaker"] = cur

            # 2. Smooth Sandwiched Turns (SPEAKER_A -> SPEAKER_B -> SPEAKER_A) & Non-Courtroom Short Flips
            for _smooth_pass in range(2):
                changed_any = False
                for idx_s in range(len(processed_segments)):
                    seg = processed_segments[idx_s]
                    if _has_explicit_legal_marker(seg):
                        continue
                    seg_dur = float(seg["end"]) - float(seg["start"])
                    seg_w_cnt = len(seg.get("words", []))
                    has_bc = bool(_find_bc_indices(seg.get("words", [])))
                    ends_q = seg.get("text", "").strip().endswith("?")

                    # Case A: Sandwiched segment between the same speaker (SPEAKER_A | seg | SPEAKER_A)
                    if idx_s > 0 and idx_s < len(processed_segments) - 1:
                        prev_spk = processed_segments[idx_s - 1]["speaker"]
                        next_spk = processed_segments[idx_s + 1]["speaker"]
                        if prev_spk == next_spk and seg["speaker"] != prev_spk and not has_bc:
                            if is_courtroom_context:
                                # In courtroom proceedings, only absorb unpunctuated mid-sentence micro-fragments (<= 3 words, <= 1.1s)
                                # that are NOT questions ("So the order?", "Can I see the...")
                                prev_ends_punct = _has_term(processed_segments[idx_s - 1]["words"][-1]) if processed_segments[idx_s - 1].get("words") else True
                                if (not ends_q) and (not prev_ends_punct) and seg_w_cnt <= 3 and seg_dur <= 1.1:
                                    seg["speaker"] = prev_spk
                                    changed_any = True
                                    continue
                            else:
                                # Non-courtroom (e.g. music / solo media): absorb <= 5 words / <= 2.2s sandwich
                                if seg_dur <= 2.2 or seg_w_cnt <= 5:
                                    seg["speaker"] = prev_spk
                                    changed_any = True
                                    continue

                    # Case B: Short isolated speaker flip in non-courtroom audio (e.g. "Check.", "I wonder why...")
                    if (not is_courtroom_context) and idx_s > 0 and not has_bc and (seg_dur < 1.8 or seg_w_cnt <= 5):
                        prev_spk = processed_segments[idx_s - 1]["speaker"]
                        if seg["speaker"] != prev_spk:
                            if idx_s in seg_embeddings_cache and prev_spk in seg_spk_centroids and seg["speaker"] in seg_spk_centroids:
                                sim_prev = float(np.dot(seg_embeddings_cache[idx_s], seg_spk_centroids[prev_spk]))
                                sim_curr = float(np.dot(seg_embeddings_cache[idx_s], seg_spk_centroids[seg["speaker"]]))
                                if sim_prev >= sim_curr - 0.05:
                                    seg["speaker"] = prev_spk
                                    changed_any = True
                            else:
                                seg["speaker"] = prev_spk
                                changed_any = True
                if not changed_any:
                    break

        # Final Same-Speaker Consolidation: stitch adjacent fragments of the same speaker
        # created during boundary healing (e.g. "Can I see the" + "prayer?", "so you have" + "various options")
        stitched_same_spk: List[Dict[str, Any]] = []
        for seg in processed_segments:
            if not seg.get("words"):
                continue
            if not stitched_same_spk:
                _rebuild_seg(seg)
                stitched_same_spk.append(seg)
            else:
                prev_s = stitched_same_spk[-1]
                gap = seg["start"] - prev_s["end"]
                prev_ends_term = _has_term(prev_s["words"][-1])
                if prev_s["speaker"] == seg["speaker"] and (
                    (gap <= 0.85 and not prev_ends_term) or
                    (gap <= 0.45 and (len(prev_s["words"]) <= 4 or len(seg["words"]) <= 4))
                ):
                    prev_s["words"].extend(seg["words"])
                    _rebuild_seg(prev_s)
                else:
                    _rebuild_seg(seg)
                    stitched_same_spk.append(seg)
        processed_segments = stitched_same_spk

        # Synchronize word-level speakers with the resolved segment speaker
        for seg in processed_segments:
            final_spk = seg["speaker"]
            for w in seg.get("words", []):
                w["speaker"] = final_spk


        # -------------------------------------------------------------------
        # STAGE 3: Chronological Speaker Normalization
        # First speaker heard = SPEAKER_00
        # Second speaker heard = SPEAKER_01, etc.
        # -------------------------------------------------------------------
        first_heard: Dict[str, float] = {}
        for seg in processed_segments:
            spk = seg["speaker"]
            if spk not in first_heard:
                first_heard[spk] = seg["start"]

        ordered_spks = sorted(first_heard.keys(), key=lambda s: first_heard[s])
        chrono_map = {orig: f"SPEAKER_{idx:02d}" for idx, orig in enumerate(ordered_spks)}

        for seg in processed_segments:
            seg["speaker"] = chrono_map.get(seg["speaker"], seg["speaker"])
            for w in seg.get("words", []):
                w["speaker"] = chrono_map.get(w.get("speaker"), w.get("speaker"))

        # -------------------------------------------------------------------
        # STAGE 4: Legal Terminology, Suit Citation & Digit Normalization
        # -------------------------------------------------------------------
        normalized_segments = [normalize_segment(seg) for seg in processed_segments]

        # Attach 192-D NeMo TitaNet voiceprint centroid to the first segment of each speaker
        if self.diarizer is not None and getattr(self.diarizer, "last_speaker_centroids", None):
            centroids = self.diarizer.last_speaker_centroids
            emitted_emb_speakers = set()
            for seg in normalized_segments:
                spk = seg.get("speaker")
                if spk and spk in centroids and spk not in emitted_emb_speakers:
                    seg["speaker_embedding"] = centroids[spk]
                    emitted_emb_speakers.add(spk)

        unique_speakers = len(set(s["speaker"] for s in normalized_segments))
        logger.info(f"reconstruct_speaker_turns: {unique_speakers} unique speakers, {len(normalized_segments)} segments preserved.")
        return normalized_segments, unique_speakers
