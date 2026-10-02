import gc
import re
import time
import logging
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
from ml_worker.normalizer import normalize_segment

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
        WorkerConfig.ensure_scratch_dir()

    def process(
        self,
        job_id: str,
        audio_url: str,
        enable_diarization: bool = True,
        enable_translation: bool = False,
        language: Optional[str] = None,
        target_language: Optional[str] = None,
        transcription_mode: str = "fast",
        progress_updater: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        raw_download_path: Optional[Path] = None
        wav_path: Optional[Path] = None

        try:
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

            # 3. Whisper Transcription
            if progress_updater:
                progress_updater("transcribing", 40)
            logger.info(f"[{job_id}] Running Whisper inference (mode={transcription_mode}, enable_translation={enable_translation}, lang={language}, target={target_language})...")

            def transcribe_progress_cb(current_sec, total_sec):
                if total_sec > 0 and progress_updater:
                    pct = 40 + int((current_sec / total_sec) * 35) # 40% to 75%
                    progress_updater("transcribing", min(75, pct))

            whisper_result = self.transcriber.transcribe(
                wav_path,
                language=language,
                target_language=target_language,
                enable_translation=enable_translation,
                mode=transcription_mode,
                progress_callback=transcribe_progress_cb
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
                    audio_arr = whisperx.load_audio(str(wav_path))
                    align_model, align_metadata = whisperx.load_align_model(
                        language_code=lang_code,
                        device=align_device
                    )
                    aligned_result = whisperx.align(
                        segments,
                        align_model,
                        align_metadata,
                        audio_arr,
                        align_device,
                        return_char_alignments=False
                    )

                    # Release alignment model from memory
                    del align_model
                    gc.collect()
                    if torch and align_device == "cuda":
                        torch.cuda.empty_cache()

                    logger.info(f"[{job_id}] Phoneme forced alignment successful across {len(aligned_result.get('segments', []))} segments.")
                except Exception as align_err:
                    logger.warning(f"[{job_id}] WhisperX phoneme alignment skipped or failed ({align_err}). Retaining Whisper timestamps.")
                    aligned_result = {"segments": segments}

            # Diarization & Word Midpoint Stitching
            total_align_segs = len(aligned_result.get("segments", []))
            failed_align_segs = sum(1 for s in aligned_result.get("segments", []) if not s.get("words"))
            logger.info(f"[{job_id}] [DIAGNOSTIC] Alignment status: {total_align_segs - failed_align_segs}/{total_align_segs} segments have word-level timestamps ({failed_align_segs} failed alignment).")

            logger.info(f"[{job_id}] Diarization check: requested={enable_diarization}, model_loaded={self.diarizer.is_loaded}")
            if self.diarizer.is_loaded and enable_diarization:
                if progress_updater:
                    progress_updater("diarizing", 85)
                logger.info(f"[{job_id}] Running Pyannote diarization on {wav_path}...")
                diarize_df = self.diarizer.diarize_dataframe(wav_path)
                logger.info(f"[{job_id}] Diarization generated {len(diarize_df)} intervals.")

                if len(diarize_df) > 0:
                    try:
                        # 1. Dump raw Pyannote intervals for the first 35 seconds
                        if hasattr(diarize_df, "iterrows"):
                            early_df = diarize_df[diarize_df["start"] <= 35.0] if "start" in diarize_df.columns else diarize_df.head(15)
                            pyannote_intervals = "\n".join([
                                f"    [{row.get('start', 0.0):.2f}s -> {row.get('end', 0.0):.2f}s] {row.get('speaker', 'UNKNOWN')}"
                                for _, row in early_df.iterrows()
                            ])
                            logger.info(f"[{job_id}] [DIAGNOSTIC 1: Raw Pyannote intervals (first 35s)]:\n{pyannote_intervals}")

                        import whisperx
                        stitched = whisperx.assign_word_speakers(diarize_df, aligned_result, fill_nearest=True)

                        # 2. Dump raw word speakers directly after whisperx.assign_word_speakers
                        raw_early_words = []
                        for s in stitched.get("segments", []):
                            for w in s.get("words", []):
                                if float(w.get("start", 0.0)) <= 35.0:
                                    raw_early_words.append(f"{w.get('word', '')}[{w.get('speaker', 'NONE')}]")
                        logger.info(f"[{job_id}] [DIAGNOSTIC 2: Raw Word Speakers from WhisperX (first 35s)]:\n" + " ".join(raw_early_words[:80]))

                        segments, num_speakers = self.reconstruct_speaker_turns(stitched)

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
            # Guarantee scratch file cleanup
            cleanup_file(raw_download_path)
            cleanup_file(wav_path)

    def reconstruct_speaker_turns(self, stitched_result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Takes WhisperX stitched word-level speaker assignments and builds
        clean, conversational speaker turns.

        The pipeline runs 5 correction passes before finalizing:

        PASS 0 - Whisper-Segment Majority-Vote Unification
            Whisper transcribes one utterance as one segment. WhisperX then
            assigns per-word speaker labels from Pyannote, but Pyannote
            boundary lag means the first or last 1-3 words of a segment get
            the WRONG speaker. We unify each short segment (<12s, >=3 words)
            to its majority speaker — the one who "owns" >60% of the words.
            This is the primary fix for "sentence split across two speakers".

        PASS 1 - Multi-word Boundary Jitter Smoothing (windows of 1, 2, 3 words)
            Even after Pass 0, words at the seam between two Whisper segments
            may still be mis-tagged. If a tight run of 1-3 words is flanked on
            BOTH sides by the same speaker with no large gap or sentence
            ending, snap the run to that speaker.

        PASS 2 - Conversational Turn Re-segmentation
            Group smoothed words into natural speaker turns using:
            - Speaker transition (always split)
            - Silence gap >= 1.2s
            - Sentence-end + gap >= 0.4s or block >= 7s
            - Max block guard >= 18s

        PASS 2b - Short Orphan Segment Absorption
            A segment of <= 3 words or < 1.5s sandwiched between two segments
            of the SAME speaker is almost certainly a ghost boundary artifact.
            Absorb it into the preceding segment rather than creating a
            spurious speaker turn.

        PASS 3 - Chronological Speaker Normalization
            Renumber speakers so the first voice heard = SPEAKER_00.

        PASS 4 - Legal Text Normalization
            Numbers as digits, honorific casing, suit citations, etc.
        """
        all_words: List[Dict[str, Any]] = []
        for seg in stitched_result.get("segments", []):
            words = seg.get("words", [])
            seg_id = id(seg)  # stable identity key for this segment object
            if not words:
                txt = seg.get("text", "").strip()
                if txt:
                    tokens = txt.split()
                    st = float(seg.get("start", 0.0))
                    et = float(seg.get("end", st + 0.1))
                    step = max(0.01, (et - st) / max(1, len(tokens)))
                    for idx, tok in enumerate(tokens):
                        all_words.append({
                            "word": tok,
                            "start": round(st + (idx * step), 3),
                            "end": round(st + ((idx + 1) * step), 3),
                            "speaker": seg.get("speaker") or "SPEAKER_00",
                            "score": None,
                            "_seg_id": seg_id,
                        })
            else:
                for w in words:
                    w_txt = str(w.get("word", "")).strip()
                    if w_txt:
                        w_start = float(w.get("start", 0.0))
                        w_end = float(w.get("end", w_start + 0.05))
                        if w_end < w_start:
                            w_end = w_start + 0.05
                        all_words.append({
                            "word": w_txt,
                            "start": round(w_start, 3),
                            "end": round(w_end, 3),
                            "speaker": str(w.get("speaker") or seg.get("speaker") or "SPEAKER_00"),
                            "score": round(float(w.get("score")), 3) if w.get("score") is not None else None,
                            "_seg_id": seg_id,
                        })

        if not all_words:
            return stitched_result.get("segments", []), 1

        # -------------------------------------------------------------------
        # PASS 0: Whisper-Segment Majority-Vote Speaker Unification
        # -------------------------------------------------------------------
        # Group word indices by their source Whisper segment
        seg_word_groups: Dict[int, List[int]] = {}
        for i, w in enumerate(all_words):
            sid = w.get("_seg_id", 0)
            seg_word_groups.setdefault(sid, []).append(i)

        for sid, indices in seg_word_groups.items():
            if len(indices) < 3:
                continue
            seg_dur = all_words[indices[-1]]["end"] - all_words[indices[0]]["start"]
            # Only unify segments short enough to realistically be one speaker.
            # Long segments (>12s) can legitimately span a speaker handoff.
            if seg_dur > 12.0:
                continue
            speaker_counts = Counter(all_words[i]["speaker"] for i in indices)
            majority_speaker, majority_count = speaker_counts.most_common(1)[0]
            # Apply only if majority is clear (>60%) — avoids forcing ambiguous segments
            if majority_count / len(indices) > 0.60:
                for i in indices:
                    all_words[i]["speaker"] = majority_speaker

        # -------------------------------------------------------------------
        # PASS 1: Multi-word Boundary Jitter Smoothing
        # -------------------------------------------------------------------
        n = len(all_words)

        def _snap_run(start_idx: int, end_idx: int, target_spk: str) -> None:
            for k in range(start_idx, end_idx + 1):
                all_words[k]["speaker"] = target_spk

        for window in (1, 2, 3):
            i = window
            while i < n - window:
                run_indices = list(range(i, i + window))
                left_spk  = all_words[i - 1]["speaker"]
                right_idx = i + window
                right_spk = all_words[right_idx]["speaker"] if right_idx < n else None

                # Flanking speakers must agree and differ from the run
                if left_spk != right_spk or right_spk is None:
                    i += 1
                    continue

                run_spks = [all_words[j]["speaker"] for j in run_indices]
                if any(s == left_spk for s in run_spks):
                    i += 1
                    continue

                gap_before = all_words[i]["start"]           - all_words[i - 1]["end"]
                gap_after  = (all_words[right_idx]["start"]  - all_words[i + window - 1]["end"]) if right_idx < n else 0.0
                run_dur    = all_words[i + window - 1]["end"] - all_words[i]["start"]
                prev_text  = str(all_words[i - 1]["word"]).strip()
                ends_sent  = any(prev_text.endswith(p) for p in (".", "?", "!"))

                if (
                    not ends_sent
                    and gap_before < 0.30
                    and gap_after  < 0.30
                    and run_dur    < (0.40 * window)
                ):
                    _snap_run(run_indices[0], run_indices[-1], left_spk)
                    i += window + 1
                else:
                    i += 1

        # -------------------------------------------------------------------
        # PASS 2: Conversational Turn Re-segmentation
        # -------------------------------------------------------------------
        refined_segments: List[Dict[str, Any]] = []
        curr_speaker: Optional[str] = None
        curr_words: List[Dict[str, Any]] = []

        for w in all_words:
            spk = w["speaker"]
            if curr_speaker is None:
                curr_speaker = spk
                curr_words = [w]
                continue

            last_w = curr_words[-1]
            gap = w["start"] - last_w["end"]
            seg_duration = w["end"] - curr_words[0]["start"]
            last_text = last_w["word"]
            ends_sentence = any(last_text.endswith(p) for p in (".", "?", "!"))

            should_split = (
                (spk != curr_speaker) or
                (gap >= 1.2) or
                (ends_sentence and (seg_duration >= 7.0 or gap >= 0.4)) or
                (seg_duration >= 18.0)
            )

            if should_split:
                seg_text = " ".join(cw["word"] for cw in curr_words).strip()
                refined_segments.append({
                    "start": curr_words[0]["start"],
                    "end": curr_words[-1]["end"],
                    "text": seg_text,
                    "source_text": None,
                    "speaker": curr_speaker,
                    "words": curr_words,
                })
                curr_speaker = spk
                curr_words = [w]
            else:
                curr_words.append(w)

        if curr_words:
            seg_text = " ".join(cw["word"] for cw in curr_words).strip()
            refined_segments.append({
                "start": curr_words[0]["start"],
                "end": curr_words[-1]["end"],
                "text": seg_text,
                "source_text": None,
                "speaker": curr_speaker,
                "words": curr_words,
            })

        # -------------------------------------------------------------------
        # PASS 2b: Short Orphan Segment Absorption
        # -------------------------------------------------------------------
        # Example of what this fixes:
        #   [SPEAKER_00: "My Lord"] [SPEAKER_01: "I"] [SPEAKER_00: "submit that the..."]
        #   becomes:
        #   [SPEAKER_00: "My Lord I submit that the..."]
        #
        # A segment is an "orphan" if:
        #   - It has <= 3 words OR lasts < 1.5s
        #   - Its immediate neighbours both belong to the SAME speaker
        #   - That speaker is different from the orphan's speaker
        i = 1
        while i < len(refined_segments) - 1:
            prev_seg = refined_segments[i - 1]
            curr_seg = refined_segments[i]
            next_seg = refined_segments[i + 1]
            seg_dur  = curr_seg["end"] - curr_seg["start"]
            n_words  = len(curr_seg.get("words", []))

            if (
                prev_seg["speaker"] == next_seg["speaker"]
                and curr_seg["speaker"] != prev_seg["speaker"]
                and (n_words <= 3 or seg_dur < 1.5)
            ):
                # Fix: absorb orphan into prev_seg
                prev_seg["words"].extend(curr_seg.get("words", []))
                prev_seg["end"] = curr_seg["end"]
                prev_seg["text"] = " ".join(w["word"] for w in prev_seg["words"]).strip()
                refined_segments.pop(i)
                # Do not increment — re-check at same index after pop
            else:
                i += 1

        # -------------------------------------------------------------------
        # PASS 3: Chronological Speaker Normalization
        # First speaker heard in the audio = SPEAKER_00
        # -------------------------------------------------------------------
        first_heard: Dict[str, float] = {}
        for seg in refined_segments:
            spk = seg["speaker"]
            if spk not in first_heard:
                first_heard[spk] = seg["start"]

        ordered_spks = sorted(first_heard.keys(), key=lambda s: first_heard[s])
        chrono_map = {orig: f"SPEAKER_{idx:02d}" for idx, orig in enumerate(ordered_spks)}

        for seg in refined_segments:
            seg["speaker"] = chrono_map.get(seg["speaker"], seg["speaker"])
            for w in seg.get("words", []):
                w["speaker"] = chrono_map.get(w.get("speaker"), w.get("speaker"))

        # PASS 4: Legal Terminology, Suit Citation & Digit Normalization
        normalized_segments = [normalize_segment(seg) for seg in refined_segments]

        unique_speakers = len(set(s["speaker"] for s in normalized_segments))
        logger.info(f"reconstruct_speaker_turns: {unique_speakers} unique speakers, {len(normalized_segments)} segments.")
        return normalized_segments, unique_speakers
