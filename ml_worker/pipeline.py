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
                logger.info(f"[{job_id}] Running {self.diarizer.backend.upper()} diarization on {wav_path}...")
                diarize_df = self.diarizer.diarize_dataframe(wav_path)
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

        # Legal honorifics indicating counsel speaking to the bench
        HONORIFIC_LEGAL_COUNSEL_PATTERNS = [
            r"\bmy\s+lord\b", r"\bmilord\b", r"\bme\s+lord\b", r"\bmy\s+noble\s+lord\b",
            r"\byour\s+lordship\b", r"\byour\s+lordships\b", r"\byour\s+honou?r\b",
            r"\byour\s+ladyship\b", r"\byour\s+worship\b",
            r"\bas\s+the\s+court\s+pleases\b", r"\bas\s+your\s+lordship\b",
            r"\bif\s+your\s+lordship\b", r"\bcourt\s+pleases\b"
        ]
        re_counsel_honorifics = [re.compile(p, re.IGNORECASE) for p in HONORIFIC_LEGAL_COUNSEL_PATTERNS]

        # Judge authority patterns (inquiries, directions, remarks about counsel/judge)
        JUDGE_AUTHORITY_PATTERNS = [
            r"\bcan\s+i\s+see\s+the\s+(?:prayer|order)\b",
            r"\bshow\s+me\s+the\s+order\b",
            r"\bwhat\s+are\s+you\s+talking\s+about\b",
            r"\bwhat\s+are\s+you\s+saying\b",
            r"\bchoose\s+the\s+one\s+you\s+want\b",
            r"\bi\s+don't\s+like\s+this\s+idea\s+of\s+counc?il\b",
            r"\bso\s+your\s+understanding\s+of\b",
            r"\bdoes\s+it\s+say\b"
        ]
        re_judge_authority = [re.compile(p, re.IGNORECASE) for p in JUDGE_AUTHORITY_PATTERNS]

        processed_segments: List[Dict[str, Any]] = []

        # -------------------------------------------------------------------
        # STAGE 1: Process Each Segment Intact (Word-level preservation)
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

            # Calculate dominant speaker for this segment based on word durations
            spk_durations: Dict[str, float] = {}
            for w in clean_words:
                spk = w["speaker"]
                dur = max(0.01, w["end"] - w["start"])
                spk_durations[spk] = spk_durations.get(spk, 0.0) + dur

            dominant_spk = max(spk_durations, key=spk_durations.get) if spk_durations else (seg.get("speaker") or "SPEAKER_00")

            # Check if there is a genuine intra-segment speaker handoff
            # Only split if two distinct speakers each speak for >= 0.8s
            # AND a natural clause boundary separates them.
            sub_chunks = []
            curr_chunk_words = []
            curr_chunk_spk = None

            for w in clean_words:
                w_spk = w["speaker"]
                if curr_chunk_spk is None:
                    curr_chunk_spk = w_spk
                    curr_chunk_words = [w]
                elif w_spk == curr_chunk_spk:
                    curr_chunk_words.append(w)
                else:
                    # Potential intra-segment handoff
                    chunk_dur = curr_chunk_words[-1]["end"] - curr_chunk_words[0]["start"]
                    last_w_txt = curr_chunk_words[-1]["word"]
                    ends_clause = any(last_w_txt.endswith(p) for p in (".", "?", "!", ",", ";"))
                    gap = w["start"] - curr_chunk_words[-1]["end"]

                    if chunk_dur >= 0.8 and (ends_clause or gap >= 0.35):
                        sub_chunks.append((curr_chunk_spk, curr_chunk_words))
                        curr_chunk_spk = w_spk
                        curr_chunk_words = [w]
                    else:
                        # Absorb micro-jitter into current chunk
                        w["speaker"] = curr_chunk_spk
                        curr_chunk_words.append(w)

            if curr_chunk_words:
                sub_chunks.append((curr_chunk_spk, curr_chunk_words))

            if len(sub_chunks) <= 1:
                # Intact segment: assign dominant speaker to all words
                for w in clean_words:
                    w["speaker"] = dominant_spk
                processed_segments.append({
                    "start": clean_words[0]["start"] if clean_words else seg_start,
                    "end": clean_words[-1]["end"] if clean_words else seg_end,
                    "text": " ".join(cw["word"] for cw in clean_words).strip() if clean_words else raw_text,
                    "source_text": seg.get("source_text"),
                    "speaker": dominant_spk,
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

        # -------------------------------------------------------------------
        # STAGE 2: Conversational Legal Attribution & Speaker Consolidation
        # -------------------------------------------------------------------
        # Identify the primary speakers.
        # In a courtroom:
        # Speaker 0 (usually first speaker) = Counsel addressing "My Lord"
        # Speaker 1 = Judge presiding and inquiring
        # Extraneous clusters (e.g. SPEAKER_02) that acoustic diarization produced
        # due to temporal pauses or pitch shifts are merged into the appropriate role.

        # 1. Identify which speaker is Counsel vs Judge based on linguistic markers
        speaker_counsel_votes: Dict[str, int] = {}
        speaker_judge_votes: Dict[str, int] = {}
        speaker_total_dur: Dict[str, float] = {}

        for seg in processed_segments:
            spk = seg["speaker"]
            dur = max(0.05, seg["end"] - seg["start"])
            speaker_total_dur[spk] = speaker_total_dur.get(spk, 0.0) + dur
            txt = seg["text"].lower()

            if any(p.search(txt) for p in re_counsel_honorifics):
                speaker_counsel_votes[spk] = speaker_counsel_votes.get(spk, 0) + 1
            if any(p.search(txt) for p in re_judge_authority):
                speaker_judge_votes[spk] = speaker_judge_votes.get(spk, 0) + 1

        # Determine Primary Counsel ID
        counsel_candidates = [s for s in speaker_counsel_votes if speaker_counsel_votes[s] > 0]
        if counsel_candidates:
            counsel_spk = max(counsel_candidates, key=lambda s: speaker_counsel_votes[s])
        else:
            counsel_spk = processed_segments[0]["speaker"]

        # Determine Primary Judge ID
        judge_candidates = [s for s in speaker_judge_votes if s != counsel_spk and speaker_judge_votes[s] > 0]
        if judge_candidates:
            judge_spk = max(judge_candidates, key=lambda s: speaker_judge_votes[s])
        else:
            other_spks = [s for s in speaker_total_dur if s != counsel_spk]
            judge_spk = max(other_spks, key=lambda s: speaker_total_dur[s]) if other_spks else None

        # Consolidate extraneous clusters (e.g. SPEAKER_02) into Counsel or Judge
        cluster_remap: Dict[str, str] = {}
        for spk in speaker_total_dur:
            if spk in (counsel_spk, judge_spk):
                continue
            c_votes = speaker_counsel_votes.get(spk, 0)
            j_votes = speaker_judge_votes.get(spk, 0)
            if c_votes > j_votes:
                cluster_remap[spk] = counsel_spk
            elif j_votes > c_votes and judge_spk is not None:
                cluster_remap[spk] = judge_spk
            elif judge_spk is not None:
                cluster_remap[spk] = judge_spk

        # Apply cluster consolidation
        for seg in processed_segments:
            orig = seg["speaker"]
            if orig in cluster_remap:
                target = cluster_remap[orig]
                seg["speaker"] = target
                for w in seg.get("words", []):
                    w["speaker"] = target

        # 2. Segment-level honorific correction:
        # A segment containing "My Lord" or "As the Court pleases" MUST be Counsel!
        # A segment containing Judge authority MUST be Judge!
        for seg in processed_segments:
            txt = seg["text"].lower()
            if any(p.search(txt) for p in re_counsel_honorifics):
                if seg["speaker"] != counsel_spk:
                    seg["speaker"] = counsel_spk
                    for w in seg.get("words", []):
                        w["speaker"] = counsel_spk
            elif any(p.search(txt) for p in re_judge_authority) and judge_spk is not None:
                if seg["speaker"] != judge_spk:
                    seg["speaker"] = judge_spk
                    for w in seg.get("words", []):
                        w["speaker"] = judge_spk

        # 3. Alternating Short Interjection Smoothing
        SHORT_AFFIRMATIONS = {"sir?", "sir", "yes sir", "yes my lord", "no my lord", "as the court pleases"}
        for i in range(len(processed_segments)):
            seg = processed_segments[i]
            txt_clean = seg["text"].lower().strip(".,!?;:\"' ")

            # Short Counsel interjections
            if txt_clean in SHORT_AFFIRMATIONS or (len(seg.get("words", [])) <= 3 and any(p.search(seg["text"].lower()) for p in re_counsel_honorifics)):
                if seg["speaker"] != counsel_spk:
                    seg["speaker"] = counsel_spk
                    for w in seg.get("words", []):
                        w["speaker"] = counsel_spk

            # Short Judge questions between Counsel turns (e.g. "...not included." -> "Can I see the prayer?" -> "Sir?")
            if 0 < i < len(processed_segments) - 1 and judge_spk is not None:
                prev_seg = processed_segments[i - 1]
                next_seg = processed_segments[i + 1]
                if prev_seg["speaker"] == counsel_spk and next_seg["speaker"] == counsel_spk:
                    if seg["text"].strip().endswith("?") and not any(p.search(seg["text"].lower()) for p in re_counsel_honorifics):
                        if seg["speaker"] != judge_spk:
                            seg["speaker"] = judge_spk
                            for w in seg.get("words", []):
                                w["speaker"] = judge_spk

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

        unique_speakers = len(set(s["speaker"] for s in normalized_segments))
        logger.info(f"reconstruct_speaker_turns: {unique_speakers} unique speakers, {len(normalized_segments)} segments preserved.")
        return normalized_segments, unique_speakers
