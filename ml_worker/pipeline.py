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

    @staticmethod
    def _split_segment_runs(clean_words: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Splits contiguous words inside a single Whisper segment into distinct speaker sub-chunks
        when a genuine interjection/handoff occurs (e.g. cross-talk without pause).
        Filters out single-word acoustic jitter (< 0.35s) so natural sentences aren't fractured.
        """
        if not clean_words:
            return []

        # 1. Group contiguous words into initial speaker runs
        runs: List[Dict[str, Any]] = []
        for w in clean_words:
            w_spk = w.get("speaker", "SPEAKER_00")
            if not runs or runs[-1]["speaker"] != w_spk:
                runs.append({"speaker": w_spk, "words": [w]})
            else:
                runs[-1]["words"].append(w)

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
            r"\bis\s+that\s+what\s+(?:you|was)\s+granted\b"
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

        total_audio_speech = 0.0
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

        # Identify Primary Counsel (most honorifics addressed to bench)
        counsel_spk = None
        counsel_candidates = [s for s in speaker_counsel_votes if speaker_counsel_votes[s] > 0]
        if counsel_candidates:
            counsel_spk = max(counsel_candidates, key=lambda s: speaker_counsel_votes[s])
        else:
            counsel_spk = processed_segments[0]["speaker"]

        # Identify Primary Judge (The Bench)
        judge_spk = None
        judge_candidates = [s for s in speaker_judge_votes if speaker_judge_votes[s] > 0 and s != counsel_spk]
        if judge_candidates:
            judge_spk = max(judge_candidates, key=lambda s: speaker_judge_votes[s])
        else:
            # Score non-counsel speakers: lower honorific density + higher total duration = most likely Judge
            other_speakers = [s for s in speaker_total_dur if s != counsel_spk and speaker_total_dur[s] >= 1.0]
            if other_speakers:
                def judge_score(s: str) -> float:
                    c_votes = speaker_counsel_votes.get(s, 0)
                    dur = speaker_total_dur.get(s, 1.0)
                    return dur / (1.0 + (c_votes * 5.0))
                judge_spk = max(other_speakers, key=judge_score)

        logger.info(f"Courtroom Role Discovery: Counsel={counsel_spk} (votes={speaker_counsel_votes}), Judge={judge_spk} (votes={speaker_judge_votes})")

        # Micro-cluster pruning: only prune true acoustic glitches/clicks (< 0.8s, <= 1 word, isolated turn)
        # Legitimate 2nd / 3rd participants (e.g. Father, Registrar, 2nd Counsel, Witness) are strictly preserved!
        cluster_remap: Dict[str, str] = {}
        target_max_speakers = getattr(WorkerConfig, "MAX_SPEAKERS", None)

        for spk, dur in speaker_total_dur.items():
            if spk in (counsel_spk, judge_spk):
                continue
            word_cnt = speaker_word_counts.get(spk, 0)
            seg_cnt = speaker_seg_counts.get(spk, 0)

            # True acoustic phantom: sub-second duration (< 0.8s), only 1 word or noise, and single isolated turn
            is_micro_phantom = (dur < 0.8 and word_cnt <= 1 and seg_cnt <= 1)
            is_enforced_2spk = (target_max_speakers == 2 and len(speaker_total_dur) > 2)

            if is_micro_phantom or is_enforced_2spk:
                c_votes = speaker_counsel_votes.get(spk, 0)
                j_votes = speaker_judge_votes.get(spk, 0)
                if c_votes > j_votes:
                    cluster_remap[spk] = counsel_spk
                elif j_votes > c_votes and judge_spk is not None:
                    cluster_remap[spk] = judge_spk
                elif judge_spk is not None and spk != counsel_spk:
                    cluster_remap[spk] = judge_spk

        # Apply micro-cluster remapping
        if cluster_remap:
            for seg in processed_segments:
                orig = seg["speaker"]
                if orig in cluster_remap:
                    target = cluster_remap[orig]
                    seg["speaker"] = target
                    for w in seg.get("words", []):
                        w["speaker"] = target

        # Pass 1: Universal Courtroom Rule: The Judge NEVER addresses themselves as "My Lord" or "As the Court pleases"
        # If the Judge was acoustically misassigned to a segment with counsel honorifics, reassign to Counsel
        if counsel_spk is not None:
            for seg in processed_segments:
                if judge_spk is not None and seg["speaker"] == judge_spk:
                    txt = seg["text"].lower()
                    if any(p.search(txt) for p in re_counsel_honorifics):
                        seg["speaker"] = counsel_spk
                        for w in seg.get("words", []):
                            w["speaker"] = counsel_spk

        # Pass 1B: Universal Courtroom Rule: Only the Bench issues judicial orders, maintenance directives, and contempt warnings
        if judge_spk is not None:
            for seg in processed_segments:
                if seg["speaker"] != judge_spk:
                    txt = seg["text"].lower()
                    if RE_JUDGE_DIRECTIVES.search(txt):
                        seg["speaker"] = judge_spk
                        for w in seg.get("words", []):
                            w["speaker"] = judge_spk

        # Pass 2: Conversational Alternation & Response Sandwiches
        SHORT_AFFIRMATIONS = {"sir?", "sir", "yes sir", "yes my lord", "no my lord", "as the court pleases"}
        if judge_spk is not None and counsel_spk is not None:
            for i in range(len(processed_segments)):
                seg = processed_segments[i]
                txt_clean = seg["text"].lower().strip(".,!?;:\"' ")

                # Case A: Bench Document Inquiries ("Can I see the prayer?", "Can I see the order?")
                if RE_BENCH_INQUIRY.search(txt_clean):
                    seg["speaker"] = judge_spk
                    continue

                # Case B: Submission Sandwich ("As the Court pleases")
                # When Counsel submits to the Bench, the preceding segment was the Judge
                if i > 0 and RE_SUBMISSION.search(txt_clean):
                    prev_seg = processed_segments[i - 1]
                    if prev_seg["speaker"] == counsel_spk:
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

                # Case E: Run-on Sentence Continuation across brief pauses (gap <= 0.8s)
                if i > 0:
                    prev_seg = processed_segments[i - 1]
                    gap = seg["start"] - prev_seg["end"]
                    prev_tokens = prev_seg["text"].lower().strip(".,!?;:\"' ").split()
                    prev_last_word = prev_tokens[-1] if prev_tokens else ""
                    if gap <= 0.8 and prev_last_word in DANGLING_CONJUNCTIONS:
                        first_word = seg["text"].lower().strip(".,!?;:\"' ").split()[0] if seg["text"].strip() else ""
                        if first_word not in {"i", "my", "we", "no", "yes"}:
                            if prev_seg["speaker"] == judge_spk and RE_JUDGE_DIRECTIVES.search(prev_seg["text"]):
                                seg["speaker"] = judge_spk
                            elif seg["speaker"] == counsel_spk and prev_seg["speaker"] == judge_spk:
                                prev_seg["speaker"] = counsel_spk

                # Case F: Short Affirmations ("Sir?", "Yes sir", "As the Court pleases")
                if txt_clean in SHORT_AFFIRMATIONS and seg["speaker"] == judge_spk:
                    seg["speaker"] = counsel_spk

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

        unique_speakers = len(set(s["speaker"] for s in normalized_segments))
        logger.info(f"reconstruct_speaker_turns: {unique_speakers} unique speakers, {len(normalized_segments)} segments preserved.")
        return normalized_segments, unique_speakers
