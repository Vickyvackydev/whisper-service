import gc
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import torch
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
            align_device = "cuda" if (torch.cuda.is_available() and WorkerConfig.WHISPER_DEVICE != "cpu") else "cpu"
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
                    if align_device == "cuda":
                        torch.cuda.empty_cache()

                    logger.info(f"[{job_id}] Phoneme forced alignment successful across {len(aligned_result.get('segments', []))} segments.")
                except Exception as align_err:
                    logger.warning(f"[{job_id}] WhisperX phoneme alignment skipped or failed ({align_err}). Retaining Whisper timestamps.")
                    aligned_result = {"segments": segments}

            # Diarization & Word Midpoint Stitching
            logger.info(f"[{job_id}] Diarization check: requested={enable_diarization}, model_loaded={self.diarizer.is_loaded}")
            if self.diarizer.is_loaded and enable_diarization:
                if progress_updater:
                    progress_updater("diarizing", 85)
                logger.info(f"[{job_id}] Running Pyannote diarization on {wav_path}...")
                diarize_df = self.diarizer.diarize_dataframe(wav_path)
                logger.info(f"[{job_id}] Diarization generated {len(diarize_df)} intervals.")

                if len(diarize_df) > 0:
                    try:
                        import whisperx
                        stitched = whisperx.assign_word_speakers(diarize_df, aligned_result, fill_nearest=True)
                        segments, num_speakers = self.reconstruct_speaker_turns(stitched)
                        logger.info(f"[{job_id}] AssemblyAI-grade speaker alignment complete: {num_speakers} unique speakers, {len(segments)} turns.")
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
        Splits immediately on any speaker transition (matching AssemblyAI behavior)
        while merging continuous speech from the same speaker.
        """
        all_words = []
        for seg in stitched_result.get("segments", []):
            words = seg.get("words", [])
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
                            "score": None
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
                            "score": round(float(w.get("score")), 3) if w.get("score") is not None else None
                        })

        if not all_words:
            return stitched_result.get("segments", []), 1

        # 1. Chronological speaker normalization (first speaker heard = SPEAKER_00)
        speaker_first_seen = {}
        for w in all_words:
            spk = w["speaker"]
            if spk not in speaker_first_seen:
                speaker_first_seen[spk] = w["start"]

        ordered = sorted(speaker_first_seen.keys(), key=lambda s: speaker_first_seen[s])
        chrono_map = {orig: f"SPEAKER_{i:02d}" for i, orig in enumerate(ordered)}
        for w in all_words:
            w["speaker"] = chrono_map.get(w["speaker"], w["speaker"])

        # 2. Re-segment contiguous words by speaker transitions
        refined_segments = []
        curr_speaker = None
        curr_words = []

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

            # Boundary rules:
            # - Speaker transition: ALWAYS split immediately
            # - Same speaker thinking pause: split if silence gap >= 1.8s
            # - Same speaker long paragraph: split at sentence end if duration >= 8s
            should_split = (
                (spk != curr_speaker) or
                (gap >= 1.8) or
                (ends_sentence and (seg_duration >= 8.0 or gap >= 0.5)) or
                (seg_duration >= 25.0)
            )

            if should_split:
                seg_text = " ".join(cw["word"] for cw in curr_words).strip()
                refined_segments.append({
                    "start": curr_words[0]["start"],
                    "end": curr_words[-1]["end"],
                    "text": seg_text,
                    "source_text": None,
                    "speaker": curr_speaker,
                    "words": curr_words
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
                "words": curr_words
            })

        # 3. Apply legal terminology & digit normalization
        normalized_segments = [normalize_segment(seg) for seg in refined_segments]

        unique_speakers = len(set(s["speaker"] for s in normalized_segments))
        return normalized_segments, unique_speakers
