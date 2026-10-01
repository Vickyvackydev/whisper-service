import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import torch
import numpy as np
from ml_worker.config import WorkerConfig

logger = logging.getLogger("ml_worker.diarizer")

class SpeakerDiarizer:
    def __init__(self):
        self.pipeline = None
        self.is_loaded = False
        self.model_name = WorkerConfig.DIARIZATION_MODEL
        self.hf_token = WorkerConfig.HF_TOKEN
        self.device = "cuda" if (torch.cuda.is_available() and WorkerConfig.WHISPER_DEVICE != "cpu") else "cpu"
        self.device_obj = torch.device(self.device)

    def load_model(self):
        if not self.hf_token:
            logger.warning("HF_TOKEN is not configured. Pyannote speaker diarization will be unavailable.")
            return

        try:
            logger.info(f"Loading Diarization pipeline: {self.model_name} on {self.device}...")
            from pyannote.audio import Pipeline

            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.hf_token
            )

            if self.device == "cuda":
                self.pipeline.to(self.device_obj)
                logger.info("Pyannote Diarization pipeline loaded on CUDA GPU.")
            else:
                logger.info("Pyannote Diarization pipeline loaded on CPU.")

            # Calibrate clustering threshold if custom override is set
            try:
                threshold = getattr(WorkerConfig, "DIARIZATION_THRESHOLD", None)
                if threshold is not None:
                    self.pipeline.instantiate({"clustering": {"threshold": float(threshold)}})
                    logger.info(f"Pyannote clustering threshold calibrated to {threshold}.")
                else:
                    logger.info("Using Pyannote default calibrated clustering threshold.")
            except Exception as thresh_err:
                logger.debug(f"Note on clustering threshold setting: {thresh_err}")
            
            self.is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load Pyannote diarization pipeline: {e}")
            self.pipeline = None
            self.is_loaded = False

    def diarize(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs diarization on audio and returns list of speaker intervals:
        [{ "start": 0.5, "end": 4.2, "speaker": "SPEAKER_00" }, ...]
        """
        if not self.is_loaded or self.pipeline is None:
            logger.warning("Diarization pipeline not loaded.")
            return []

        try:
            logger.info(f"Running speaker diarization on {audio_path.name}...")
            import soundfile as sf
            data, sample_rate = sf.read(str(audio_path), dtype="float32")
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            # Ensure 1D float32 array
            data = np.ascontiguousarray(data, dtype=np.float32)
            
            # Prepare kwargs for min/max speakers (optional overrides)
            params = {}
            target_min = min_speakers if min_speakers is not None else getattr(WorkerConfig, "MIN_SPEAKERS", 2)
            if target_min is not None and target_min > 0:
                params["min_speakers"] = target_min
            target_max = max_speakers if max_speakers is not None else getattr(WorkerConfig, "MAX_SPEAKERS", None)
            if target_max is not None and target_max > 0:
                params["max_speakers"] = target_max

            diarization_output = None

            # Attempt 1: Run with GPU tensor
            try:
                waveform_cuda = torch.from_numpy(data).unsqueeze(0).to(self.device_obj)
                diarization_output = self.pipeline(
                    {"waveform": waveform_cuda, "sample_rate": sample_rate},
                    **params
                )
            except Exception as e_cuda:
                logger.warning(f"GPU tensor diarization failed ({e_cuda}), trying CPU tensor...")
                # Attempt 2: Run with CPU tensor
                try:
                    waveform_cpu = torch.from_numpy(data).unsqueeze(0).cpu()
                    diarization_output = self.pipeline(
                        {"waveform": waveform_cpu, "sample_rate": sample_rate},
                        **params
                    )
                except Exception as e_cpu:
                    logger.warning(f"CPU tensor diarization failed ({e_cpu}), trying file path...")
                    # Attempt 3: Run with string file path
                    diarization_output = self.pipeline(str(audio_path), **params)

            if diarization_output is None:
                logger.warning("Diarization output is empty.")
                return []

            # Handle DiarizeOutput, Annotation, or tuple return types
            annotation = diarization_output
            if hasattr(diarization_output, "speaker_diarization"):
                annotation = diarization_output.speaker_diarization
            elif hasattr(diarization_output, "annotation"):
                annotation = diarization_output.annotation
            elif isinstance(diarization_output, (tuple, list)) and len(diarization_output) > 0:
                annotation = diarization_output[0]

            turns = []
            if hasattr(annotation, "itertracks"):
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    turns.append({
                        "start": round(float(turn.start), 3),
                        "end": round(float(turn.end), 3),
                        "speaker": str(speaker)
                    })
            elif hasattr(annotation, "itersegments"):
                for seg in annotation.itersegments():
                    turns.append({
                        "start": round(float(seg.start), 3),
                        "end": round(float(seg.end), 3),
                        "speaker": str(getattr(seg, "label", "SPEAKER_00"))
                    })
            else:
                logger.warning(f"Unknown annotation structure: {type(annotation)}, attrs={dir(annotation)}")

            unique_detected = len(set(t["speaker"] for t in turns))
            logger.info(f"Diarization inference successful: detected {len(turns)} turns across {unique_detected} distinct speakers.")
            return turns

        except Exception as e:
            logger.error(f"Critical error during speaker diarization inference: {e}", exc_info=True)
            return []

    def assign_speakers(
        self,
        segments: List[Dict[str, Any]],
        diarization_turns: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Maps diarized speaker intervals to Whisper transcription segments and words.
        Uses global temporal overlap duration, nearest-turn proximity matching,
        conversational boundary snapping, and median-filter collar smoothing.
        """
        if not segments:
            return [], 0

        if not diarization_turns:
            logger.warning("No diarization turns available. Defaulting all segments to SPEAKER_00.")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
                for w in seg.get("words", []):
                    w["speaker"] = "SPEAKER_00"
            return segments, 1

        # 1. Clean and sort diarization turns
        valid_turns = []
        for t in diarization_turns:
            try:
                s = float(t["start"])
                e = float(t["end"])
                spk = str(t.get("speaker", "SPEAKER_00")).strip()
                if e > s and spk:
                    valid_turns.append({"start": round(s, 3), "end": round(e, 3), "speaker": spk})
            except (ValueError, KeyError, TypeError):
                continue

        if not valid_turns:
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
                for w in seg.get("words", []):
                    w["speaker"] = "SPEAKER_00"
            return segments, 1

        valid_turns.sort(key=lambda t: t["start"])

        # 2. Merge consecutive turns of the SAME speaker if separated by micro-pause <= 0.45s
        merged_turns = []
        for t in valid_turns:
            if not merged_turns:
                merged_turns.append(dict(t))
            else:
                last = merged_turns[-1]
                if last["speaker"] == t["speaker"] and (t["start"] - last["end"]) <= 0.45:
                    last["end"] = max(last["end"], t["end"])
                else:
                    merged_turns.append(dict(t))

        # 3. Build consistent chronological speaker mapping (SPEAKER_00, SPEAKER_01, ...)
        speaker_first_seen = {}
        for t in merged_turns:
            spk = t["speaker"]
            if spk not in speaker_first_seen:
                speaker_first_seen[spk] = t["start"]
        ordered_speakers = sorted(speaker_first_seen.keys(), key=lambda s: speaker_first_seen[s])
        speaker_map = {orig: f"SPEAKER_{i:02d}" for i, orig in enumerate(ordered_speakers)}

        for t in merged_turns:
            t["speaker"] = speaker_map.get(t["speaker"], t["speaker"])

        # 4. Flatten all words across all segments with references
        flat_words = []
        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            words = seg.get("words", [])
            if not words:
                txt = seg.get("text", "").strip()
                if txt:
                    tokens = txt.split()
                    dur = max(0.1, seg_end - seg_start)
                    step = dur / len(tokens)
                    words = [
                        {
                            "word": tok,
                            "start": round(seg_start + i * step, 3),
                            "end": round(seg_start + (i + 1) * step, 3),
                            "speaker": "SPEAKER_00"
                        }
                        for i, tok in enumerate(tokens)
                    ]
                    seg["words"] = words
            for w in words:
                flat_words.append(w)

        if not flat_words:
            return segments, 1

        # 5. Word-Level Speaker Assignment: Overlap scoring + Proximity matching
        for w in flat_words:
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", w_start))
            if w_end < w_start:
                w_end = w_start + 0.1

            # A. Overlap duration with turns
            best_overlap = 0.0
            overlap_speaker = None
            for turn in merged_turns:
                ov_start = max(w_start, turn["start"])
                ov_end = min(w_end, turn["end"])
                ov = max(0.0, ov_end - ov_start)
                if ov > best_overlap:
                    best_overlap = ov
                    overlap_speaker = turn["speaker"]

            if best_overlap > 0.0 and overlap_speaker is not None:
                w["speaker"] = overlap_speaker
                continue

            # B. Nearest turn proximity (when word falls in micro-pause or VAD edge)
            best_dist = float("inf")
            closest_speaker = None
            for turn in merged_turns:
                if w_end <= turn["start"]:
                    dist = turn["start"] - w_end
                elif w_start >= turn["end"]:
                    dist = w_start - turn["end"]
                else:
                    dist = 0.0
                if dist < best_dist:
                    best_dist = dist
                    closest_speaker = turn["speaker"]

            if best_dist <= 0.85 and closest_speaker is not None:
                w["speaker"] = closest_speaker
            else:
                w["speaker"] = None

        # 6. Fill unassigned words from closest neighbors
        last_known = ordered_speakers[0] if ordered_speakers else "SPEAKER_00"
        last_known_mapped = speaker_map.get(last_known, "SPEAKER_00")
        for i in range(len(flat_words)):
            if flat_words[i].get("speaker") is not None:
                last_known_mapped = flat_words[i]["speaker"]
            else:
                flat_words[i]["speaker"] = last_known_mapped

        # 7. Word-Level Smoothing Collar: Eliminate transient 1-word and 2-word acoustic blips/chattering mid-sentence
        n_words = len(flat_words)
        # Pass A: 1-word mid-sentence blip [A, B, A] -> [A, A, A]
        for i in range(1, n_words - 1):
            prev_w = flat_words[i - 1]
            curr_w = flat_words[i]
            next_w = flat_words[i + 1]

            prev_spk = prev_w.get("speaker")
            curr_spk = curr_w.get("speaker")
            next_spk = next_w.get("speaker")

            if prev_spk == next_spk and curr_spk != prev_spk:
                gap_prev = float(curr_w.get("start", 0)) - float(prev_w.get("end", 0))
                gap_next = float(next_w.get("start", 0)) - float(curr_w.get("end", 0))

                txt_prev = str(prev_w.get("word", "")).strip()
                txt_curr = str(curr_w.get("word", "")).strip()
                has_punct = any(txt_prev.endswith(p) for p in [".", "?", "!"]) or any(txt_curr.endswith(p) for p in [".", "?", "!"])
                clean_curr = txt_curr.lower().strip(".,!?;:\"'()[]{}")
                is_affirmation = clean_curr in ("yes", "no", "okay", "yeah", "right")

                if not has_punct and not is_affirmation and gap_prev < 0.45 and gap_next < 0.45:
                    curr_w["speaker"] = prev_spk

        # Pass B: 2-word mid-sentence blip [A, B, B, A] -> [A, A, A, A]
        for i in range(1, n_words - 2):
            prev_w = flat_words[i - 1]
            w1 = flat_words[i]
            w2 = flat_words[i + 1]
            next_w = flat_words[i + 2]

            prev_spk = prev_w.get("speaker")
            spk1 = w1.get("speaker")
            spk2 = w2.get("speaker")
            next_spk = next_w.get("speaker")

            if prev_spk == next_spk and spk1 == spk2 and spk1 != prev_spk:
                gap_prev = float(w1.get("start", 0)) - float(prev_w.get("end", 0))
                gap_next = float(next_w.get("start", 0)) - float(w2.get("end", 0))

                txt_prev = str(prev_w.get("word", "")).strip()
                txt_w2 = str(w2.get("word", "")).strip()
                has_punct = any(txt_prev.endswith(p) for p in [".", "?", "!"]) or any(txt_w2.endswith(p) for p in [".", "?", "!"])
                c1 = str(w1.get("word", "")).lower().strip(".,!?;:\"'()[]{}")
                c2 = str(w2.get("word", "")).lower().strip(".,!?;:\"'()[]{}")
                is_affirmation = (c1 in ("yes", "no", "yeah")) and (c2 in ("sir", "my", "lord", "no", "yes"))

                if not has_punct and not is_affirmation and gap_prev < 0.40 and gap_next < 0.40:
                    w1["speaker"] = prev_spk
                    w2["speaker"] = prev_spk

        # 8. Reconstruct segments: Group consecutive words by speaker
        refined_segments = []
        curr_speaker = None
        curr_words = []

        for w in flat_words:
            spk = w.get("speaker", "SPEAKER_00")
            if curr_speaker is None:
                curr_speaker = spk
                curr_words = [w]
            elif spk != curr_speaker:
                # Flush segment
                seg_text = " ".join(str(cw.get("word", "")).strip() for cw in curr_words).strip()
                refined_segments.append({
                    "start": curr_words[0].get("start", 0.0),
                    "end": curr_words[-1].get("end", 0.0),
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
            seg_text = " ".join(str(cw.get("word", "")).strip() for cw in curr_words).strip()
            refined_segments.append({
                "start": curr_words[0].get("start", 0.0),
                "end": curr_words[-1].get("end", 0.0),
                "text": seg_text,
                "source_text": None,
                "speaker": curr_speaker,
                "words": curr_words
            })

        unique_speakers = len(set(s["speaker"] for s in refined_segments))
        logger.info(f"Assigned {unique_speakers} unique speakers across {len(refined_segments)} refined segments.")
        return refined_segments, unique_speakers

