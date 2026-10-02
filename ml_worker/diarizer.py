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

            # Calibrate hyperparameters for courtrooms
            try:
                instantiate_params = {}
                threshold = getattr(WorkerConfig, "DIARIZATION_THRESHOLD", None)
                if threshold is not None:
                    instantiate_params["clustering"] = {"threshold": float(threshold)}
                min_off = getattr(WorkerConfig, "MIN_DURATION_OFF", 0.2)
                min_on = getattr(WorkerConfig, "MIN_DURATION_ON", 0.1)
                seg_dict = {}
                if min_off is not None:
                    seg_dict["min_duration_off"] = float(min_off)
                if min_on is not None:
                    seg_dict["min_duration_on"] = float(min_on)
                if seg_dict:
                    instantiate_params["segmentation"] = seg_dict
                if instantiate_params:
                    self.pipeline.instantiate(instantiate_params)
                    logger.info(f"Pyannote calibrated with court hyperparameters: {instantiate_params}")
                else:
                    logger.info("Using Pyannote default calibrated clustering threshold.")
            except Exception as thresh_err:
                logger.debug(f"Note on clustering/segmentation settings: {thresh_err}")
            
            self.is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load Pyannote diarization pipeline: {e}")
            self.pipeline = None
            self.is_loaded = False

    def diarize_dataframe(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ):
        import pandas as pd
        turns = self.diarize(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
        if not turns:
            return pd.DataFrame(columns=["start", "end", "speaker"])
        return pd.DataFrame(turns)

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

        # 2. Merge consecutive turns of the SAME speaker if separated by micro-pause <= 0.30s
        # AND no other speaker turn intervenes
        merged_turns = []
        for t in valid_turns:
            if not merged_turns:
                merged_turns.append(dict(t))
            else:
                last = merged_turns[-1]
                # Check if any other turn started between last["end"] and t["start"]
                gap = t["start"] - last["end"]
                if last["speaker"] == t["speaker"] and gap <= 0.30:
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

        # 5. Word-Level Speaker Assignment: Specificity-Weighted Overlap scoring + Proximity matching
        for w in flat_words:
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", w_start))
            if w_end < w_start:
                w_end = w_start + 0.1
            w_len = max(0.05, w_end - w_start)

            best_score = -1.0
            best_overlap = 0.0
            overlap_speaker = None
            for turn in merged_turns:
                ov_start = max(w_start, turn["start"])
                ov_end = min(w_end, turn["end"])
                ov = max(0.0, ov_end - ov_start)
                if ov > 0.0:
                    turn_dur = max(0.05, turn["end"] - turn["start"])
                    # Specificity scoring: penalize long background turns so short interjection turns win
                    score = ov / (turn_dur ** 0.15)
                    if score > best_score:
                        best_score = score
                        best_overlap = ov
                        overlap_speaker = turn["speaker"]

            if best_overlap > 0.0 and overlap_speaker is not None:
                w["speaker"] = overlap_speaker
                w["from_diarization"] = True
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
                w["from_diarization"] = False
            else:
                w["speaker"] = None
                w["from_diarization"] = False

        # 6. Fill unassigned words from closest neighbors
        last_known_mapped = speaker_map.get(ordered_speakers[0], "SPEAKER_00") if ordered_speakers else "SPEAKER_00"
        for i in range(len(flat_words)):
            if flat_words[i].get("speaker") is not None:
                last_known_mapped = flat_words[i]["speaker"]
            else:
                flat_words[i]["speaker"] = last_known_mapped

        # 7. Preserve Legal Honorific & Affirmation Honorific Pairs ("My Lord", "Your Honor", "Yes Sir")
        # Prevents splitting honorific titles across two speaker turns
        n_words = len(flat_words)
        HONORIFIC_LEADS = {"my", "your"}
        HONORIFIC_TRAILS = {"lord", "lordship", "honor", "honour", "worship", "lady", "friend", "learned"}
        
        for i in range(n_words - 1):
            w1 = flat_words[i]
            w2 = flat_words[i + 1]
            txt1 = str(w1.get("word", "")).lower().strip(".,!?;:\"'()[]{}")
            txt2 = str(w2.get("word", "")).lower().strip(".,!?;:\"'()[]{}")
            gap = float(w2.get("start", 0)) - float(w1.get("end", 0))

            if txt1 in HONORIFIC_LEADS and txt2 in HONORIFIC_TRAILS and gap <= 0.35:
                # Align title to whichever word came from a direct diarization turn
                if w2.get("from_diarization") and not w1.get("from_diarization"):
                    w1["speaker"] = w2["speaker"]
                else:
                    w2["speaker"] = w1["speaker"]

        # 7b. Gentle Collar Smoothing: ONLY smooth transient 1-word proximity blips that were NOT backed by Pyannote turns
        for i in range(1, n_words - 1):
            prev_w = flat_words[i - 1]
            curr_w = flat_words[i]
            next_w = flat_words[i + 1]

            prev_spk = prev_w.get("speaker")
            curr_spk = curr_w.get("speaker")
            next_spk = next_w.get("speaker")

            # Only smooth if curr_w was NOT from a direct Pyannote diarization turn
            if prev_spk == next_spk and curr_spk != prev_spk and not curr_w.get("from_diarization", False):
                gap_prev = float(curr_w.get("start", 0)) - float(prev_w.get("end", 0))
                gap_next = float(next_w.get("start", 0)) - float(curr_w.get("end", 0))
                if gap_prev < 0.25 and gap_next < 0.25:
                    curr_w["speaker"] = prev_spk

        # 7c. Legal Discourse Semantic Pass: Identify Judge vs Counsel acoustic roles and correct acoustic cross-talk
        speaker_scores = {}
        for w in flat_words:
            spk = w.get("speaker")
            if not spk:
                continue
            if spk not in speaker_scores:
                speaker_scores[spk] = {"judge_score": 0, "counsel_score": 0}

        n_w = len(flat_words)
        for i in range(n_w):
            w = flat_words[i]
            spk = w.get("speaker")
            if not spk:
                continue

            txt = str(w.get("word", "")).lower().strip(".,!?;:\"'()[]{}")

            # Counsel markers: addressing the court/bench
            if txt in ("milord", "lordship"):
                speaker_scores[spk]["counsel_score"] += 5
            elif txt == "my" and i + 1 < n_w:
                next_txt = str(flat_words[i + 1].get("word", "")).lower().strip(".,!?;:\"'()[]{}")
                if next_txt in ("lord", "lordship", "honor", "honour", "worship", "lady", "learned"):
                    speaker_scores[spk]["counsel_score"] += 5
            elif txt == "your" and i + 1 < n_w:
                next_txt = str(flat_words[i + 1].get("word", "")).lower().strip(".,!?;:\"'()[]{}")
                if next_txt in ("honor", "honour", "lordship", "worship"):
                    speaker_scores[spk]["counsel_score"] += 5

            # Judge markers: issuing rulings, bench commands, adjournment dates
            elif txt in ("sustained", "overruled", "adjourned", "struckout"):
                speaker_scores[spk]["judge_score"] += 4
            elif txt in ("court", "president") and i > 0:
                prev_txt = str(flat_words[i - 1].get("word", "")).lower().strip(".,!?;:\"'()[]{}")
                if prev_txt in ("the", "this", "honorable", "honourable"):
                    speaker_scores[spk]["judge_score"] += 3

        judge_speakers = set()
        counsel_speakers = set()
        for spk, scores in speaker_scores.items():
            if scores["judge_score"] > scores["counsel_score"] and scores["judge_score"] >= 3:
                judge_speakers.add(spk)
            elif scores["counsel_score"] > scores["judge_score"] and scores["counsel_score"] >= 3:
                counsel_speakers.add(spk)

        # Realign Honorific Addresses: If a phrase starts with "My Lord" / "Your Honor" and is assigned to a Judge speaker,
        # realign that honorific and its immediate spoken clause to Counsel!
        if judge_speakers and counsel_speakers:
            primary_counsel = list(counsel_speakers)[0]
            idx = 0
            while idx < n_w - 1:
                w1 = flat_words[idx]
                w2 = flat_words[idx + 1]
                t1 = str(w1.get("word", "")).lower().strip(".,!?;:\"'()[]{}")
                t2 = str(w2.get("word", "")).lower().strip(".,!?;:\"'()[]{}")

                is_honorific = (t1 in ("my", "your") and t2 in ("lord", "lordship", "honor", "honour", "worship")) or (t1 in ("milord", "milord,"))
                if is_honorific and w1.get("speaker") in judge_speakers:
                    j = idx
                    while j < min(n_w, idx + 12):
                        wj = flat_words[j]
                        wj["speaker"] = primary_counsel
                        txt_j = str(wj.get("word", "")).strip()
                        if any(txt_j.endswith(p) for p in (".", "?", "!")):
                            break
                        j += 1
                    idx = max(idx + 1, j)
                else:
                    idx += 1

        # 7d. Re-index speakers chronologically so the first spoken word is strictly SPEAKER_00
        first_appearance = []
        for w in flat_words:
            spk = w.get("speaker")
            if spk and spk not in first_appearance:
                first_appearance.append(spk)

        chrono_map = {orig: f"SPEAKER_{idx:02d}" for idx, orig in enumerate(first_appearance)}
        for w in flat_words:
            if w.get("speaker") in chrono_map:
                w["speaker"] = chrono_map[w["speaker"]]

        # 8. Reconstruct segments: Group consecutive words by speaker and natural sentence/pause boundaries
        refined_segments = []
        curr_speaker = None
        curr_words = []

        for w in flat_words:
            spk = w.get("speaker", "SPEAKER_00")
            if curr_speaker is None:
                curr_speaker = spk
                curr_words = [w]
                continue

            last_w = curr_words[-1]
            gap = float(w.get("start", 0.0)) - float(last_w.get("end", 0.0))
            seg_duration = float(w.get("end", 0.0)) - float(curr_words[0].get("start", 0.0))
            last_text = str(last_w.get("word", "")).strip()
            ends_sentence = any(last_text.endswith(p) for p in (".", "?", "!"))

            should_split = (
                (spk != curr_speaker) or
                (gap >= 1.2) or
                (ends_sentence and (seg_duration >= 6.0 or gap >= 0.4)) or
                (seg_duration >= 20.0)
            )

            if should_split:
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

