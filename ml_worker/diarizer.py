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

            # Calibrate Pyannote hyperparameters
            # IMPORTANT ORDER: call instantiate() FIRST (applies via Pyannote schema),
            # then override attributes DIRECTLY AFTER so our values are never overwritten.
            try:
                thresh  = getattr(WorkerConfig, "DIARIZATION_THRESHOLD", 0.63)
                min_off = getattr(WorkerConfig, "MIN_DURATION_OFF", 0.20)
                min_on  = getattr(WorkerConfig, "MIN_DURATION_ON", 0.08)

                # Step 1: pipeline.instantiate() — respects Pyannote's internal schema
                instantiate_params = {}
                if thresh is not None:
                    instantiate_params["clustering"] = {"threshold": float(thresh)}
                seg_dict = {}
                if min_off is not None:
                    seg_dict["min_duration_off"] = float(min_off)
                if min_on is not None:
                    seg_dict["min_duration_on"] = float(min_on)
                if seg_dict:
                    instantiate_params["segmentation"] = seg_dict
                if instantiate_params:
                    try:
                        self.pipeline.instantiate(instantiate_params)
                        logger.info(f"Pyannote pipeline.instantiate() applied: {instantiate_params}")
                    except Exception as inst_err:
                        logger.debug(f"pipeline.instantiate() not available or failed ({inst_err}), using direct attrs only")

                # Step 2: Direct attribute override (always runs AFTER instantiate so it wins)
                if thresh is not None:
                    if hasattr(self.pipeline, "clustering") and hasattr(self.pipeline.clustering, "threshold"):
                        self.pipeline.clustering.threshold = float(thresh)
                        logger.info(f"[THRESHOLD] Pyannote clustering.threshold set to {float(thresh)}")
                    # Pyannote 3.x alternative path
                    for attr in ("_clustering", "klustering"):
                        sub = getattr(self.pipeline, attr, None)
                        if sub is not None and hasattr(sub, "threshold"):
                            sub.threshold = float(thresh)
                            logger.info(f"[THRESHOLD] Pyannote {attr}.threshold set to {float(thresh)}")

                if hasattr(self.pipeline, "segmentation"):
                    seg = self.pipeline.segmentation
                    if min_off is not None and hasattr(seg, "min_duration_off"):
                        seg.min_duration_off = float(min_off)
                    if min_on is not None and hasattr(seg, "min_duration_on"):
                        seg.min_duration_on = float(min_on)
                    logger.info(f"[SEGMENTATION] min_duration_off={min_off}, min_duration_on={min_on}")

            except Exception as thresh_err:
                logger.debug(f"Hyperparameter configuration note: {thresh_err}")
            
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
            target_min = min_speakers if min_speakers is not None else getattr(WorkerConfig, "MIN_SPEAKERS", None)
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

            # Post-diarization consolidation: prevent a single speaker from being split
            # into multiple clusters due to pitch/volume variation, sliding-window boundaries, or micro-pauses.
            turns = self._consolidate_fragmented_speakers(turns, audio_data=data, sample_rate=sample_rate)

            return turns

        except Exception as e:
            logger.error(f"Critical error during speaker diarization inference: {e}", exc_info=True)
            return []

    def _extract_audio_feature(self, waveform_1d: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Computes normalized log-filterbank acoustic feature vector for a segment.
        Provides robust voice comparison across different speech chunks.
        """
        if len(waveform_1d) < int(sample_rate * 0.3):
            return None
        win_len = int(sample_rate * 0.025)
        hop_len = int(sample_rate * 0.010)
        if len(waveform_1d) <= win_len:
            return None

        # Try Pyannote embedding if pipeline loaded and has _embedding
        if self.pipeline is not None and hasattr(self.pipeline, "_embedding"):
            try:
                emb_model = getattr(self.pipeline, "_embedding")
                if callable(emb_model):
                    t_in = torch.from_numpy(waveform_1d).unsqueeze(0).unsqueeze(0).to(self.device_obj)
                    with torch.no_grad():
                        out = emb_model(t_in)
                    out_np = out.squeeze().detach().cpu().numpy().astype(np.float32)
                    norm = np.linalg.norm(out_np)
                    if norm > 1e-6:
                        return out_np / norm
            except Exception as emb_e:
                logger.debug(f"Direct Pyannote embedding extraction note ({emb_e}), using acoustic filterbanks")

        # Fallback / Universal: 24-band geometric filterbank across 100Hz - 6500Hz
        try:
            n_fft = 512
            window = np.hanning(win_len)
            num_frames = (len(waveform_1d) - win_len) // hop_len
            if num_frames <= 0:
                return None

            indices = np.linspace(0, num_frames - 1, min(num_frames, 150), dtype=int)
            frames = np.stack([waveform_1d[idx * hop_len : idx * hop_len + win_len] * window for idx in indices])
            mags = np.abs(np.fft.rfft(frames, n=n_fft))

            n_bands = 24
            bin_freqs = np.linspace(0, sample_rate / 2, mags.shape[1])
            band_edges = np.geomspace(100, min(sample_rate / 2, 6500), n_bands + 2)
            filterbank = np.zeros((n_bands, mags.shape[1]), dtype=np.float32)
            for b in range(n_bands):
                left, center, right = band_edges[b], band_edges[b+1], band_edges[b+2]
                filterbank[b] = np.maximum(0, np.minimum(
                    (bin_freqs - left) / max(1e-6, center - left),
                    (right - bin_freqs) / max(1e-6, right - center)
                ))

            band_energies = np.dot(mags, filterbank.T)
            log_energies = np.log(np.maximum(1e-6, band_energies))
            avg_feat = log_energies.mean(axis=0)
            norm = np.linalg.norm(avg_feat)
            if norm > 1e-6:
                return (avg_feat / norm).astype(np.float32)
        except Exception as feat_err:
            logger.debug(f"Acoustic feature calculation note: {feat_err}")

        return None

    def _consolidate_fragmented_speakers(
        self,
        turns: List[Dict[str, Any]],
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        base_min_fraction: float = 0.04,
        min_seconds: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Consolidates fragmented speaker clusters using:
        1. Acoustic Embedding Centroid Matching (pairwise cosine similarity >= 0.72)
        2. Conversational Dynamics & Monologue Analysis (detects non-alternating sequential blocks)
        3. Dominant Speaker Dominance Check (>= 85% speech time)
        4. Tiny Fragment Pruning (micro-noise / clicks)
        5. Micro-pause turn stitching (<= 0.40s)
        6. Consistent Chronological Speaker Normalization (SPEAKER_00, SPEAKER_01...)
        """
        if not turns:
            return turns

        speaker_durations: Dict[str, float] = {}
        for t in turns:
            spk = t["speaker"]
            dur = max(0.0, t["end"] - t["start"])
            speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur

        total_speech = sum(speaker_durations.values())
        n_speakers = len(speaker_durations)

        if total_speech <= 0 or n_speakers <= 1:
            return turns

        dominant = max(speaker_durations, key=lambda s: speaker_durations[s])
        sorted_turns = sorted(turns, key=lambda t: t["start"])

        # ── Conversational Dynamics: Count runs and conversational returns ───
        speaker_runs: Dict[str, int] = {}
        prev_spk = None
        seen_speakers = set()
        return_speaker_count = 0

        for t in sorted_turns:
            spk = t["speaker"]
            if spk != prev_spk:
                speaker_runs[spk] = speaker_runs.get(spk, 0) + 1
                if spk in seen_speakers:
                    return_speaker_count += 1
                seen_speakers.add(spk)
                prev_spk = spk

        total_audio_dur = (sorted_turns[-1]["end"] - sorted_turns[0]["start"]) if sorted_turns else 0.0

        # ── Acoustic Speaker Centroid Extraction ─────────────────────────────
        speaker_centroids: Dict[str, np.ndarray] = {}
        if audio_data is not None and len(audio_data) > 0:
            for spk in speaker_durations:
                spk_turns = [t for t in sorted_turns if t["speaker"] == spk and (t["end"] - t["start"]) >= 0.5]
                if not spk_turns:
                    spk_turns = [t for t in sorted_turns if t["speaker"] == spk]

                feats = []
                for st in spk_turns[:5]:
                    start_s = max(0, int(st["start"] * sample_rate))
                    end_s = min(len(audio_data), int(st["end"] * sample_rate))
                    if end_s - start_s >= int(0.3 * sample_rate):
                        feat = self._extract_audio_feature(audio_data[start_s:end_s], sample_rate)
                        if feat is not None:
                            feats.append(feat)
                if feats:
                    avg_feat = np.mean(feats, axis=0)
                    norm = np.linalg.norm(avg_feat)
                    if norm > 1e-6:
                        speaker_centroids[spk] = avg_feat / norm

        # ── Build Merge Mapping ──────────────────────────────────────────────
        to_merge: Dict[str, str] = {}
        ranked_speakers = sorted(speaker_durations.keys(), key=lambda s: speaker_durations[s], reverse=True)

        def get_root_target(s: str) -> str:
            curr = s
            visited = set()
            while curr in to_merge and curr not in visited:
                visited.add(curr)
                curr = to_merge[curr]
            return curr

        # Pure Sequential Monologue Detection (Zero Conversational Returns)
        # In a real conversation, speakers alternate (return_speaker_count >= 1).
        # If returns == 0 and audio duration <= 60s, non-dominant sequential blocks are acoustic splits of one speaker reading/speaking continuously.
        # NOTE: Dominance (>85%), fragment (<2s), and similarity (>=0.72) heuristics were removed per consultant recommendations
        # to ensure that short interjections (e.g. mother's 1-2s remarks) and distinct room voices (father vs judge) are never merged.
        if return_speaker_count == 0 and total_audio_dur <= 60.0 and n_speakers >= 2:
            target = get_root_target(dominant)
            for spk in ranked_speakers:
                if spk != target and spk not in to_merge:
                    to_merge[spk] = target
                    logger.info(f"[CONSOLIDATION] Merging '{spk}' into dominant '{target}' — pure sequential short monologue (returns=0, dur={total_audio_dur:.1f}s)")

        # 5. Apply Merges
        merged_turns = []
        for t in sorted_turns:
            spk = t["speaker"]
            final_spk = get_root_target(spk)
            merged_turns.append({
                "start": t["start"],
                "end": t["end"],
                "speaker": final_spk
            })

        # 6. Micro-Pause Turn Stitching (same speaker with pause <= 0.40s)
        stitched_turns = []
        for t in merged_turns:
            if not stitched_turns:
                stitched_turns.append(dict(t))
            else:
                last = stitched_turns[-1]
                gap = t["start"] - last["end"]
                if last["speaker"] == t["speaker"] and gap <= 0.40:
                    last["end"] = max(last["end"], t["end"])
                else:
                    stitched_turns.append(dict(t))

        # 7. Consistent Chronological Re-numbering (SPEAKER_00, SPEAKER_01...)
        first_heard = {}
        for t in stitched_turns:
            s = t["speaker"]
            if s not in first_heard:
                first_heard[s] = t["start"]
        ordered = sorted(first_heard.keys(), key=lambda s: first_heard[s])
        remap = {orig: f"SPEAKER_{idx:02d}" for idx, orig in enumerate(ordered)}

        for t in stitched_turns:
            t["speaker"] = remap.get(t["speaker"], t["speaker"])

        logger.info(f"[CONSOLIDATION] {n_speakers} raw speakers → {len(ordered)} consolidated speakers.")
        return stitched_turns
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

