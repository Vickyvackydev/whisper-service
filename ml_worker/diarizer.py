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

            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=self.hf_token
                )
            except TypeError:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=self.hf_token
                )

            if self.device == "cuda":
                self.pipeline.to(self.device_obj)
                logger.info("Pyannote Diarization pipeline loaded on CUDA GPU.")
            else:
                logger.info("Pyannote Diarization pipeline loaded on CPU.")
            
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
            
            # Prepare kwargs for min/max speakers
            params = {}
            if min_speakers is not None and min_speakers > 0:
                params["min_speakers"] = min_speakers
            if max_speakers is not None and max_speakers > 0:
                params["max_speakers"] = max_speakers

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

            turns = []
            # Pyannote annotation extraction
            for turn, _, speaker in diarization_output.itertracks(yield_label=True):
                turns.append({
                    "start": round(float(turn.start), 3),
                    "end": round(float(turn.end), 3),
                    "speaker": str(speaker)
                })

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
        """
        if not diarization_turns:
            logger.warning("No diarization turns available. Defaulting all segments to SPEAKER_00.")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
                for w in seg.get("words", []):
                    w["speaker"] = "SPEAKER_00"
            return segments, 1

        # Build consistent speaker mapping (SPEAKER_00, SPEAKER_01, etc.)
        # Ordered by chronological first appearance
        speaker_first_seen = {}
        for t in diarization_turns:
            spk = t["speaker"]
            if spk not in speaker_first_seen:
                speaker_first_seen[spk] = t["start"]
        
        ordered_speakers = sorted(speaker_first_seen.keys(), key=lambda s: speaker_first_seen[s])
        speaker_map = {orig: f"SPEAKER_{i:02d}" for i, orig in enumerate(ordered_speakers)}

        normalized_turns = [
            {
                "start": t["start"],
                "end": t["end"],
                "speaker": speaker_map.get(t["speaker"], t["speaker"])
            }
            for t in diarization_turns
        ]

        def get_best_speaker(start: float, end: float, fallback: str = "SPEAKER_00") -> str:
            best_speaker = fallback
            max_overlap = 0.0

            # 1. Look for maximum temporal overlap
            for turn in normalized_turns:
                overlap_start = max(start, turn["start"])
                overlap_end = min(end, turn["end"])
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = turn["speaker"]

            # 2. If no direct overlap, find closest turn in time
            if max_overlap == 0.0 and normalized_turns:
                mid = (start + end) / 2.0
                closest_turn = min(
                    normalized_turns,
                    key=lambda t: abs(mid - ((t["start"] + t["end"]) / 2.0))
                )
                best_speaker = closest_turn["speaker"]

            return best_speaker

        current_speaker = normalized_turns[0]["speaker"] if normalized_turns else "SPEAKER_00"

        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            
            seg_speaker = get_best_speaker(seg_start, seg_end, current_speaker)
            seg["speaker"] = seg_speaker
            current_speaker = seg_speaker

            # Word-level speaker assignment
            for w in seg.get("words", []):
                w_start = float(w.get("start", seg_start))
                w_end = float(w.get("end", seg_end))
                w["speaker"] = get_best_speaker(w_start, w_end, seg_speaker)

        num_speakers = len(ordered_speakers) if ordered_speakers else 1
        logger.info(f"Assigned {num_speakers} unique speakers across {len(segments)} segments.")
        return segments, num_speakers
