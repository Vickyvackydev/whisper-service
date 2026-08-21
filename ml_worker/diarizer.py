import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import torch
from ml_worker.config import WorkerConfig

logger = logging.getLogger("ml_worker.diarizer")

class SpeakerDiarizer:
    def __init__(self):
        self.pipeline = None
        self.is_loaded = False
        self.model_name = WorkerConfig.DIARIZATION_MODEL
        self.hf_token = WorkerConfig.HF_TOKEN

    def load_model(self):
        if not self.hf_token:
            logger.warning("HF_TOKEN is not configured. Pyannote speaker diarization will be unavailable or use fallback.")
            return

        try:
            logger.info(f"Loading Diarization pipeline: {self.model_name}...")
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

            if torch.cuda.is_available() and WorkerConfig.WHISPER_DEVICE != "cpu":
                self.pipeline.to(torch.device("cuda"))
                logger.info("Diarization pipeline loaded on CUDA GPU.")
            else:
                logger.info("Diarization pipeline loaded on CPU.")
            
            self.is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load Pyannote diarization pipeline: {e}")
            self.pipeline = None
            self.is_loaded = False

    def diarize(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Runs diarization on 16kHz mono audio and returns speaker intervals:
        [{ "start": 0.5, "end": 4.2, "speaker": "SPEAKER_00" }, ...]
        """
        if not self.is_loaded or self.pipeline is None:
            logger.warning("Diarization pipeline not loaded. Assigning default single speaker SPEAKER_00.")
            return []

        try:
            logger.info(f"Running speaker diarization on {audio_path.name}...")
            import soundfile as sf
            data, sample_rate = sf.read(str(audio_path), dtype="float32")
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            waveform_tensor = torch.from_numpy(data).unsqueeze(0)

            try:
                diarization_output = self.pipeline({"waveform": waveform_tensor, "sample_rate": sample_rate})
            except Exception:
                diarization_output = self.pipeline(str(audio_path))
            
            turns = []
            # Pyannote annotation: turn (Segment), track, speaker (str)
            for turn, _, speaker in diarization_output.itertracks(yield_label=True):
                turns.append({
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker
                })
            
            logger.info(f"Diarization completed. Detected {len(turns)} speaker turn intervals.")
            return turns
        except Exception as e:
            logger.error(f"Error during speaker diarization inference: {e}")
            return []

    def assign_speakers(
        self,
        segments: List[Dict[str, Any]],
        diarization_turns: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Maps diarized speaker intervals to Whisper transcription segments and words
        based on maximum temporal overlap.
        """
        if not diarization_turns:
            # All default to SPEAKER_00
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
                for w in seg.get("words", []):
                    w["speaker"] = "SPEAKER_00"
            return segments, 1

        # Map speaker IDs consistently: e.g. SPEAKER_00, SPEAKER_01
        unique_speakers = sorted(list({t["speaker"] for t in diarization_turns}))
        speaker_map = {orig: f"SPEAKER_{i:02d}" for i, orig in enumerate(unique_speakers)}
        
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

            for turn in normalized_turns:
                overlap_start = max(start, turn["start"])
                overlap_end = min(end, turn["end"])
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = turn["speaker"]

            # If no direct overlap, find closest turn
            if max_overlap == 0.0 and normalized_turns:
                closest_dist = float("inf")
                for turn in normalized_turns:
                    mid = (start + end) / 2.0
                    turn_mid = (turn["start"] + turn["end"]) / 2.0
                    dist = abs(mid - turn_mid)
                    if dist < closest_dist:
                        closest_dist = dist
                        best_speaker = turn["speaker"]

            return best_speaker

        current_speaker = "SPEAKER_00"
        for seg in segments:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            seg_speaker = get_best_speaker(seg_start, seg_end, current_speaker)
            seg["speaker"] = seg_speaker
            current_speaker = seg_speaker

            # Word-level speaker assignment
            for w in seg.get("words", []):
                w_start = w.get("start", seg_start)
                w_end = w.get("end", seg_end)
                w["speaker"] = get_best_speaker(w_start, w_end, seg_speaker)

        num_speakers = len(unique_speakers) if unique_speakers else 1
        return segments, num_speakers
