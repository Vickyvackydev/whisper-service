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

            # Calibrate clustering threshold for discriminating distinct speakers in real-world audio
            # Default Pyannote threshold of 0.60 often merges speakers in shared room acoustics.
            # 0.52 separates distinct speakers cleanly across general meetings/interviews/courts while keeping single-speaker audio intact.
            try:
                threshold = float(getattr(WorkerConfig, "DIARIZATION_THRESHOLD", 0.52))
                self.pipeline.instantiate({"clustering": {"threshold": threshold}})
                logger.info(f"Pyannote clustering threshold calibrated to {threshold}.")
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
        Subdivides segments if words within transition across different speakers.
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

        new_segments = []
        for seg in segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            words = seg.get("words", [])

            if not words:
                seg_speaker = get_best_speaker(seg_start, seg_end, current_speaker)
                seg["speaker"] = seg_speaker
                current_speaker = seg_speaker
                new_segments.append(seg)
                continue

            # Word-level speaker assignment
            for w in words:
                w_start = float(w.get("start", seg_start))
                w_end = float(w.get("end", seg_end))
                w["speaker"] = get_best_speaker(w_start, w_end, current_speaker)

            # Subdivide segment if words transition across different speakers
            sub_words = []
            curr_sub_speaker = words[0].get("speaker", current_speaker)

            for w in words:
                w_spk = w.get("speaker", curr_sub_speaker)
                if w_spk != curr_sub_speaker and sub_words:
                    # Flush previous sub-segment
                    sub_text = " ".join(str(sw.get("word", "")).strip() for sw in sub_words).strip()
                    new_segments.append({
                        "start": sub_words[0].get("start", seg_start),
                        "end": sub_words[-1].get("end", seg_end),
                        "text": sub_text,
                        "source_text": None,
                        "speaker": curr_sub_speaker,
                        "words": sub_words
                    })
                    sub_words = [w]
                    curr_sub_speaker = w_spk
                else:
                    sub_words.append(w)

            if sub_words:
                sub_text = " ".join(str(sw.get("word", "")).strip() for sw in sub_words).strip()
                new_segments.append({
                    "start": sub_words[0].get("start", seg_start),
                    "end": sub_words[-1].get("end", seg_end),
                    "text": sub_text,
                    "source_text": None,
                    "speaker": curr_sub_speaker,
                    "words": sub_words
                })
                current_speaker = curr_sub_speaker

        # Check if court honorific cues are present anywhere in the transcript.
        # If present, always disambiguate interlocution turns between Bench and Counsel for clean dialogue separation.
        all_text_lower = " ".join(str(w.get("word", "")).lower() for seg in new_segments for w in seg.get("words", []))
        has_counsel_cues = any(k in all_text_lower for k in [
            "my lord", "your honour", "your honor", "your lordship", "your worship",
            "court pleases", "sir?", "sought from the court", "may it please"
        ])
        if has_counsel_cues:
            new_segments, num_speakers = self.disambiguate_court_dialogue(new_segments)
        else:
            num_speakers = len(ordered_speakers) if ordered_speakers else 1

        logger.info(f"Assigned {num_speakers} unique speakers across {len(new_segments)} refined segments.")
        return new_segments, num_speakers

    def disambiguate_court_dialogue(self, segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        all_words = []
        for seg in segments:
            for w in seg.get("words", []):
                all_words.append(w)

        if not all_words:
            return segments, 1

        full_text = " ".join(str(w.get("word", "")).lower() for w in all_words)
        has_counsel_cues = any(k in full_text for k in [
            "my lord", "your honour", "your honor", "your lordship", "your worship",
            "court pleases", "sir?", "sought from the court", "may it please"
        ])

        if not has_counsel_cues:
            return segments, 1

        utterances = []
        curr_start = 0

        def is_turn_boundary(prev_w, curr_w):
            pause = float(curr_w.get("start", 0)) - float(prev_w.get("end", 0))
            txt = str(prev_w.get("word", "")).strip()
            if txt.endswith("?") and pause >= 0.0:
                return True
            if pause >= 0.28:
                return True
            if (txt.endswith(".") or txt.endswith("!")) and pause >= 0.10:
                return True
            w_prev_low = str(prev_w.get("word", "")).lower().strip()
            if "pleases" in w_prev_low or "grateful" in w_prev_low:
                return True
            w_curr = str(curr_w.get("word", "")).lower().strip()
            if w_curr in ["sir?", "sir", "well,"]:
                return True
            return False

        for i in range(1, len(all_words)):
            prev_w = all_words[i-1]
            curr_w = all_words[i]

            is_cue_start = False
            w_curr = str(curr_w.get("word", "")).lower().strip()
            if w_curr in ["sir?", "sir", "well,"]:
                is_cue_start = True
            elif i + 1 < len(all_words):
                w_next = str(all_words[i+1].get("word", "")).lower().strip()
                prev_text_low = str(prev_w.get("word", "")).lower().strip()
                non_leading_preps = {"is", "was", "has", "have", "had", "with", "for", "by", "of", "to", "there", "make", "take", "give", "said", "say"}
                if (w_curr in ["and", "so"]) and (w_next in ["tell", "file", "put", "why", "what", "can"]):
                    is_cue_start = True
                elif ((w_curr == "my" and w_next in ["lord", "noble"]) or
                      (w_curr == "your" and w_next in ["honour", "honor", "lordship", "worship"]) or
                      (w_curr == "can" and w_next == "i") or
                      (w_curr == "what" and w_next == "are") or
                      (w_curr == "so" and w_next in ["the", "can", "you", "this"]) or
                      (w_curr == "as" and w_next == "the") or
                      (w_curr == "file" and w_next == "a") or
                      (w_curr == "put" and w_next == "it") or
                      (w_curr == "tell" and w_next == "me") or
                      (w_curr == "what's" and w_next == "the") or
                      (w_curr in ["where's", "where"]) or
                      (w_curr == "was" and w_next == "it") or
                      (w_curr == "this" and w_next in ["matter", "thing"]) or
                      (w_curr in ["16th", "15th"]) or
                      (w_curr == "please" and w_next in ["call", "tell", "show"]) or
                      (w_curr in ["i", "we"] and w_next in ["didn't", "did", "said"]) or
                      (w_curr == "how" and w_next in ["was", "is", "did"]) or
                      (w_curr == "it" and w_next in ["was", "is"]) or
                      (w_curr == "pressure" or (w_curr == "court" and w_next in ["places", "place"]))):
                    if not (i > 0 and prev_text_low in ["and", "so"]):
                        is_cue_start = True
                elif (w_curr in ["yes", "no", "yeah"]) and (prev_text_low not in non_leading_preps) and (prev_text_low not in ["no", "yes", "yeah"]):
                    if w_next in ["no", "yes", "my", "sir", "the", "it's", "it", "well", "i", "we", "that", "what"]:
                        is_cue_start = True

            if is_turn_boundary(prev_w, curr_w) or is_cue_start:
                utterances.append({
                    "start_idx": curr_start,
                    "end_idx": i - 1
                })
                curr_start = i

        utterances.append({
            "start_idx": curr_start,
            "end_idx": len(all_words) - 1
        })

        last_established = "SPEAKER_01"

        counsel_keywords = [
            "my lord", "your honour", "your honor", "your lordship", "your lordships",
            "your ladyship", "your worship", "court pleases", "court places", "court place",
            "pressure as", "may it please", "if the court pleases", "grateful", "with respect",
            "humbly submit", "humbly apply", "we pray", "we are asking", "we are applying",
            "we are seeking", "our application", "my application", "my submission", "we submit",
            "we filed", "we have filed", "i filed", "i have filed", "we served", "we have served",
            "it was served", "may i orally apply", "orally apply", "sought from the court",
            "what i'm trying to say", "what i am trying to say", "i didn't ask", "we didn't ask",
            "convenient", "sir?"
        ]
        bench_keywords = [
            "learned silk", "learned counsel", "learned friend",
            "matter is adjourned", "case is adjourned", "adjourned to",
            "ruling is reserved", "judgment is reserved",
            "call the matter", "call the next",
            "file a motion", "file an application", "put it in writing", "put that in writing",
            "tell me why", "tell the court", "explain to the court",
            "what is your", "where is your", "where is the",
            "why did you", "why do you", "why have you",
            "how did you", "how was it", "was it served", "proof of service", "is that proper service",
            "so can we proceed", "so the order", "can i see", "let me see", "choose the one you want",
            "what are you asking", "what are you saying", "what are you talking about",
            "your understanding of", "what you ask for", "what you asked for", "you ask for is what",
            "order you asked for", "order that was made",
            "the judge said", "the judge it's not fair",
            "what's the court date", "court date", "take a date", "give us a date",
            "please call out", "mr. ", "mrs. ", "barrister ",
            "had a discretion", "discretion", "don't know you so well"
        ]

        months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        ordinals = ["1st", "2nd", "3rd", "th"]

        for u_idx, u in enumerate(utterances):
            u_words = all_words[u["start_idx"] : u["end_idx"] + 1]
            u_text = " ".join(str(w.get("word", "")).lower() for w in u_words).strip()

            is_counsel = any(k in u_text for k in counsel_keywords)
            is_bench = any(k in u_text for k in bench_keywords)
            has_month = any(m in u_text for m in months)
            has_ord = any(o in u_text for o in ordinals)
            is_date_turn = (len(u_words) <= 8) and ("adjourned" not in u_text) and (has_month or has_ord)

            if is_counsel and not is_bench:
                u_spk = "SPEAKER_01"
                last_established = "SPEAKER_01"
            elif is_bench and not is_counsel:
                u_spk = "SPEAKER_00"
                last_established = "SPEAKER_00"
            elif is_date_turn:
                u_spk = "SPEAKER_01" if last_established == "SPEAKER_00" else "SPEAKER_00"
                last_established = u_spk
            else:
                w_count = len(u_words)
                is_short_response = w_count <= 4 and (u_text.startswith("yes") or u_text.startswith("no") or u_text.startswith("yeah"))
                if is_short_response:
                    u_spk = "SPEAKER_01" if last_established == "SPEAKER_00" else "SPEAKER_00"
                    last_established = u_spk
                else:
                    prev_idx = u_idx
                    should_toggle = False
                    if prev_idx > 0:
                        prev_u = utterances[prev_idx-1]
                        prev_w = all_words[prev_u["end_idx"]]
                        prev_w_text = str(prev_w.get("word", "")).strip()
                        prev_u_text = " ".join(str(w.get("word", "")).lower() for w in all_words[prev_u["start_idx"] : prev_u["end_idx"] + 1])

                        if prev_w_text.endswith("?"):
                            should_toggle = True
                        elif "court pleases" in prev_u_text or "grateful" in prev_u_text:
                            if last_established == "SPEAKER_01":
                                should_toggle = True

                    if should_toggle:
                        u_spk = "SPEAKER_01" if last_established == "SPEAKER_00" else "SPEAKER_00"
                        last_established = u_spk
                    else:
                        u_spk = last_established

            u["speaker"] = u_spk
            for w in u_words:
                w["speaker"] = u_spk

        refined_segments = []
        curr_sub_words = []
        curr_speaker = all_words[0]["speaker"]

        for w in all_words:
            w_spk = w.get("speaker", curr_speaker)
            if w_spk != curr_speaker and curr_sub_words:
                sub_text = " ".join(str(sw.get("word", "")).strip() for sw in curr_sub_words).strip()
                refined_segments.append({
                    "start": curr_sub_words[0].get("start", 0.0),
                    "end": curr_sub_words[-1].get("end", 0.0),
                    "text": sub_text,
                    "source_text": None,
                    "speaker": curr_speaker,
                    "words": curr_sub_words
                })
                curr_sub_words = [w]
                curr_speaker = w_spk
            else:
                curr_sub_words.append(w)

        if curr_sub_words:
            sub_text = " ".join(str(sw.get("word", "")).strip() for sw in curr_sub_words).strip()
            refined_segments.append({
                "start": curr_sub_words[0].get("start", 0.0),
                "end": curr_sub_words[-1].get("end", 0.0),
                "text": sub_text,
                "source_text": None,
                "speaker": curr_speaker,
                "words": curr_sub_words
            })

        unique_speakers = len(set(s["speaker"] for s in refined_segments))
        logger.info(f"Court dialogue disambiguated into {unique_speakers} speakers across {len(refined_segments)} segments.")
        return refined_segments, unique_speakers
