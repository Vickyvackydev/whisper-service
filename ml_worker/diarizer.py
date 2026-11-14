import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import torch
import numpy as np
from ml_worker.config import WorkerConfig

logger = logging.getLogger("ml_worker.diarizer")


def _patch_lightning_for_nemo():
    """
    Ensures NeMo 3.0+ imports cleanly even when pytorch-lightning >= 2.5 removes NeptuneLogger.
    """
    for mod_name in ("pytorch_lightning.loggers", "lightning.pytorch.loggers"):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, "NeptuneLogger"):
                class _DummyNeptuneLogger:
                    def __init__(self, *args, **kwargs):
                        pass
                setattr(mod, "NeptuneLogger", _DummyNeptuneLogger)
        except Exception:
            pass


class SpeakerDiarizer:
    def __init__(self):
        self.backend = getattr(WorkerConfig, "DIARIZATION_BACKEND", "nemo").lower()
        self.nemo_model = None
        self.speaker_emb_model = None
        self.pipeline = None
        self.is_loaded = False
        self.nemo_model_name = getattr(WorkerConfig, "NEMO_DIARIZATION_MODEL", "nvidia/diar_sortformer_4spk-v1")
        self.speaker_emb_model_name = getattr(
            WorkerConfig, "NEMO_SPEAKER_EMBEDDING_MODEL", "nvidia/speakerverification_en_titanet_large"
        )
        self.pyannote_model_name = getattr(WorkerConfig, "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
        self.hf_token = WorkerConfig.HF_TOKEN
        self.device = "cuda" if (torch.cuda.is_available() and WorkerConfig.WHISPER_DEVICE != "cpu") else "cpu"
        self.device_obj = torch.device(self.device)
        self.last_speaker_centroids: Dict[str, List[float]] = {}

    def _load_speaker_embedding_model(self):
        if self.speaker_emb_model is not None:
            return
        try:
            _patch_lightning_for_nemo()
            from nemo.collections.asr.models import EncDecSpeakerLabelModel

            logger.info(
                f"Loading NVIDIA NeMo TitaNet-Large Speaker Verification model: "
                f"{self.speaker_emb_model_name} on {self.device}..."
            )
            self.speaker_emb_model = EncDecSpeakerLabelModel.from_pretrained(self.speaker_emb_model_name)
            self.speaker_emb_model.to(self.device_obj)
            self.speaker_emb_model.eval()
            logger.info(f"NVIDIA NeMo TitaNet-Large 192-D voiceprint model loaded successfully on {self.device}.")
        except Exception as emb_err:
            logger.warning(
                f"Note loading NeMo TitaNet speaker embedding model '{self.speaker_emb_model_name}' ({emb_err}); "
                f"using cepstral voiceprint fallback."
            )
            self.speaker_emb_model = None

    def load_model(self):
        if self.backend == "nemo":
            try:
                _patch_lightning_for_nemo()
                logger.info(f"Loading NVIDIA NeMo Diarization model: {self.nemo_model_name} on {self.device}...")
                from nemo.collections.asr.models import SortformerEncLabelModel

                self.nemo_model = SortformerEncLabelModel.from_pretrained(self.nemo_model_name)
                self.nemo_model.to(self.device_obj)
                self.nemo_model.eval()
                self._load_speaker_embedding_model()
                self.is_loaded = True
                logger.info(f"NVIDIA NeMo Sortformer Diarization model loaded successfully on {self.device}.")
                return
            except Exception as nemo_err:
                logger.warning(
                    f"Failed to load NVIDIA NeMo model '{self.nemo_model_name}' ({nemo_err}). "
                    f"Falling back to Pyannote pipeline..."
                )
                self.backend = "pyannote"

        self._load_pyannote()
        self._load_speaker_embedding_model()

    def _load_pyannote(self):
        if not self.hf_token:
            logger.warning("HF_TOKEN is not configured. Pyannote speaker diarization will be unavailable.")
            return

        try:
            logger.info(f"Loading Diarization pipeline: {self.pyannote_model_name} on {self.device}...")
            from pyannote.audio import Pipeline

            self.pipeline = Pipeline.from_pretrained(
                self.pyannote_model_name,
                token=self.hf_token
            )

            if self.device == "cuda":
                self.pipeline.to(self.device_obj)
                logger.info("Pyannote Diarization pipeline loaded on CUDA GPU.")
            else:
                logger.info("Pyannote Diarization pipeline loaded on CPU.")

            # Calibrate Pyannote hyperparameters
            try:
                thresh  = getattr(WorkerConfig, "DIARIZATION_THRESHOLD", 0.63)
                min_off = getattr(WorkerConfig, "MIN_DURATION_OFF", 0.20)
                min_on  = getattr(WorkerConfig, "MIN_DURATION_ON", 0.08)

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
                        logger.debug(f"pipeline.instantiate() not available ({inst_err}), using direct attrs")

                if thresh is not None:
                    if hasattr(self.pipeline, "clustering") and hasattr(self.pipeline.clustering, "threshold"):
                        self.pipeline.clustering.threshold = float(thresh)
                    for attr in ("_clustering", "klustering"):
                        sub = getattr(self.pipeline, attr, None)
                        if sub is not None and hasattr(sub, "threshold"):
                            sub.threshold = float(thresh)

                if hasattr(self.pipeline, "segmentation"):
                    seg = self.pipeline.segmentation
                    if min_off is not None and hasattr(seg, "min_duration_off"):
                        seg.min_duration_off = float(min_off)
                    if min_on is not None and hasattr(seg, "min_duration_on"):
                        seg.min_duration_on = float(min_on)

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
        max_speakers: Optional[int] = None,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000
    ):
        import pandas as pd
        turns = self.diarize(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            audio_data=audio_data,
            sample_rate=sample_rate
        )
        if not turns:
            return pd.DataFrame(columns=["start", "end", "speaker"])
        return pd.DataFrame(turns)

    def diarize(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        """
        Runs diarization on audio and returns list of speaker intervals:
        [{ "start": 0.5, "end": 4.2, "speaker": "SPEAKER_00" }, ...]
        """
        if self.backend == "nemo" and self.nemo_model is not None:
            return self._diarize_nemo(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                audio_data=audio_data,
                sample_rate=sample_rate
            )
        return self._diarize_pyannote(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            audio_data=audio_data,
            sample_rate=sample_rate
        )

    def _parse_nemo_preds(self, preds: Any, offset_sec: float = 0.0) -> List[Dict[str, Any]]:
        if isinstance(preds, tuple):
            preds = preds[0]

        if isinstance(preds, str):
            raw_segments = preds.strip().splitlines()
        elif isinstance(preds, list):
            if len(preds) > 0 and isinstance(preds[0], list) and not isinstance(preds[0], str):
                raw_segments = preds[0]
            else:
                raw_segments = preds
        else:
            raw_segments = []

        turns = []
        for item in raw_segments:
            if not item:
                continue

            st, et, spk_label = None, None, None

            # Format A: String line (e.g. "0.905 3.720 speaker_0" or RTTM "SPEAKER file 1 0.905 2.815 <NA> <NA> speaker_0 <NA> <NA>")
            if isinstance(item, str):
                parts = item.strip().split()
                if not parts:
                    continue
                if parts[0] == "SPEAKER" and len(parts) >= 8:
                    try:
                        st = float(parts[3])
                        dur = float(parts[4])
                        et = st + dur
                        spk_label = parts[7]
                    except (ValueError, IndexError):
                        continue
                elif len(parts) >= 3:
                    try:
                        st = float(parts[0])
                        et = float(parts[1])
                        spk_label = parts[2]
                    except ValueError:
                        continue
                elif len(parts) == 2:
                    try:
                        st = float(parts[0])
                        et = float(parts[1])
                        spk_label = "speaker_0"
                    except ValueError:
                        continue

            # Format B: List or tuple: [start, end, speaker]
            elif isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    try:
                        st = float(item[0])
                        et = float(item[1])
                        spk_label = str(item[2])
                    except (ValueError, TypeError):
                        continue
                elif len(item) == 2:
                    try:
                        st = float(item[0])
                        et = float(item[1])
                        spk_label = "speaker_0"
                    except (ValueError, TypeError):
                        continue

            if st is None or et is None or spk_label is None:
                continue

            st = round(float(st) + offset_sec, 3)
            et = round(float(et) + offset_sec, 3)
            if et <= st:
                continue

            spk_str = str(spk_label).strip().lower()
            num = None
            if spk_str.startswith("speaker_"):
                try:
                    num = int(spk_str.replace("speaker_", "").strip())
                except ValueError:
                    pass
            elif spk_str.isdigit():
                num = int(spk_str)

            if num is not None:
                formatted_spk = f"SPEAKER_{num:02d}"
            else:
                formatted_spk = f"SPEAKER_{spk_str.upper()}"

            turns.append({
                "start": st,
                "end": et,
                "speaker": formatted_spk
            })
        return turns

    def _diarize_nemo_chunked(
        self,
        audio_path: Path,
        audio_data: np.ndarray,
        sample_rate: int,
        chunk_sec: float = 480.0,
        overlap_sec: float = 30.0
    ) -> List[Dict[str, Any]]:
        """
        Runs NVIDIA NeMo Sortformer in overlapping windows (default 8-min chunks with 30s overlap)
        and stitches speaker identities across chunk boundaries using overlap turn co-occurrence
        and acoustic centroid cosine similarity. Guarantees bounded VRAM and zero Pyannote fallback.
        """
        import gc
        import os
        import tempfile
        import soundfile as sf

        total_samples = len(audio_data)
        total_sec = total_samples / float(sample_rate)
        step_sec = max(60.0, chunk_sec - overlap_sec)

        logger.info(
            f"Running chunked NVIDIA NeMo Sortformer diarization on {audio_path.name} "
            f"(duration={total_sec:.1f}s, chunk={chunk_sec:.0f}s, overlap={overlap_sec:.0f}s)..."
        )

        global_turns: List[Dict[str, Any]] = []
        global_centroids: Dict[str, np.ndarray] = {}

        # Build chunk descriptors up front so we can skip silent courtroom recess windows
        # and run NeMo Sortformer in multi-chunk GPU batches (batch_size=4) on 3h-10h recordings
        chunk_specs: List[Tuple[int, float, float, int, int]] = []
        chunk_idx = 0
        start_sec = 0.0
        while start_sec < total_sec:
            end_sec = min(total_sec, start_sec + chunk_sec)
            if total_sec - end_sec < overlap_sec + 15.0:
                end_sec = total_sec
            start_samp = int(start_sec * sample_rate)
            end_samp = int(end_sec * sample_rate)
            chunk_specs.append((chunk_idx, start_sec, end_sec, start_samp, end_samp))
            if end_sec >= total_sec:
                break
            start_sec += step_sec
            chunk_idx += 1

        nemo_batch_size = 4
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                free_b, _ = torch.cuda.mem_get_info(0)
                free_gb = free_b / (1024 ** 3)
                if free_gb < 4.5:
                    nemo_batch_size = 1
                elif free_gb < 7.0:
                    nemo_batch_size = 2
            except Exception:
                nemo_batch_size = 2

        chunk_turns_by_idx: Dict[int, List[Dict[str, Any]]] = {}
        active_specs = []
        skipped_silent = 0
        for spec in chunk_specs:
            c_idx, s_sec, e_sec, s_samp, e_samp = spec
            c_audio = audio_data[s_samp:e_samp]
            if len(c_audio) == 0:
                chunk_turns_by_idx[c_idx] = []
                continue
            rms = float(np.sqrt(np.mean(c_audio ** 2)))
            if rms < 1.5e-4:
                chunk_turns_by_idx[c_idx] = []
                skipped_silent += 1
                continue
            active_specs.append(spec)

        if skipped_silent > 0:
            logger.info(f"Skipped {skipped_silent} silent recess chunk(s) during NeMo diarization.")

        for b_start in range(0, len(active_specs), nemo_batch_size):
            batch_slice = active_specs[b_start : b_start + nemo_batch_size]
            tmp_paths: List[Path] = []
            try:
                for (c_idx, s_sec, e_sec, s_samp, e_samp) in batch_slice:
                    fd, tmp_str = tempfile.mkstemp(suffix=f"_nemo_chunk{c_idx}.wav")
                    os.close(fd)
                    p = Path(tmp_str)
                    sf.write(str(p), audio_data[s_samp:e_samp], sample_rate)
                    tmp_paths.append(p)

                batch_preds = None
                if len(tmp_paths) > 1:
                    try:
                        with torch.inference_mode():
                            batch_preds = self.nemo_model.diarize(
                                audio=[str(p) for p in tmp_paths],
                                batch_size=len(tmp_paths)
                            )
                        if isinstance(batch_preds, tuple):
                            batch_preds = batch_preds[0]
                    except Exception as batch_err:
                        logger.debug(f"Multi-chunk NeMo batch fallback to single-chunk ({batch_err})")
                        batch_preds = None

                if isinstance(batch_preds, list) and len(batch_preds) == len(batch_slice):
                    for idx_in_b, (c_idx, s_sec, _, _, _) in enumerate(batch_slice):
                        chunk_turns_by_idx[c_idx] = self._parse_nemo_preds(batch_preds[idx_in_b], offset_sec=s_sec)
                else:
                    for idx_in_b, (c_idx, s_sec, _, _, _) in enumerate(batch_slice):
                        with torch.inference_mode():
                            single_pred = self.nemo_model.diarize(audio=str(tmp_paths[idx_in_b]), batch_size=1)
                        chunk_turns_by_idx[c_idx] = self._parse_nemo_preds(single_pred, offset_sec=s_sec)
            finally:
                for p in tmp_paths:
                    if p and p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass

        for (chunk_idx, start_sec, end_sec, start_samp, end_samp) in chunk_specs:
            chunk_turns = chunk_turns_by_idx.get(chunk_idx, [])

            if chunk_turns:
                # Extract centroids for speakers in this chunk
                chunk_centroids: Dict[str, np.ndarray] = {}
                local_speakers = sorted(set(t["speaker"] for t in chunk_turns))
                for spk in local_speakers:
                    spk_turns = [t for t in chunk_turns if t["speaker"] == spk and (t["end"] - t["start"]) >= 0.4]
                    if not spk_turns:
                        spk_turns = [t for t in chunk_turns if t["speaker"] == spk]
                    feats = []
                    for st in spk_turns[:8]:
                        s_i = max(0, int(st["start"] * sample_rate))
                        e_i = min(total_samples, int(st["end"] * sample_rate))
                        if e_i - s_i >= int(0.3 * sample_rate):
                            feat = self._extract_audio_feature(audio_data[s_i:e_i], sample_rate)
                            if feat is not None:
                                feats.append(feat)
                    if feats:
                        avg_f = np.mean(feats, axis=0)
                        nrm = np.linalg.norm(avg_f)
                        if nrm > 1e-6:
                            chunk_centroids[spk] = avg_f / nrm

                if chunk_idx == 0 or not global_turns:
                    global_turns.extend(chunk_turns)
                    global_centroids.update(chunk_centroids)
                else:
                    # Map local_speakers -> global_speakers using:
                    # 1. Temporal overlap in [start_sec, start_sec + overlap_sec]
                    # 2. Acoustic centroid similarity
                    overlap_end = start_sec + overlap_sec
                    prev_overlap_turns = [t for t in global_turns if t["end"] > start_sec and t["start"] < overlap_end]
                    curr_overlap_turns = [t for t in chunk_turns if t["end"] > start_sec and t["start"] < overlap_end]

                    spk_map: Dict[str, str] = {}
                    used_global = set()

                    # Sort local speakers by chunk speech duration descending so primary speakers claim global IDs first
                    local_dur_map = {
                        s: sum(max(0.0, t["end"] - t["start"]) for t in chunk_turns if t["speaker"] == s)
                        for s in local_speakers
                    }
                    ordered_local_spks = sorted(local_speakers, key=lambda s: local_dur_map.get(s, 0.0), reverse=True)

                    for lspk in ordered_local_spks:
                        best_gspk = None
                        best_score = -1.0
                        best_ac_sim = -1.0

                        all_global_spks = sorted(set(t["speaker"] for t in global_turns))
                        for gspk in all_global_spks:
                            ov_dur = 0.0
                            total_l_ov = 0.0
                            for lt in curr_overlap_turns:
                                if lt["speaker"] != lspk:
                                    continue
                                total_l_ov += (min(lt["end"], overlap_end) - max(lt["start"], start_sec))
                                for gt in prev_overlap_turns:
                                    if gt["speaker"] != gspk:
                                        continue
                                    o = min(lt["end"], gt["end"], overlap_end) - max(lt["start"], gt["start"], start_sec)
                                    if o > 0:
                                        ov_dur += o
                            ov_ratio = (ov_dur / total_l_ov) if total_l_ov > 0.2 else 0.0

                            ac_sim = 0.0
                            if lspk in chunk_centroids and gspk in global_centroids:
                                ac_sim = float(np.dot(chunk_centroids[lspk], global_centroids[gspk]))

                            # If gspk is already used in this chunk, only allow another local cluster to merge into it
                            # if the TitaNet voiceprint similarity is unmistakably the same person (>= 0.62)
                            if gspk in used_global and ac_sim < 0.62:
                                continue

                            combined = (ov_ratio * 0.55) + (ac_sim * 0.45) if total_l_ov > 0.2 else ac_sim
                            if combined > best_score:
                                best_score = combined
                                best_ac_sim = ac_sim
                                best_gspk = gspk

                        match_thresh = 0.42 if self.speaker_emb_model is not None else 0.45
                        if best_gspk is not None and (best_score >= match_thresh or best_ac_sim >= 0.56):
                            spk_map[lspk] = best_gspk
                            used_global.add(best_gspk)
                            if lspk in chunk_centroids and best_gspk in global_centroids:
                                blended = 0.75 * global_centroids[best_gspk] + 0.25 * chunk_centroids[lspk]
                                nrm = np.linalg.norm(blended)
                                if nrm > 1e-6:
                                    global_centroids[best_gspk] = blended / nrm
                        else:
                            existing_ids = set(t["speaker"] for t in global_turns) | set(spk_map.values())
                            new_id = f"SPEAKER_{len(existing_ids):02d}"
                            spk_map[lspk] = new_id
                            if lspk in chunk_centroids:
                                global_centroids[new_id] = chunk_centroids[lspk]

                    # Stitch at the midpoint of the overlap window to avoid boundary edge artifacts
                    stitch_time = start_sec + (overlap_sec * 0.5)
                    kept_global = []
                    for gt in global_turns:
                        if gt["end"] <= stitch_time:
                            kept_global.append(gt)
                        elif gt["start"] < stitch_time:
                            gt_copy = dict(gt)
                            gt_copy["end"] = round(stitch_time, 3)
                            if gt_copy["end"] > gt_copy["start"]:
                                kept_global.append(gt_copy)
                    global_turns = kept_global

                    for lt in chunk_turns:
                        mapped_spk = spk_map.get(lt["speaker"], lt["speaker"])
                        if lt["end"] <= stitch_time:
                            continue
                        st_clamped = max(lt["start"], round(stitch_time, 3))
                        if lt["end"] > st_clamped:
                            global_turns.append({
                                "start": st_clamped,
                                "end": lt["end"],
                                "speaker": mapped_spk
                            })

        return global_turns

    def _diarize_nemo(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        """
        Runs NVIDIA NeMo Sortformer end-to-end speaker diarization.
        Sortformer outputs (start, end, speaker) chronologically sorted.
        Strictly uses NeMo Sortformer (with automatic chunked fallback on long audio or tight VRAM)
        and never degrades to Pyannote when configured for NeMo.
        """
        import gc
        import soundfile as sf

        logger.info(f"Running NVIDIA NeMo speaker diarization on {audio_path.name}...")
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass

        if audio_data is not None and len(audio_data) > 0:
            data = audio_data
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            data = np.ascontiguousarray(data, dtype=np.float32)
        else:
            data, sample_rate = sf.read(str(audio_path), dtype="float32")
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            data = np.ascontiguousarray(data, dtype=np.float32)
        duration_sec = len(data) / float(sample_rate) if sample_rate > 0 else 0.0

        free_vram_gb = 99.0
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                free_vram_gb = free_bytes / (1024 ** 3)
                logger.info(f"GPU VRAM before NeMo Sortformer: {free_vram_gb:.2f} GiB free / {total_bytes / (1024 ** 3):.2f} GiB total (audio={duration_sec:.1f}s)")
            except Exception:
                pass

        turns: List[Dict[str, Any]] = []
        # Proactively use chunked Sortformer if audio > 12 mins or free VRAM < 3.2 GiB
        use_chunked = (duration_sec > 720.0) or (free_vram_gb < 3.2)
        chunk_sec = 300.0 if free_vram_gb < 3.2 else 480.0

        if not use_chunked:
            try:
                with torch.inference_mode():
                    preds = self.nemo_model.diarize(audio=str(audio_path), batch_size=1)
                turns = self._parse_nemo_preds(preds, offset_sec=0.0)
            except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
                err_msg = str(e)
                is_oom = "out of memory" in err_msg.lower() or ("cuda" in err_msg.lower() and "memory" in err_msg.lower())
                if is_oom:
                    logger.warning(
                        f"NVIDIA NeMo full-file Sortformer hit CUDA OOM on {audio_path.name} ({duration_sec:.1f}s). "
                        f"Clearing GPU cache and retrying with chunked NVIDIA NeMo Sortformer (240s windows)..."
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    use_chunked = True
                    chunk_sec = 240.0
                else:
                    logger.error(f"NVIDIA NeMo full-file diarization error ({e}), retrying with chunked NeMo Sortformer...", exc_info=True)
                    use_chunked = True
                    chunk_sec = 240.0

        if use_chunked:
            try:
                turns = self._diarize_nemo_chunked(
                    audio_path=audio_path,
                    audio_data=data,
                    sample_rate=sample_rate,
                    chunk_sec=chunk_sec,
                    overlap_sec=30.0
                )
            except Exception as chunk_err:
                logger.error(f"Chunked NVIDIA NeMo Sortformer failed ({chunk_err}), retrying with 120s micro-windows...", exc_info=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                turns = self._diarize_nemo_chunked(
                    audio_path=audio_path,
                    audio_data=data,
                    sample_rate=sample_rate,
                    chunk_sec=120.0,
                    overlap_sec=20.0
                )

        # Post-diarization consolidation to prevent pitch/pause fractures
        try:
            turns = self._consolidate_fragmented_speakers(turns, audio_data=data, sample_rate=sample_rate)
        except Exception as consol_e:
            logger.debug(f"NeMo post-consolidation note ({consol_e}), using raw turns")
        finally:
            if audio_data is None:
                del data

        unique_detected = len(set(t["speaker"] for t in turns))
        logger.info(f"NVIDIA NeMo diarization successful: detected {len(turns)} turns across {unique_detected} distinct speakers.")
        return turns

    def _diarize_pyannote(
        self,
        audio_path: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        audio_data: Optional[np.ndarray] = None,
        sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        if not self.is_loaded or self.pipeline is None:
            logger.warning("Diarization pipeline not loaded.")
            return []

        try:
            logger.info(f"Running speaker diarization on {audio_path.name}...")
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.set_device(0)
                except Exception:
                    pass
            import soundfile as sf
            if audio_data is not None and len(audio_data) > 0:
                data = audio_data
            else:
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
        finally:
            try:
                del data
                if "waveform_cuda" in locals():
                    del waveform_cuda
                if "waveform_cpu" in locals():
                    del waveform_cpu
            except Exception:
                pass
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_audio_feature(self, waveform_1d: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Extracts an L2-normalized speaker voiceprint vector for a speech segment:
        1. Primary: NVIDIA NeMo TitaNet-Large 192-D neural speaker verification embedding (invariant to volume, pitch shift, and background music/noise).
        2. Secondary: Pyannote neural speaker embedding (if Pyannote backend is loaded).
        3. Fallback: Cepstral Mean-Normalized (CMN) MFCC + harmonic voiceprint (channel/volume invariant).
        """
        if waveform_1d is None or len(waveform_1d) < int(sample_rate * 0.28):
            return None

        # Focus on the highest-energy 6.0s window if segment is very long, and peak-normalize amplitude
        max_samples = int(sample_rate * 6.0)
        wav = np.asarray(waveform_1d, dtype=np.float32)
        if len(wav) > max_samples:
            mid = len(wav) // 2
            half = max_samples // 2
            wav = wav[max(0, mid - half) : min(len(wav), mid + half)]

        peak = float(np.max(np.abs(wav)))
        if peak < 1e-4:
            return None
        wav_norm = wav / peak

        # 1. Primary: NVIDIA NeMo TitaNet-Large 192-D Neural Speaker Verification Embedding
        if self.speaker_emb_model is not None:
            try:
                # Pad short clips (< 0.6s) slightly by tiling so TitaNet's pooling layer has sufficient frames
                min_titanet_samples = int(sample_rate * 0.65)
                wav_in = wav_norm
                if len(wav_in) < min_titanet_samples:
                    reps = (min_titanet_samples // len(wav_in)) + 1
                    wav_in = np.tile(wav_in, reps)[:min_titanet_samples]

                t_in = torch.from_numpy(np.ascontiguousarray(wav_in, dtype=np.float32)).unsqueeze(0).to(self.device_obj)
                t_len = torch.tensor([len(wav_in)], dtype=torch.long, device=self.device_obj)
                with torch.inference_mode():
                    out = self.speaker_emb_model.forward(input_signal=t_in, input_signal_length=t_len)
                emb_tensor = out[1] if (isinstance(out, (tuple, list)) and len(out) >= 2) else out
                emb_np = emb_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
                norm = float(np.linalg.norm(emb_np))
                if norm > 1e-6:
                    return (emb_np / norm).astype(np.float32)
            except Exception as titanet_err:
                logger.debug(f"TitaNet embedding extraction note ({titanet_err}), falling back")

        # 2. Secondary: Pyannote embedding if pipeline loaded and has _embedding
        if self.pipeline is not None and hasattr(self.pipeline, "_embedding"):
            try:
                emb_model = getattr(self.pipeline, "_embedding")
                if callable(emb_model):
                    t_in = torch.from_numpy(wav_norm).unsqueeze(0).unsqueeze(0).to(self.device_obj)
                    with torch.no_grad():
                        out = emb_model(t_in)
                    out_np = out.squeeze().detach().cpu().numpy().astype(np.float32)
                    norm = float(np.linalg.norm(out_np))
                    if norm > 1e-6:
                        return (out_np / norm).astype(np.float32)
            except Exception as emb_e:
                logger.debug(f"Direct Pyannote embedding extraction note ({emb_e}), using cepstral voiceprint")

        # 3. Fallback: Cepstral Mean-Normalized (CMN) Mel-Cepstral Voiceprint (36-D)
        # Subtracting across-time mean log-spectrum removes static background beat/channel coloration
        win_len = int(sample_rate * 0.025)
        hop_len = int(sample_rate * 0.010)
        if len(wav_norm) <= win_len:
            return None
        try:
            n_fft = 512
            window = np.hanning(win_len)
            num_frames = (len(wav_norm) - win_len) // hop_len
            if num_frames <= 0:
                return None

            indices = np.linspace(0, num_frames - 1, min(num_frames, 180), dtype=int)
            frames = np.stack([wav_norm[idx * hop_len : idx * hop_len + win_len] * window for idx in indices])
            frame_energy = np.mean(frames ** 2, axis=1)
            # Keep voiced frames (top 70% energy) to exclude pauses/drums between words
            if len(frames) >= 6:
                e_thresh = np.percentile(frame_energy, 30)
                voiced = frames[frame_energy >= e_thresh]
                if len(voiced) >= 3:
                    frames = voiced

            mags = np.abs(np.fft.rfft(frames, n=n_fft))
            n_bands = 32
            bin_freqs = np.linspace(0, sample_rate / 2, mags.shape[1])
            band_edges = np.geomspace(120, min(sample_rate / 2, 6800), n_bands + 2)
            filterbank = np.zeros((n_bands, mags.shape[1]), dtype=np.float32)
            for b in range(n_bands):
                left, center, right = band_edges[b], band_edges[b + 1], band_edges[b + 2]
                filterbank[b] = np.maximum(0, np.minimum(
                    (bin_freqs - left) / max(1e-6, center - left),
                    (right - bin_freqs) / max(1e-6, right - center)
                ))

            band_energies = np.dot(mags, filterbank.T)
            log_energies = np.log(np.maximum(1e-6, band_energies))
            # DCT-II to decorrelate vocal tract formants (MFCCs c_1..c_19) from overall loudness (c_0)
            k_idx = np.arange(1, 20)[:, None]
            n_idx = np.arange(n_bands)[None, :]
            dct_basis = np.cos(np.pi * k_idx * (2 * n_idx + 1) / (2.0 * n_bands)).astype(np.float32)
            mfcc = np.dot(log_energies, dct_basis.T)
            mfcc_mean = mfcc.mean(axis=0)
            mfcc_std = mfcc.std(axis=0)
            feat = np.concatenate([mfcc_mean, mfcc_std * 0.5]).astype(np.float32)
            norm = float(np.linalg.norm(feat))
            if norm > 1e-6:
                return (feat / norm).astype(np.float32)
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
        1. Neural Voiceprint Matching (NVIDIA NeMo TitaNet-Large 192-D cosine similarity & top-turn matching)
        2. Simultaneous Cross-Talk vs Sequential Turn Dynamics
        3. Minor / Micro-Fragment Absorption (prevents pitch/beat/pause splits like 'Check.' or mid-monologue shifts)
        4. Micro-pause turn stitching (<= 0.55s when no other speaker intervenes)
        5. Consistent Chronological Speaker Normalization (SPEAKER_00, SPEAKER_01...)
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

        # ── Extract Neural Voiceprint Centroids & Per-Turn Embeddings ────────
        speaker_centroids: Dict[str, np.ndarray] = {}
        speaker_turn_embs: Dict[str, List[np.ndarray]] = {}
        if audio_data is not None and len(audio_data) > 0:
            for spk in speaker_durations:
                # Prioritize longest turns (>= 0.45s) for cleanest voiceprint extraction
                spk_turns = sorted(
                    [t for t in sorted_turns if t["speaker"] == spk and (t["end"] - t["start"]) >= 0.45],
                    key=lambda x: (x["end"] - x["start"]),
                    reverse=True,
                )
                if not spk_turns:
                    spk_turns = sorted(
                        [t for t in sorted_turns if t["speaker"] == spk],
                        key=lambda x: (x["end"] - x["start"]),
                        reverse=True,
                    )

                feats = []
                for st in spk_turns[:10]:
                    start_s = max(0, int(st["start"] * sample_rate))
                    end_s = min(len(audio_data), int(st["end"] * sample_rate))
                    if end_s - start_s >= int(0.28 * sample_rate):
                        feat = self._extract_audio_feature(audio_data[start_s:end_s], sample_rate)
                        if feat is not None:
                            feats.append(feat)
                if feats:
                    speaker_turn_embs[spk] = feats
                    avg_feat = np.mean(feats, axis=0)
                    norm = float(np.linalg.norm(avg_feat))
                    if norm > 1e-6:
                        speaker_centroids[spk] = (avg_feat / norm).astype(np.float32)

        def _simultaneous_overlap_ratio(spk_a: str, spk_b: str) -> float:
            turns_a = [t for t in sorted_turns if t["speaker"] == spk_a]
            turns_b = [t for t in sorted_turns if t["speaker"] == spk_b]
            dur_b = max(0.01, speaker_durations.get(spk_b, 0.01))
            ov = 0.0
            for tb in turns_b:
                for ta in turns_a:
                    if ta["end"] <= tb["start"]:
                        continue
                    if ta["start"] >= tb["end"]:
                        break
                    o = min(ta["end"], tb["end"]) - max(ta["start"], tb["start"])
                    if o > 0.10:
                        ov += o
            return ov / dur_b

        def _pair_voiceprint_similarity(spk_a: str, spk_b: str) -> Tuple[float, float]:
            c_sim = -1.0
            if spk_a in speaker_centroids and spk_b in speaker_centroids:
                c_sim = float(np.dot(speaker_centroids[spk_a], speaker_centroids[spk_b]))
            embs_a = speaker_turn_embs.get(spk_a, [])
            embs_b = speaker_turn_embs.get(spk_b, [])
            if not embs_a or not embs_b:
                return c_sim, c_sim
            sims = [float(np.dot(ea, eb)) for ea in embs_a for eb in embs_b]
            sims.sort(reverse=True)
            top_k = sims[: min(3, len(sims))]
            top_sim = float(np.mean(top_k))
            return c_sim, top_sim

        # ── Build Voiceprint & Fragment Merge Mapping ────────────────────────
        to_merge: Dict[str, str] = {}
        ranked_speakers = sorted(speaker_durations.keys(), key=lambda s: speaker_durations[s], reverse=True)
        uses_neural_emb = (self.speaker_emb_model is not None) or (
            self.pipeline is not None and hasattr(self.pipeline, "_embedding")
        )

        def get_root_target(s: str) -> str:
            curr = s
            visited = set()
            while curr in to_merge and curr not in visited:
                visited.add(curr)
                curr = to_merge[curr]
            return curr

        for idx_min in range(1, len(ranked_speakers)):
            spk_min = ranked_speakers[idx_min]
            if spk_min in to_merge:
                continue
            dur_min = speaker_durations.get(spk_min, 0.0)
            frac_min = dur_min / max(0.1, total_speech)
            min_turns = [t for t in sorted_turns if t["speaker"] == spk_min]
            median_turn_dur = float(np.median([t["end"] - t["start"] for t in min_turns])) if min_turns else 0.0

            best_target = None
            best_c_sim = -1.0
            best_top_sim = -1.0

            for idx_maj in range(idx_min):
                cand_maj = get_root_target(ranked_speakers[idx_maj])
                if cand_maj == spk_min:
                    continue
                ov_ratio = _simultaneous_overlap_ratio(cand_maj, spk_min)
                # If two speakers talk over each other simultaneously for > 22% of the minor speaker's time, keep separate
                if ov_ratio > 0.22:
                    continue
                c_sim, top_sim = _pair_voiceprint_similarity(cand_maj, spk_min)
                if c_sim > best_c_sim:
                    best_c_sim = c_sim
                    best_top_sim = top_sim
                    best_target = cand_maj

            if best_target is None:
                # If spk_min had no valid audio feature (e.g. < 0.3s micro-clicks) and total duration < 2.0s, absorb into dominant
                if spk_min not in speaker_centroids and dur_min < 2.0:
                    to_merge[spk_min] = get_root_target(dominant)
                continue

            should_merge = False
            if uses_neural_emb:
                # Calibrated on live RunPod courtroom telemetry:
                # - Distinct speakers in the same courtroom have centroid_sim ~ 0.45 - 0.68 (e.g. 0.568)
                # - Identical speaker split across chunks/turns has centroid_sim >= 0.88 (e.g. 0.930)
                if best_c_sim >= 0.88 and best_top_sim >= 0.82:
                    should_merge = True
                # Ultra-short micro-glitch (< 1.0s total across entire recording)
                elif dur_min < 1.0 and frac_min < 0.015 and best_c_sim >= 0.72:
                    should_merge = True
            else:
                # Cepstral fallback thresholds
                if best_c_sim >= 0.92 and best_top_sim >= 0.88:
                    should_merge = True
                elif dur_min < 1.0 and frac_min < 0.015 and best_c_sim >= 0.82:
                    should_merge = True

            if should_merge:
                logger.info(
                    f"[VOICEPRINT MERGE] Consolidating {spk_min} ({dur_min:.1f}s) -> {best_target} "
                    f"(centroid_sim={best_c_sim:.3f}, top_turn_sim={best_top_sim:.3f})"
                )
                to_merge[spk_min] = best_target

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

        # Preserve 192-D NeMo TitaNet speaker voiceprint centroids for recurring speaker recognition
        remapped_centroids: Dict[str, List[float]] = {}
        for orig_spk, new_spk in remap.items():
            if orig_spk in speaker_centroids:
                vec = speaker_centroids[orig_spk]
                remapped_centroids[new_spk] = [round(float(x), 5) for x in vec.tolist()]
        self.last_speaker_centroids = remapped_centroids

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

        # Re-align 192-D TitaNet centroids with chronological speaker IDs (SPEAKER_00, SPEAKER_01...)
        if getattr(self, "last_speaker_centroids", None):
            updated_centroids: Dict[str, List[float]] = {}
            for old_spk, final_spk in chrono_map.items():
                if old_spk in self.last_speaker_centroids:
                    updated_centroids[final_spk] = self.last_speaker_centroids[old_spk]
            self.last_speaker_centroids = updated_centroids

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

        # Attach 192-D NeMo TitaNet voiceprint centroid to the first segment of each speaker
        centroids = getattr(self, "last_speaker_centroids", {}) or {}
        emitted_emb_speakers = set()
        for seg in refined_segments:
            spk = seg.get("speaker")
            if spk and spk in centroids and spk not in emitted_emb_speakers:
                seg["speaker_embedding"] = centroids[spk]
                emitted_emb_speakers.add(spk)

        unique_speakers = len(set(s["speaker"] for s in refined_segments))
        logger.info(f"Assigned {unique_speakers} unique speakers across {len(refined_segments)} refined segments.")
        return refined_segments, unique_speakers

