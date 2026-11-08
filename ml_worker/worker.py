import os
import sys

# Prevent PyTorch CUDA memory allocator fragmentation
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import signal
import threading
import logging
from typing import Optional, Set
import torch

from ml_worker.config import WorkerConfig
from ml_worker.db import Database
from ml_worker.transcriber import Transcriber
from ml_worker.diarizer import SpeakerDiarizer
from ml_worker.pipeline import InferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ml_worker")

class MLWorker:
    def __init__(self):
        self.config = WorkerConfig
        self.db = Database(self.config.DATABASE_URL)
        self.transcriber = Transcriber()
        self.diarizer = SpeakerDiarizer()
        self.pipeline: Optional[InferencePipeline] = None
        
        self.is_running = False
        self.current_job_id: Optional[str] = None
        self.active_job_ids: Set[str] = set()
        self._state_lock = threading.Lock()
        self._job_counter = 0
        self.worker_status = "idle"
        self.heartbeat_thread: Optional[threading.Thread] = None
        self._last_active_timestamp = time.time()
        self._last_burst_wake_timestamp = 0.0
        self._dynamic_burst_pod_id: Optional[str] = None

    def _terminate_duplicate_workers(self):
        """
        Ensures only a single ml_worker.worker process holds GPU VRAM unless ALLOW_MULTI_WORKER=true.
        Prevents duplicate background workers from consuming 2x 8.7 GiB on a 20 GiB GPU and causing NeMo OOM.
        """
        if os.getenv("ALLOW_MULTI_WORKER", "false").lower() in ("true", "1", "yes"):
            return
        my_pid = os.getpid()
        if sys.platform != "win32" and os.path.isdir("/proc"):
            killed = []
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                if pid == my_pid:
                    continue
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        cmd = f.read().decode("utf-8", errors="ignore").replace("\x00", " ")
                    if "ml_worker.worker" in cmd or "ml_worker/worker.py" in cmd:
                        logger.warning(f"Found existing ml_worker.worker process (PID {pid}: '{cmd.strip()}'). Terminating it to free GPU VRAM...")
                        os.kill(pid, signal.SIGTERM)
                        killed.append(pid)
                except Exception:
                    continue
            if killed:
                time.sleep(2.5)
                for pid in killed:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass

    def initialize_models(self):
        logger.info("==================================================")
        logger.info(f"Initializing ML Worker: {self.config.WORKER_ID} (PID {os.getpid()})")
        logger.info("==================================================")

        # 0. Prevent duplicate worker processes from hogging GPU VRAM
        self._terminate_duplicate_workers()

        # 1. Connect to PostgreSQL
        self.db.connect()

        # Set default CUDA device ordinal 0
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
                logger.info(f"Active CUDA device initialized to ordinal 0: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                logger.warning(f"Could not pin CUDA device 0: {e}")

        # 2. Load Whisper Model (once at startup)
        self.transcriber.load_model()

        # 3. Load Diarization Pipeline (once at startup if enabled)
        if getattr(self.config, "ENABLE_DIARIZATION_DEFAULT", True):
            self.diarizer.load_model()

        # 4. Initialize Inference Pipeline
        self.pipeline = InferencePipeline(self.transcriber, self.diarizer)
        logger.info("ML Worker models and pipeline initialized successfully.")

    def get_gpu_telemetry(self):
        gpu_available = torch.cuda.is_available()
        gpu_name = None
        gpu_total_mb = None
        gpu_used_mb = None

        if gpu_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                    gpu_total_mb = int(total_bytes / (1024 * 1024))
                    gpu_used_mb = int((total_bytes - free_bytes) / (1024 * 1024))
                except Exception:
                    props = torch.cuda.get_device_properties(0)
                    gpu_total_mb = int(props.total_memory / (1024 * 1024))
                    gpu_used_mb = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
            except Exception as e:
                logger.debug(f"Failed to read GPU properties: {e}")

        return gpu_available, gpu_name, gpu_total_mb, gpu_used_mb

    def _is_within_court_hours(self) -> bool:
        """
        Returns True if BURST_KEEP_ALIVE_COURT_HOURS is enabled and current time in WAT (UTC+1)
        is Monday-Friday between COURT_START_HOUR_WAT (8 AM) and COURT_END_HOUR_WAT (5 PM).
        Prevents the secondary burst GPU from stopping mid-day and losing its physical host GPU to another renter.
        """
        if not getattr(self.config, "BURST_KEEP_ALIVE_COURT_HOURS", True):
            return False
        import datetime
        wat_tz = datetime.timezone(datetime.timedelta(hours=1))
        now_wat = datetime.datetime.now(wat_tz)
        if now_wat.weekday() >= 5:  # Saturday (5) or Sunday (6)
            return False
        start_hr = int(getattr(self.config, "COURT_START_HOUR_WAT", 8))
        end_hr = int(getattr(self.config, "COURT_END_HOUR_WAT", 17))
        return start_hr <= now_wat.hour < end_hr

    def _deploy_fallback_burst_pod(self) -> Optional[str]:
        """
        If the stopped burst pod's physical host GPU was rented by someone else overnight,
        automatically deploys a new On-Demand GPU Pod on any available host in RunPod.
        """
        api_key = getattr(self.config, "RUNPOD_API_KEY", "").strip()
        if not api_key or not getattr(self.config, "BURST_AUTO_PROVISION_FALLBACK", True):
            return None

        import requests
        url = f"https://api.runpod.io/graphql?api_key={api_key}"
        candidate_gpus = [
            "NVIDIA RTX 4000 Ada Generation",
            "NVIDIA RTX A4500",
            "NVIDIA RTX A5000",
            "NVIDIA L4",
            "NVIDIA GeForce RTX 3090",
        ]
        image_name = getattr(self.config, "RUNPOD_IMAGE_NAME", "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04")
        net_vol_id = getattr(self.config, "RUNPOD_NETWORK_VOLUME_ID", "").strip()

        startup_cmd = (
            "bash -c 'if [ ! -d /workspace/whisper-service ]; then "
            "git clone https://github.com/Vickyvackydev/whisper-service.git /workspace/whisper-service; "
            "fi && cd /workspace/whisper-service && git pull origin main && "
            "pip install -q -r requirements.txt && python3 -m ml_worker.worker'"
        )

        env_items = [
            {"key": "DATABASE_URL", "value": self.config.DATABASE_URL},
            {"key": "WORKER_ID", "value": "gpu-worker-runpod-burst-auto"},
            {"key": "WHISPER_MODEL", "value": self.config.WHISPER_MODEL},
            {"key": "WHISPER_DEVICE", "value": "cuda"},
            {"key": "WHISPER_COMPUTE_TYPE", "value": self.config.WHISPER_COMPUTE_TYPE},
            {"key": "DIARIZATION_BACKEND", "value": self.config.DIARIZATION_BACKEND},
            {"key": "HF_TOKEN", "value": self.config.HF_TOKEN},
            {"key": "IS_BURST_WORKER", "value": "true"},
            {"key": "IDLE_SHUTDOWN_MINUTES", "value": "15"},
            {"key": "RUNPOD_API_KEY", "value": api_key},
        ]

        mutation = """
        mutation($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                desiredStatus
            }
        }
        """

        for gpu_type in candidate_gpus:
            try:
                pod_input = {
                    "cloudType": "ALL",
                    "gpuCount": 1,
                    "volumeInGb": 40,
                    "containerDiskInGb": 30,
                    "minVcpuCount": 4,
                    "minMemoryInGb": 16,
                    "gpuTypeId": gpu_type,
                    "name": "whisper-burst-worker-auto",
                    "imageName": image_name,
                    "dockerArgs": startup_cmd,
                    "volumeMountPath": "/workspace",
                    "env": env_items,
                }
                if net_vol_id:
                    pod_input["networkVolumeId"] = net_vol_id

                resp = requests.post(url, json={"query": mutation, "variables": {"input": pod_input}}, timeout=15)
                if resp.status_code == 200:
                    payload = resp.json()
                    deployed = (payload.get("data") or {}).get("podFindAndDeployOnDemand")
                    if deployed and deployed.get("id"):
                        new_id = deployed["id"]
                        logger.info(
                            f"[Auto-Burst GPU] Successfully provisioned new fallback burst GPU pod {new_id} "
                            f"on free host ({gpu_type})!"
                        )
                        self._dynamic_burst_pod_id = new_id
                        return new_id
            except Exception as deploy_err:
                logger.debug(f"[Auto-Burst GPU] Could not provision fallback GPU {gpu_type}: {deploy_err}")
        return None

    def _resume_runpod_pod(self, pod_id: str):
        api_key = getattr(self.config, "RUNPOD_API_KEY", "")
        target_pod = self._dynamic_burst_pod_id or pod_id
        if not api_key or not target_pod:
            return
        try:
            import requests
            url = f"https://api.runpod.io/graphql?api_key={api_key}"
            query = f'mutation {{ podResume(input: {{ podId: "{target_pod}", gpuCount: 1 }}) {{ id desiredStatus }} }}'
            resp = requests.post(url, json={"query": query}, timeout=12)
            if resp.status_code == 200:
                body = {}
                try:
                    body = resp.json()
                except Exception:
                    pass
                errors = body.get("errors")
                resumed = (body.get("data") or {}).get("podResume")
                if resumed and not errors:
                    logger.info(f"[Auto-Burst GPU] Triggered RunPod wake-up (podResume) for secondary GPU pod {target_pod}: {resp.text}")
                    return
                logger.warning(
                    f"[Auto-Burst GPU] Secondary pod {target_pod} could not resume (host GPU may be occupied: {errors}). "
                    f"Attempting automatic new-host failover..."
                )
                self._deploy_fallback_burst_pod()
            else:
                logger.warning(f"[Auto-Burst GPU] RunPod wake-up HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"[Auto-Burst GPU] Failed to wake secondary pod {target_pod}: {e}")

    def _stop_runpod_pod(self, pod_id: str):
        api_key = getattr(self.config, "RUNPOD_API_KEY", "")
        if not api_key or not pod_id:
            return
        try:
            import requests
            url = f"https://api.runpod.io/graphql?api_key={api_key}"
            query = f'mutation {{ podStop(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus }} }}'
            resp = requests.post(url, json={"query": query}, timeout=12)
            logger.info(f"[Auto-Burst GPU] Triggered RunPod auto-stop (podStop) for idle burst pod {pod_id}: {resp.text}")
        except Exception as e:
            logger.warning(f"[Auto-Burst GPU] Failed to auto-stop burst pod {pod_id}: {e}")

    def heartbeat_loop(self):
        logger.info("Heartbeat telemetry loop started.")
        while self.is_running:
            try:
                gpu_avail, gpu_name, gpu_tot, gpu_used = self.get_gpu_telemetry()
                with self._state_lock:
                    active_list = list(self.active_job_ids)
                    cur_id = self.current_job_id or (active_list[0] if active_list else None)
                    status = "busy" if active_list else "idle"
                    if active_list:
                        self._last_active_timestamp = time.time()

                self.db.send_worker_heartbeat(
                    worker_id=self.config.WORKER_ID,
                    gpu_available=gpu_avail,
                    gpu_name=gpu_name,
                    gpu_memory_total_mb=gpu_tot,
                    gpu_memory_used_mb=gpu_used,
                    model_loaded=self.transcriber.model_name,
                    diarization_loaded=self.diarizer.is_loaded,
                    status=status,
                    current_job_id=cur_id,
                    active_job_ids=active_list
                )

                # Multi-GPU Auto-Burst Wake-Up & Court-Hours Protected Auto-Stop check
                burst_pod_id = getattr(self.config, "RUNPOD_BURST_POD_ID", "").strip()
                is_burst_worker = getattr(self.config, "IS_BURST_WORKER", False)
                idle_shutdown_mins = int(getattr(self.config, "IDLE_SHUTDOWN_MINUTES", 0))

                if burst_pod_id or (is_burst_worker and idle_shutdown_mins > 0):
                    queued_count, active_workers = self.db.get_queue_and_worker_counts()
                    now_ts = time.time()

                    # 1. Primary Worker: Wake secondary burst GPU pod when queue builds up (or at start of court rush)
                    threshold = int(getattr(self.config, "BURST_QUEUE_THRESHOLD", 3))
                    if (
                        burst_pod_id
                        and not is_burst_worker
                        and queued_count >= threshold
                        and active_workers < 2
                        and (now_ts - self._last_burst_wake_timestamp) >= 120.0
                    ):
                        self._last_burst_wake_timestamp = now_ts
                        logger.info(
                            f"[Auto-Burst GPU] Queue depth is {queued_count} (>= {threshold}) with {active_workers} active worker(s). "
                            f"Waking secondary GPU pod {burst_pod_id}..."
                        )
                        self._resume_runpod_pod(burst_pod_id)

                    # 2. Secondary Burst Worker: Stay locked during active court hours (8 AM - 5 PM WAT Mon-Fri)
                    #    so nobody can snatch the GPU mid-day; auto-stop after court hours when idle for IDLE_SHUTDOWN_MINUTES.
                    if is_burst_worker and idle_shutdown_mins > 0:
                        if queued_count > 0 or active_list or self._is_within_court_hours():
                            self._last_active_timestamp = now_ts
                        elif (now_ts - self._last_active_timestamp) >= (idle_shutdown_mins * 60):
                            own_pod_id = getattr(self.config, "RUNPOD_POD_ID", "").strip()
                            if own_pod_id:
                                logger.info(
                                    f"[Auto-Burst GPU] Outside court hours and secondary burst worker {self.config.WORKER_ID} "
                                    f"has been idle for {idle_shutdown_mins}m with 0 queued jobs. Stopping pod {own_pod_id}..."
                                )
                                self._stop_runpod_pod(own_pod_id)
                                self._last_active_timestamp = now_ts
            except Exception as e:
                logger.debug(f"Heartbeat loop error: {e}")
            
            time.sleep(self.config.HEARTBEAT_INTERVAL)

    def _worker_slot_loop(self, slot_id: int):
        lane_name = "Express Fast-Lane (Newest First)" if slot_id >= 2 else "FIFO Backlog Lane (Oldest First)"
        logger.info(f"Worker slot #{slot_id} [{lane_name}] active and polling PostgreSQL queue...")
        while self.is_running:
            job_id: Optional[str] = None
            try:
                # Slot #1 drains FIFO backlog (so 3h-10h recordings and older bulk jobs never starve)
                # Slot #2+ acts as dedicated Express Lane (so new single courtroom uploads start immediately)
                prefer_newest = (slot_id >= 2)
                job = self.db.dequeue_job(self.config.WORKER_ID, prefer_newest=prefer_newest)
                if not job:
                    time.sleep(self.config.POLL_INTERVAL)
                    continue

                job_id = str(job["id"])
                with self._state_lock:
                    self._job_counter += 1
                    self.active_job_ids.add(job_id)
                    self.current_job_id = job_id
                    self.worker_status = "busy"

                logger.info(f"==> [Slot #{slot_id}] Dequeued Job {job_id} (Audio: {job['audio_url']})")

                # 2. Process job with progress updates
                def update_progress(stage: str, progress: int):
                    self.db.update_job_progress(job_id, stage, progress, self.config.WORKER_ID)

                enable_diarize = bool(job.get("enable_diarization", False))
                enable_translate = bool(job.get("enable_translation", False))
                lang = job.get("language")
                target_lang = job.get("target_language")
                mode = job.get("transcription_mode", "fast")
                callback_url = job.get("callback_url")

                try:
                    result = self.pipeline.process(
                        job_id=job_id,
                        audio_url=job["audio_url"],
                        enable_diarization=enable_diarize,
                        enable_translation=enable_translate,
                        language=lang,
                        target_language=target_lang,
                        transcription_mode=mode,
                        progress_updater=update_progress
                    )

                    # 3. Mark completed
                    self.db.complete_job(
                        job_id=job_id,
                        result=result,
                        audio_duration=result["duration"],
                        processing_time=result["processing_time"],
                        callback_url=callback_url
                    )
                    logger.info(f"==> [Slot #{slot_id}] Successfully completed Job {job_id}")

                except torch.cuda.OutOfMemoryError as oom:
                    logger.error(f"[Slot #{slot_id}] GPU OOM while processing Job {job_id}: {oom}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if not self.db.requeue_job(job_id, f"GPU Out of Memory: {oom}"):
                        self.db.fail_job(
                            job_id=job_id,
                            error_message="Inference failed due to GPU Out of Memory (OOM)",
                            error_details={"error_type": "GPU_OOM", "details": str(oom)},
                            callback_url=callback_url
                        )
                except Exception as proc_err:
                    logger.error(f"[Slot #{slot_id}] Error processing Job {job_id}: {proc_err}", exc_info=True)
                    err_str = str(proc_err)
                    err_lower = err_str.lower()
                    is_oom = "out of memory" in err_lower or ("cuda" in err_lower and "memory" in err_lower)
                    is_cuda_transient = is_oom or any(
                        kw in err_lower for kw in ("invalid device ordinal", "cudaerror", "parallel_for", "cublas", "cudnn")
                    )
                    if is_cuda_transient:
                        gc.collect()
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.set_device(0)
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        if self.db.requeue_job(job_id, err_str):
                            if "invalid device ordinal" in err_lower or "cudaerror" in err_lower:
                                logger.warning("CUDA device context error detected; self-restarting worker process in-place to reset CUDA context...")
                                self.db.close()
                                os.execv(sys.executable, [sys.executable, "-m", "ml_worker.worker"])
                            continue

                    self.db.fail_job(
                        job_id=job_id,
                        error_message=err_str,
                        error_details={"error_type": "GPU_OOM" if is_oom else type(proc_err).__name__, "details": err_str},
                        callback_url=callback_url
                    )

            except Exception as loop_err:
                logger.error(f"[Slot #{slot_id}] Worker loop unexpected error: {loop_err}", exc_info=True)
                time.sleep(2.0)
            finally:
                if job_id is not None:
                    with self._state_lock:
                        self.active_job_ids.discard(job_id)
                        if self.active_job_ids:
                            self.current_job_id = next(iter(self.active_job_ids))
                            self.worker_status = "busy"
                        else:
                            self.current_job_id = None
                            self.worker_status = "idle"
                try:
                    gc.collect()
                except Exception as cleanup_err:
                    logger.debug(f"Post-job GC note: {cleanup_err}")

    def run(self):
        self.initialize_models()
        self.is_running = True

        # Start heartbeat background thread
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

        concurrency = max(1, int(getattr(self.config, "WORKER_CONCURRENCY", 2)))
        logger.info(
            f"Worker {self.config.WORKER_ID} listening 24/7 with {concurrency} pipelined worker slots "
            f"for transcription jobs on PostgreSQL queue (auto-shutdown disabled)..."
        )

        slot_threads = []
        for s_idx in range(2, concurrency + 1):
            t = threading.Thread(target=self._worker_slot_loop, args=(s_idx,), daemon=True)
            t.start()
            slot_threads.append(t)

        # Slot #1 runs on main thread
        self._worker_slot_loop(1)

    def stop(self):
        logger.info("Stopping ML Worker...")
        self.is_running = False
        self.db.close()
        logger.info("ML Worker shutdown complete.")

if __name__ == "__main__":
    worker = MLWorker()

    # Ignore SIGHUP so closing the RunPod web terminal never terminates the worker
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def handle_signal(sig, frame):
        logger.info(f"Signal {sig} received, exiting...")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    worker.run()
