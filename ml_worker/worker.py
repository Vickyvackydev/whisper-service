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
        self.concurrency = max(1, getattr(self.config, "WORKER_CONCURRENCY", 2))
        self._state_lock = threading.Lock()
        self.active_job_ids: Set[str] = set()
        self.last_active_time = time.time()
        self.heartbeat_thread: Optional[threading.Thread] = None

    @property
    def current_job_id(self) -> Optional[str]:
        with self._state_lock:
            return next(iter(self.active_job_ids), None)

    @property
    def worker_status(self) -> str:
        with self._state_lock:
            return "busy" if len(self.active_job_ids) > 0 else "idle"

    def initialize_models(self):
        logger.info("==================================================")
        logger.info(f"Initializing ML Worker: {self.config.WORKER_ID} (concurrency={self.concurrency})")
        logger.info("==================================================")
        
        # 1. Connect to PostgreSQL
        self.db.connect()

        # Set default CUDA device ordinal 0
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
                logger.info(f"Active CUDA device initialized to ordinal 0: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                logger.warning(f"Could not pin CUDA device 0: {e}")

        # 2. Load Whisper Model (once at startup, shared across concurrent worker slots)
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
                props = torch.cuda.get_device_properties(0)
                gpu_total_mb = int(props.total_memory / (1024 * 1024))
                gpu_used_mb = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
            except Exception as e:
                logger.debug(f"Failed to read GPU properties: {e}")

        return gpu_available, gpu_name, gpu_total_mb, gpu_used_mb

    def heartbeat_loop(self):
        logger.info("Heartbeat telemetry loop started.")
        while self.is_running:
            try:
                gpu_avail, gpu_name, gpu_tot, gpu_used = self.get_gpu_telemetry()
                self.db.send_worker_heartbeat(
                    worker_id=self.config.WORKER_ID,
                    gpu_available=gpu_avail,
                    gpu_name=gpu_name,
                    gpu_memory_total_mb=gpu_tot,
                    gpu_memory_used_mb=gpu_used,
                    model_loaded=self.transcriber.model_name,
                    diarization_loaded=self.diarizer.is_loaded,
                    status=self.worker_status,
                    current_job_id=self.current_job_id
                )
            except Exception as e:
                logger.debug(f"Heartbeat loop error: {e}")
            
            time.sleep(self.config.HEARTBEAT_INTERVAL)

    def _worker_slot_loop(self, slot_id: int):
        # When running 3+ concurrent slots, dedicate the highest slot as an Express Fast-Lane (prefer_newest=True)
        # so brand-new live uploads are processed immediately without waiting behind bulk backlog jobs.
        prefer_newest = (self.concurrency >= 3 and slot_id == self.concurrency)
        lane_label = "EXPRESS-NEWEST" if prefer_newest else "FIFO-QUEUE"
        logger.info(f"[Slot-{slot_id} ({lane_label})] Concurrent job slot started.")
        while self.is_running:
            job_id: Optional[str] = None
            try:
                # 1. Atomic job dequeue
                job = self.db.dequeue_job(self.config.WORKER_ID, prefer_newest=prefer_newest)
                if not job:
                    with self._state_lock:
                        active_count = len(self.active_job_ids)
                        idle_seconds = time.time() - self.last_active_time

                    if (
                        slot_id == 1
                        and active_count == 0
                        and self.config.IDLE_SHUTDOWN_MINUTES > 0
                        and idle_seconds >= (self.config.IDLE_SHUTDOWN_MINUTES * 60)
                    ):
                        logger.info(
                            f"Worker idle for {int(idle_seconds // 60)}m "
                            f"(>= {self.config.IDLE_SHUTDOWN_MINUTES}m threshold). Triggering RunPod auto-stop..."
                        )
                        self.trigger_runpod_auto_stop()
                        self.stop()
                        os._exit(0)

                    time.sleep(self.config.POLL_INTERVAL)
                    continue

                job_id = str(job["id"])
                with self._state_lock:
                    self.active_job_ids.add(job_id)
                    self.last_active_time = time.time()

                logger.info(f"[Slot-{slot_id}] ==> Dequeued Job {job_id} (Audio: {job['audio_url']})")

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
                    logger.info(f"[Slot-{slot_id}] ==> Successfully completed Job {job_id}")

                except torch.cuda.OutOfMemoryError as oom:
                    logger.error(f"[Slot-{slot_id}] GPU OOM while processing Job {job_id}: {oom}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.db.fail_job(
                        job_id=job_id,
                        error_message="Inference failed due to GPU Out of Memory (OOM)",
                        error_details={"error_type": "GPU_OOM", "details": str(oom)},
                        callback_url=callback_url
                    )
                except Exception as proc_err:
                    logger.error(f"[Slot-{slot_id}] Error processing Job {job_id}: {proc_err}", exc_info=True)
                    err_str = str(proc_err)
                    is_oom = "out of memory" in err_str.lower() or ("cuda" in err_str.lower() and "memory" in err_str.lower())
                    if is_oom:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    self.db.fail_job(
                        job_id=job_id,
                        error_message=str(proc_err),
                        error_details={"error_type": "GPU_OOM" if is_oom else type(proc_err).__name__, "details": str(proc_err)},
                        callback_url=callback_url
                    )

                    # Fatal CUDA state corruption recovery
                    is_fatal_cuda = any(pattern in err_str.lower() for pattern in [
                        "invalid device ordinal",
                        "cudaerrorinvaliddevice",
                        "illegal memory access",
                        "unspecified launch failure",
                        "cuda error: all cuda-capable devices are busy"
                    ])
                    if is_fatal_cuda:
                        logger.critical(f"FATAL CUDA DRIVER ERROR: {proc_err}. Exiting worker process for fresh restart...")
                        time.sleep(1.0)
                        os._exit(1)

            except Exception as loop_err:
                logger.error(f"[Slot-{slot_id}] Worker loop unexpected error: {loop_err}", exc_info=True)
                time.sleep(2.0)
            finally:
                if job_id is not None:
                    with self._state_lock:
                        self.active_job_ids.discard(job_id)
                        self.last_active_time = time.time()
                        remaining_active = len(self.active_job_ids)
                    try:
                        gc.collect()
                        if remaining_active == 0 and torch and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as cleanup_err:
                        logger.debug(f"Post-job GPU cleanup note: {cleanup_err}")

    def run(self):
        self.initialize_models()
        self.is_running = True

        # Start heartbeat background thread
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

        logger.info(
            f"Worker {self.config.WORKER_ID} listening for transcription jobs with {self.concurrency} concurrent slots "
            f"(Idle shutdown after {self.config.IDLE_SHUTDOWN_MINUTES}m)..."
        )

        slot_threads = []
        for slot_id in range(2, self.concurrency + 1):
            t = threading.Thread(target=self._worker_slot_loop, args=(slot_id,), daemon=True)
            t.start()
            slot_threads.append(t)

        # Run slot 1 on main thread
        self._worker_slot_loop(1)

    def trigger_runpod_auto_stop(self):
        pod_id = self.config.RUNPOD_POD_ID or "yf989jc63uke3g"
        api_key = self.config.RUNPOD_API_KEY
        
        logger.info(f"Initiating auto-stop for RunPod Pod ID: {pod_id}...")

        # Method 1: Native runpodctl CLI tool (built into every RunPod container)
        try:
            import subprocess
            cmd = ["runpodctl", "stop", "pod", pod_id]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                logger.info(f"Successfully stopped pod using runpodctl CLI: {res.stdout.strip()}")
                return
            else:
                logger.warning(f"runpodctl stop pod returned code {res.returncode}: {res.stderr.strip() or res.stdout.strip()}")
        except Exception as cli_err:
            logger.warning(f"runpodctl CLI execution failed ({cli_err}). Trying REST API v2...")

        # Method 2: RunPod REST API v2
        if api_key:
            try:
                import urllib.request
                import json

                url = f"https://api.runpod.io/v2/pods/{pod_id}/action"
                req_data = json.dumps({"action": "stop"}).encode("utf-8")
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }
                req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    logger.info(f"Successfully triggered RunPod REST API v2 stop for Pod ID: {pod_id} (HTTP {resp.status})")
                    return
            except Exception as api_err:
                logger.error(f"REST API v2 stop failed: {api_err}")

    def stop(self):
        logger.info("Stopping ML Worker...")
        self.is_running = False
        self.db.close()
        logger.info("ML Worker shutdown complete.")

if __name__ == "__main__":
    worker = MLWorker()

    def handle_signal(sig, frame):
        logger.info(f"Signal {sig} received, exiting...")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    worker.run()
