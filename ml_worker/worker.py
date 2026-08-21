import os
import sys
import time
import signal
import threading
import logging
from typing import Optional
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
        self.worker_status = "idle"
        self.heartbeat_thread: Optional[threading.Thread] = None

    def initialize_models(self):
        logger.info("==================================================")
        logger.info(f"Initializing ML Worker: {self.config.WORKER_ID}")
        logger.info("==================================================")
        
        # 1. Connect to PostgreSQL
        self.db.connect()

        # 2. Load Whisper Model (once at startup)
        self.transcriber.load_model()

        # 3. Load Diarization Pipeline (once at startup if enabled)
        if self.config.ENABLE_DIARIZATION_DEFAULT:
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

    def run(self):
        self.initialize_models()
        self.is_running = True

        # Start heartbeat background thread
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

        logger.info(f"Worker {self.config.WORKER_ID} listening for transcription jobs on PostgreSQL queue...")

        while self.is_running:
            try:
                # 1. Atomic job dequeue
                job = self.db.dequeue_job(self.config.WORKER_ID)
                if not job:
                    time.sleep(self.config.POLL_INTERVAL)
                    continue

                job_id = str(job["id"])
                self.current_job_id = job_id
                self.worker_status = "busy"
                logger.info(f"==> Dequeued Job {job_id} (Audio: {job['audio_url']})")

                # 2. Process job with progress updates
                def update_progress(stage: str, progress: int):
                    self.db.update_job_progress(job_id, stage, progress, self.config.WORKER_ID)

                enable_diarize = bool(job.get("enable_diarization", False))
                lang = job.get("language")
                mode = job.get("transcription_mode", "fast")
                callback_url = job.get("callback_url")

                try:
                    result = self.pipeline.process(
                        job_id=job_id,
                        audio_url=job["audio_url"],
                        enable_diarization=enable_diarize,
                        language=lang,
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
                    logger.info(f"==> Successfully completed Job {job_id}")

                except torch.cuda.OutOfMemoryError as oom:
                    logger.error(f"GPU OOM while processing Job {job_id}: {oom}")
                    torch.cuda.empty_cache()
                    self.db.fail_job(
                        job_id=job_id,
                        error_message="Inference failed due to GPU Out of Memory (OOM)",
                        error_details={"error_type": "GPU_OOM", "details": str(oom)},
                        callback_url=callback_url
                    )
                except Exception as proc_err:
                    logger.error(f"Error processing Job {job_id}: {proc_err}", exc_info=True)
                    self.db.fail_job(
                        job_id=job_id,
                        error_message=str(proc_err),
                        error_details={"error_type": type(proc_err).__name__, "details": str(proc_err)},
                        callback_url=callback_url
                    )

            except Exception as loop_err:
                logger.error(f"Worker main loop unexpected error: {loop_err}", exc_info=True)
                time.sleep(2.0)
            finally:
                self.current_job_id = None
                self.worker_status = "idle"

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
