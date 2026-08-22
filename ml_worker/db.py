import json
import logging
from typing import Optional, Dict, Any, Tuple
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from ml_worker.config import WorkerConfig

logger = logging.getLogger("ml_worker.db")

class Database:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool: Optional[pool.ThreadedConnectionPool] = None

    def connect(self):
        logger.info("Initializing PostgreSQL connection pool for ML Worker...")
        self.pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=self.db_url
        )
        logger.info("PostgreSQL worker connection pool established.")

    def get_conn(self):
        if self.pool is None:
            self.connect()
        return self.pool.getconn()

    def put_conn(self, conn):
        if self.pool and conn:
            self.pool.putconn(conn)

    def close(self):
        if self.pool:
            self.pool.closeall()

    def dequeue_job(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Atomically leases the next queued transcription job using FOR UPDATE SKIP LOCKED.
        """
        conn = self.get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    UPDATE transcription_jobs
                    SET status = 'processing',
                        stage = 'downloading',
                        worker_id = %s,
                        worker_heartbeat_at = NOW(),
                        started_at = NOW(),
                        updated_at = NOW()
                    WHERE id = (
                        SELECT id
                        FROM transcription_jobs
                        WHERE status = 'queued'
                        ORDER BY created_at ASC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING 
                        id, idempotency_key, status, stage, progress, audio_url,
                        enable_diarization, enable_translation, enable_summarization,
                        language, target_language, transcription_mode, summary_format,
                        callback_url, retry_count, max_retries;
                """
                cur.execute(query, (worker_id,))
                job = cur.fetchone()
                conn.commit()
                return dict(job) if job else None
        except Exception as e:
            conn.rollback()
            logger.error(f"Error while dequeuing job: {e}")
            return None
        finally:
            self.put_conn(conn)

    def update_job_progress(self, job_id: str, stage: str, progress: int, worker_id: str):
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                query = """
                    UPDATE transcription_jobs
                    SET stage = %s,
                        progress = %s,
                        worker_heartbeat_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s AND worker_id = %s;
                """
                cur.execute(query, (stage, progress, job_id, worker_id))
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Failed to update job progress for {job_id}: {e}")
        finally:
            self.put_conn(conn)

    def complete_job(
        self,
        job_id: str,
        result: Dict[str, Any],
        audio_duration: float,
        processing_time: float,
        callback_url: Optional[str]
    ):
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                result_json = json.dumps(result)
                query = """
                    UPDATE transcription_jobs
                    SET status = 'completed',
                        stage = 'completed',
                        progress = 100,
                        result = %s,
                        audio_duration = %s,
                        processing_time = %s,
                        completed_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s;
                """
                cur.execute(query, (result_json, audio_duration, processing_time, job_id))

                # If callback_url is set, queue webhook delivery
                if callback_url:
                    webhook_payload = json.dumps({
                        "success": True,
                        "data": {
                            "transcription_id": job_id,
                            "status": "completed",
                            "result": result,
                            "processing_time": processing_time
                        },
                        "message": "Transcription completed successfully"
                    })
                    webhook_query = """
                        INSERT INTO webhook_deliveries (job_id, callback_url, payload, status, next_retry_at)
                        VALUES (%s, %s, %s, 'pending', NOW());
                    """
                    cur.execute(webhook_query, (job_id, callback_url, webhook_payload))

                conn.commit()
                logger.info(f"Job {job_id} marked as completed in database.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to mark job {job_id} as completed: {e}")
            raise
        finally:
            self.put_conn(conn)

    def fail_job(
        self,
        job_id: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]],
        callback_url: Optional[str]
    ):
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                details_json = json.dumps(error_details) if error_details else None
                query = """
                    UPDATE transcription_jobs
                    SET status = 'failed',
                        stage = 'failed',
                        error_message = %s,
                        error_details = %s,
                        completed_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s;
                """
                cur.execute(query, (error_message, details_json, job_id))

                if callback_url:
                    webhook_payload = json.dumps({
                        "success": False,
                        "data": {
                            "transcription_id": job_id,
                            "status": "failed",
                            "error_message": error_message
                        },
                        "message": f"Transcription failed: {error_message}"
                    })
                    webhook_query = """
                        INSERT INTO webhook_deliveries (job_id, callback_url, payload, status, next_retry_at)
                        VALUES (%s, %s, %s, 'pending', NOW());
                    """
                    cur.execute(webhook_query, (job_id, callback_url, webhook_payload))

                conn.commit()
                logger.info(f"Job {job_id} marked as failed: {error_message}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to mark job {job_id} as failed: {e}")
        finally:
            self.put_conn(conn)

    def send_worker_heartbeat(
        self,
        worker_id: str,
        gpu_available: bool,
        gpu_name: Optional[str],
        gpu_memory_total_mb: Optional[int],
        gpu_memory_used_mb: Optional[int],
        model_loaded: Optional[str],
        diarization_loaded: bool,
        status: str,
        current_job_id: Optional[str]
    ):
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                query = """
                    INSERT INTO worker_heartbeats (
                        worker_id, gpu_available, gpu_name, gpu_memory_total_mb,
                        gpu_memory_used_mb, model_loaded, diarization_loaded,
                        status, current_job_id, last_heartbeat_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (worker_id) DO UPDATE SET
                        gpu_available = EXCLUDED.gpu_available,
                        gpu_name = EXCLUDED.gpu_name,
                        gpu_memory_total_mb = EXCLUDED.gpu_memory_total_mb,
                        gpu_memory_used_mb = EXCLUDED.gpu_memory_used_mb,
                        model_loaded = EXCLUDED.model_loaded,
                        diarization_loaded = EXCLUDED.diarization_loaded,
                        status = EXCLUDED.status,
                        current_job_id = EXCLUDED.current_job_id,
                        last_heartbeat_at = NOW();
                """
                cur.execute(query, (
                    worker_id, gpu_available, gpu_name, gpu_memory_total_mb,
                    gpu_memory_used_mb, model_loaded, diarization_loaded,
                    status, current_job_id
                ))
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug(f"Failed to record worker heartbeat: {e}")
        finally:
            self.put_conn(conn)
