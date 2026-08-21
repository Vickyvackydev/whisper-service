-- schema.sql
-- Whisper Asynchronous Transcription Service Database Schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table: transcription_jobs
CREATE TABLE IF NOT EXISTS transcription_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key VARCHAR(255) UNIQUE,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    stage VARCHAR(32) NOT NULL DEFAULT 'queued',
    progress INT NOT NULL DEFAULT 0,
    audio_url TEXT NOT NULL,
    enable_diarization BOOLEAN NOT NULL DEFAULT false,
    enable_summarization BOOLEAN NOT NULL DEFAULT false,
    language VARCHAR(16),
    transcription_mode VARCHAR(32) NOT NULL DEFAULT 'fast',
    summary_format VARCHAR(32) NOT NULL DEFAULT 'plain_text',
    callback_url TEXT,
    retry_count INT NOT NULL DEFAULT 0,
    max_retries INT NOT NULL DEFAULT 3,
    worker_id VARCHAR(128),
    worker_heartbeat_at TIMESTAMPTZ,
    error_message TEXT,
    error_details JSONB,
    result JSONB,
    audio_duration DOUBLE PRECISION,
    processing_time DOUBLE PRECISION,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indices for performance and queue efficiency
CREATE INDEX IF NOT EXISTS idx_jobs_queue_status_created 
    ON transcription_jobs (status, created_at) 
    WHERE status IN ('pending', 'queued');

CREATE INDEX IF NOT EXISTS idx_jobs_heartbeat_check 
    ON transcription_jobs (status, worker_heartbeat_at) 
    WHERE status = 'processing';

CREATE INDEX IF NOT EXISTS idx_jobs_status 
    ON transcription_jobs (status);

CREATE INDEX IF NOT EXISTS idx_jobs_created_at 
    ON transcription_jobs (created_at DESC);

-- Table: webhook_deliveries
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES transcription_jobs(id) ON DELETE CASCADE,
    callback_url TEXT NOT NULL,
    payload JSONB NOT NULL,
    attempt INT NOT NULL DEFAULT 1,
    max_attempts INT NOT NULL DEFAULT 5,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    status_code INT,
    response_body TEXT,
    error_message TEXT,
    next_retry_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_webhooks_pending 
    ON webhook_deliveries (status, next_retry_at) 
    WHERE status IN ('pending', 'retrying');

-- Table: worker_heartbeats (for /api/v1/ready monitoring)
CREATE TABLE IF NOT EXISTS worker_heartbeats (
    worker_id VARCHAR(128) PRIMARY KEY,
    gpu_available BOOLEAN NOT NULL DEFAULT false,
    gpu_name VARCHAR(128),
    gpu_memory_total_mb INT,
    gpu_memory_used_mb INT,
    model_loaded VARCHAR(64),
    diarization_loaded BOOLEAN NOT NULL DEFAULT false,
    status VARCHAR(32) NOT NULL DEFAULT 'idle',
    current_job_id UUID,
    last_heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Trigger to automatically update updated_at
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_timestamp_jobs ON transcription_jobs;
CREATE TRIGGER set_timestamp_jobs
BEFORE UPDATE ON transcription_jobs
FOR EACH ROW
EXECUTE FUNCTION trigger_set_timestamp();

DROP TRIGGER IF EXISTS set_timestamp_webhooks ON webhook_deliveries;
CREATE TRIGGER set_timestamp_webhooks
BEFORE UPDATE ON webhook_deliveries
FOR EACH ROW
EXECUTE FUNCTION trigger_set_timestamp();
