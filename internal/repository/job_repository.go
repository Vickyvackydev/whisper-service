package repository

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"whisper-service/internal/models"
)

var (
	ErrJobNotFound = errors.New("transcription job not found")
)

type WebhookTask struct {
	ID          uuid.UUID
	JobID       uuid.UUID
	CallbackURL string
	Payload     []byte
	Attempt     int
	MaxAttempts int
}

type JobRepository struct {
	db *DB
}

func NewJobRepository(db *DB) *JobRepository {
	return &JobRepository{db: db}
}

// CreateJob creates a new job or returns existing if idempotency_key matches
func (r *JobRepository) CreateJob(ctx context.Context, req *models.TranscriptionRequest, maxRetries int) (*models.Job, bool, error) {
	jobID := uuid.New()
	query := `
		INSERT INTO transcription_jobs (
			id, idempotency_key, status, stage, progress, audio_url,
			enable_diarization, enable_translation, enable_summarization, language, target_language,
			transcription_mode, summary_format, callback_url, max_retries
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
		)
		ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING
		RETURNING 
			id, idempotency_key, status, stage, progress, audio_url,
			enable_diarization, enable_translation, enable_summarization, language, target_language,
			transcription_mode, summary_format, callback_url,
			retry_count, max_retries, worker_id, worker_heartbeat_at,
			error_message, error_details, result, audio_duration,
			processing_time, started_at, completed_at, created_at, updated_at;
	`

	initialStatus := models.StatusQueued
	initialStage := models.StageQueued

	var job models.Job
	err := r.db.Pool.QueryRow(ctx, query,
		jobID, req.IdempotencyKey, initialStatus, initialStage, 0, req.AudioURL,
		req.EnableDiarization, req.EnableTranslation, req.EnableSummarization, req.Language, req.TargetLanguage,
		req.TranscriptionMode, req.SummaryFormat, req.CallbackURL, maxRetries,
	).Scan(
		&job.ID, &job.IdempotencyKey, &job.Status, &job.Stage, &job.Progress, &job.AudioURL,
		&job.EnableDiarization, &job.EnableTranslation, &job.EnableSummarization, &job.Language, &job.TargetLanguage,
		&job.TranscriptionMode, &job.SummaryFormat, &job.CallbackURL,
		&job.RetryCount, &job.MaxRetries, &job.WorkerID, &job.WorkerHeartbeatAt,
		&job.ErrorMessage, &job.ErrorDetails, &job.Result, &job.AudioDuration,
		&job.ProcessingTime, &job.StartedAt, &job.CompletedAt, &job.CreatedAt, &job.UpdatedAt,
	)

	if err == nil {
		return &job, false, nil
	}

	if errors.Is(err, pgx.ErrNoRows) && req.IdempotencyKey != nil {
		// Conflict on idempotency key, fetch existing job
		existing, err := r.GetJobByIdempotencyKey(ctx, *req.IdempotencyKey)
		if err != nil {
			return nil, false, fmt.Errorf("failed to fetch existing idempotent job: %w", err)
		}
		return existing, true, nil
	}

	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23505" && req.IdempotencyKey != nil {
		existing, err := r.GetJobByIdempotencyKey(ctx, *req.IdempotencyKey)
		if err != nil {
			return nil, false, fmt.Errorf("failed to fetch existing idempotent job: %w", err)
		}
		return existing, true, nil
	}

	return nil, false, fmt.Errorf("failed to insert transcription job: %w", err)
}

func (r *JobRepository) GetJobByID(ctx context.Context, id uuid.UUID) (*models.Job, error) {
	query := `
		SELECT 
			id, idempotency_key, status, stage, progress, audio_url,
			enable_diarization, enable_translation, enable_summarization, language, target_language,
			transcription_mode, summary_format, callback_url,
			retry_count, max_retries, worker_id, worker_heartbeat_at,
			error_message, error_details, result, audio_duration,
			processing_time, started_at, completed_at, created_at, updated_at
		FROM transcription_jobs
		WHERE id = $1;
	`
	var job models.Job
	err := r.db.Pool.QueryRow(ctx, query, id).Scan(
		&job.ID, &job.IdempotencyKey, &job.Status, &job.Stage, &job.Progress, &job.AudioURL,
		&job.EnableDiarization, &job.EnableTranslation, &job.EnableSummarization, &job.Language, &job.TargetLanguage,
		&job.TranscriptionMode, &job.SummaryFormat, &job.CallbackURL,
		&job.RetryCount, &job.MaxRetries, &job.WorkerID, &job.WorkerHeartbeatAt,
		&job.ErrorMessage, &job.ErrorDetails, &job.Result, &job.AudioDuration,
		&job.ProcessingTime, &job.StartedAt, &job.CompletedAt, &job.CreatedAt, &job.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrJobNotFound
		}
		return nil, fmt.Errorf("error querying job by id: %w", err)
	}
	return &job, nil
}

func (r *JobRepository) GetJobByIdempotencyKey(ctx context.Context, key string) (*models.Job, error) {
	query := `
		SELECT 
			id, idempotency_key, status, stage, progress, audio_url,
			enable_diarization, enable_translation, enable_summarization, language, target_language,
			transcription_mode, summary_format, callback_url,
			retry_count, max_retries, worker_id, worker_heartbeat_at,
			error_message, error_details, result, audio_duration,
			processing_time, started_at, completed_at, created_at, updated_at
		FROM transcription_jobs
		WHERE idempotency_key = $1;
	`
	var job models.Job
	err := r.db.Pool.QueryRow(ctx, query, key).Scan(
		&job.ID, &job.IdempotencyKey, &job.Status, &job.Stage, &job.Progress, &job.AudioURL,
		&job.EnableDiarization, &job.EnableTranslation, &job.EnableSummarization, &job.Language, &job.TargetLanguage,
		&job.TranscriptionMode, &job.SummaryFormat, &job.CallbackURL,
		&job.RetryCount, &job.MaxRetries, &job.WorkerID, &job.WorkerHeartbeatAt,
		&job.ErrorMessage, &job.ErrorDetails, &job.Result, &job.AudioDuration,
		&job.ProcessingTime, &job.StartedAt, &job.CompletedAt, &job.CreatedAt, &job.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrJobNotFound
		}
		return nil, fmt.Errorf("error querying job by idempotency key: %w", err)
	}
	return &job, nil
}

func (r *JobRepository) CancelJob(ctx context.Context, id uuid.UUID) (bool, error) {
	query := `
		UPDATE transcription_jobs
		SET status = $1, stage = $2, error_message = 'Job cancelled by user request', completed_at = NOW()
		WHERE id = $3 AND status IN ('pending', 'queued', 'processing');
	`
	res, err := r.db.Pool.Exec(ctx, query, models.StatusCancelled, models.StageFailed, id)
	if err != nil {
		return false, fmt.Errorf("failed to cancel job: %w", err)
	}
	return res.RowsAffected() > 0, nil
}

func (r *JobRepository) GetQueueMetrics(ctx context.Context) (*models.QueueMetricsInfo, error) {
	query := `
		SELECT 
			COUNT(*) FILTER (WHERE status IN ('pending', 'queued')) as queued_count,
			COUNT(*) FILTER (WHERE status = 'processing') as processing_count
		FROM transcription_jobs;
	`
	var metrics models.QueueMetricsInfo
	err := r.db.Pool.QueryRow(ctx, query).Scan(&metrics.QueuedJobsCount, &metrics.ProcessingJobsCount)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve queue metrics: %w", err)
	}
	return &metrics, nil
}

func (r *JobRepository) GetActiveWorkers(ctx context.Context, threshold time.Duration) ([]models.WorkerSummary, error) {
	query := `
		SELECT 
			worker_id, gpu_available, gpu_name, gpu_memory_total_mb,
			gpu_memory_used_mb, model_loaded, diarization_loaded, status, last_heartbeat_at
		FROM worker_heartbeats
		WHERE last_heartbeat_at >= NOW() - $1::interval
		ORDER BY last_heartbeat_at DESC;
	`
	intervalStr := fmt.Sprintf("%d seconds", int(threshold.Seconds()))
	rows, err := r.db.Pool.Query(ctx, query, intervalStr)
	if err != nil {
		return nil, fmt.Errorf("failed to query worker heartbeats: %w", err)
	}
	defer rows.Close()

	var workers []models.WorkerSummary
	for rows.Next() {
		var w models.WorkerSummary
		err := rows.Scan(
			&w.WorkerID, &w.GPUAvailable, &w.GPUName, &w.GPUMemoryTotalMB,
			&w.GPUMemoryUsedMB, &w.ModelLoaded, &w.DiarizationLoaded, &w.Status, &w.LastHeartbeatAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan worker row: %w", err)
		}
		workers = append(workers, w)
	}
	return workers, nil
}

func (r *JobRepository) RecoverStaleJobs(ctx context.Context, staleThreshold time.Duration, maxRetries int) (int64, error) {
	intervalStr := fmt.Sprintf("%d seconds", int(staleThreshold.Seconds()))
	
	// Reset stale jobs that have retries left
	queryRetry := `
		UPDATE transcription_jobs
		SET status = 'queued',
			stage = 'queued',
			retry_count = retry_count + 1,
			worker_id = NULL,
			worker_heartbeat_at = NULL,
			updated_at = NOW()
		WHERE status = 'processing'
		  AND (worker_heartbeat_at IS NULL OR worker_heartbeat_at < NOW() - $1::interval)
		  AND retry_count < max_retries;
	`
	resRetry, err := r.db.Pool.Exec(ctx, queryRetry, intervalStr)
	if err != nil {
		return 0, fmt.Errorf("failed to recover retryable stale jobs: %w", err)
	}

	// Mark stale jobs that exceeded max retries as failed
	queryFail := `
		UPDATE transcription_jobs
		SET status = 'failed',
			stage = 'failed',
			error_message = 'Job failed: worker heartbeat timeout / worker crash (exceeded max retries)',
			completed_at = NOW(),
			updated_at = NOW()
		WHERE status = 'processing'
		  AND (worker_heartbeat_at IS NULL OR worker_heartbeat_at < NOW() - $1::interval)
		  AND retry_count >= max_retries;
	`
	resFail, err := r.db.Pool.Exec(ctx, queryFail, intervalStr)
	if err != nil {
		return 0, fmt.Errorf("failed to mark exhausted stale jobs as failed: %w", err)
	}

	return resRetry.RowsAffected() + resFail.RowsAffected(), nil
}

// Webhook repository methods

func (r *JobRepository) EnqueueWebhookDelivery(ctx context.Context, jobID uuid.UUID, callbackURL string, payload []byte) error {
	query := `
		INSERT INTO webhook_deliveries (job_id, callback_url, payload, status, next_retry_at)
		VALUES ($1, $2, $3, 'pending', NOW());
	`
	_, err := r.db.Pool.Exec(ctx, query, jobID, callbackURL, payload)
	return err
}

func (r *JobRepository) GetPendingWebhooks(ctx context.Context, limit int) ([]WebhookTask, error) {
	query := `
		UPDATE webhook_deliveries
		SET status = 'delivering', updated_at = NOW()
		WHERE id IN (
			SELECT id
			FROM webhook_deliveries
			WHERE status IN ('pending', 'retrying')
			  AND (next_retry_at IS NULL OR next_retry_at <= NOW())
			ORDER BY created_at ASC
			LIMIT $1
			FOR UPDATE SKIP LOCKED
		)
		RETURNING id, job_id, callback_url, payload, attempt, max_attempts;
	`
	rows, err := r.db.Pool.Query(ctx, query, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tasks []WebhookTask
	for rows.Next() {
		var t WebhookTask
		if err := rows.Scan(&t.ID, &t.JobID, &t.CallbackURL, &t.Payload, &t.Attempt, &t.MaxAttempts); err != nil {
			return nil, err
		}
		tasks = append(tasks, t)
	}
	return tasks, nil
}

func (r *JobRepository) UpdateWebhookDelivery(ctx context.Context, id uuid.UUID, status string, statusCode int, responseBody string, errMsg string, nextRetry *time.Time) error {
	query := `
		UPDATE webhook_deliveries
		SET status = $1,
			status_code = $2,
			response_body = $3,
			error_message = $4,
			next_retry_at = $5,
			completed_at = CASE WHEN $1 = 'success' OR $1 = 'failed' THEN NOW() ELSE completed_at END,
			updated_at = NOW()
		WHERE id = $6;
	`
	_, err := r.db.Pool.Exec(ctx, query, status, statusCode, responseBody, errMsg, nextRetry, id)
	return err
}
