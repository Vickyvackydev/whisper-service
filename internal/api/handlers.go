package api

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"
	"whisper-service/internal/config"
	"whisper-service/internal/models"
	"whisper-service/internal/repository"
	"whisper-service/internal/supervisor"
)

type Handler struct {
	cfg          *config.Config
	jobRepo      *repository.JobRepository
	db           *repository.DB
	runpodClient *supervisor.RunPodClient
}

func NewHandler(cfg *config.Config, jobRepo *repository.JobRepository, db *repository.DB) *Handler {
	runpodClient := supervisor.NewRunPodClient(cfg.RunPodAPIKey, cfg.RunPodPodID)
	return &Handler{
		cfg:          cfg,
		jobRepo:      jobRepo,
		db:           db,
		runpodClient: runpodClient,
	}
}

// HealthCheck GET /api/v1/health
func (h *Handler) HealthCheck(c echo.Context) error {
	ctx := c.Request().Context()
	dbStatus := "connected"
	if err := h.db.Pool.Ping(ctx); err != nil {
		dbStatus = "disconnected"
		return c.JSON(http.StatusServiceUnavailable, models.GenericAPIResponse{
			Success: false,
			Message: "Service unhealthy: database unreachable",
			Data: models.HealthData{
				Status:    "unhealthy",
				Timestamp: time.Now().UTC(),
				Version:   "1.0.0",
				Database:  dbStatus,
			},
		})
	}

	return c.JSON(http.StatusOK, models.GenericAPIResponse{
		Success: true,
		Message: "Service is healthy",
		Data: models.HealthData{
			Status:    "healthy",
			Timestamp: time.Now().UTC(),
			Version:   "1.0.0",
			Database:  dbStatus,
		},
	})
}

// ReadyCheck GET /api/v1/ready
func (h *Handler) ReadyCheck(c echo.Context) error {
	ctx := c.Request().Context()

	// Check DB
	dbConnected := true
	if err := h.db.Pool.Ping(ctx); err != nil {
		dbConnected = false
	}

	// Check active workers (heartbeat within last 45s)
	workers, err := h.jobRepo.GetActiveWorkers(ctx, 45*time.Second)
	if err != nil {
		workers = []models.WorkerSummary{}
	}

	// Check queue metrics
	metrics, err := h.jobRepo.GetQueueMetrics(ctx)
	if err != nil {
		metrics = &models.QueueMetricsInfo{}
	}

	// Service is ready if DB is connected and at least one worker is alive (or in dev mode)
	isReady := dbConnected && (len(workers) > 0 || h.cfg.Environment == "development")

	statusCode := http.StatusOK
	if !isReady {
		statusCode = http.StatusServiceUnavailable
	}

	return c.JSON(statusCode, models.GenericAPIResponse{
		Success: isReady,
		Message: func() string {
			if isReady {
				return "Service is ready to process transcription jobs"
			}
			return "Service is not ready: no active GPU ML workers detected or database unavailable"
		}(),
		Data: models.ReadyData{
			Ready:              isReady,
			DatabaseConnected:  dbConnected,
			ActiveWorkersCount: len(workers),
			Workers:            workers,
			QueueMetrics:       *metrics,
		},
	})
}

// GetModes GET /api/v1/transcribe/modes
func (h *Handler) GetModes(c echo.Context) error {
	modes := []map[string]interface{}{
		{
			"mode":        "fast",
			"description": "Fastest inference, optimized for high throughput and quick turnaround (Whisper large-v3 float16/int8)",
			"recommended": true,
		},
		{
			"mode":        "accurate",
			"description": "Highest accuracy beam-search transcription with multi-pass diarization",
			"recommended": false,
		},
		{
			"mode":        "balanced",
			"description": "Balanced speed and accuracy with standard VAD segmentation",
			"recommended": false,
		},
	}

	return c.JSON(http.StatusOK, models.GenericAPIResponse{
		Success: true,
		Message: "Available transcription modes retrieved",
		Data:    modes,
	})
}

// GetLanguages GET /api/v1/transcribe/languages
func (h *Handler) GetLanguages(c echo.Context) error {
	languages := []map[string]string{
		{"code": "auto", "name": "Auto Detect"},
		{"code": "en", "name": "English"},
		{"code": "es", "name": "Spanish"},
		{"code": "fr", "name": "French"},
		{"code": "de", "name": "German"},
		{"code": "it", "name": "Italian"},
		{"code": "pt", "name": "Portuguese"},
		{"code": "nl", "name": "Dutch"},
		{"code": "ja", "name": "Japanese"},
		{"code": "zh", "name": "Chinese"},
		{"code": "ar", "name": "Arabic"},
		{"code": "ru", "name": "Russian"},
		{"code": "ko", "name": "Korean"},
		{"code": "hi", "name": "Hindi"},
		{"code": "tr", "name": "Turkish"},
		{"code": "pl", "name": "Polish"},
		{"code": "uk", "name": "Ukrainian"},
		{"code": "sv", "name": "Swedish"},
		{"code": "id", "name": "Indonesian"},
	}

	return c.JSON(http.StatusOK, models.GenericAPIResponse{
		Success: true,
		Message: "Supported languages retrieved",
		Data:    languages,
	})
}

// SubmitTranscription POST /api/v1/transcribe
func (h *Handler) SubmitTranscription(c echo.Context) error {
	var req models.TranscriptionRequest
	if err := c.Bind(&req); err != nil {
		return c.JSON(http.StatusBadRequest, models.GenericAPIResponse{
			Success: false,
			Message: "Invalid JSON request body",
			Error: &models.APIError{
				Code:    "BAD_REQUEST",
				Message: err.Error(),
			},
		})
	}

	// Extract idempotency key from header if not in payload
	idempotencyHeader := c.Request().Header.Get("X-Idempotency-Key")
	if idempotencyHeader != "" && req.IdempotencyKey == nil {
		req.IdempotencyKey = &idempotencyHeader
	}

	// Validate parameters
	if err := ValidateTranscriptionRequest(&req); err != nil {
		return c.JSON(http.StatusBadRequest, models.GenericAPIResponse{
			Success: false,
			Message: "Validation error",
			Error: &models.APIError{
				Code:    "VALIDATION_FAILED",
				Message: err.Error(),
			},
		})
	}

	// Insert into PostgreSQL queue
	job, isExisting, err := h.jobRepo.CreateJob(c.Request().Context(), &req, h.cfg.MaxRetries)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, models.GenericAPIResponse{
			Success: false,
			Message: "Failed to queue transcription job",
			Error: &models.APIError{
				Code:    "INTERNAL_ERROR",
				Message: "An error occurred while submitting the transcription job",
			},
		})
	}

	msg := "Transcription job submitted successfully"
	if isExisting {
		msg = "Transcription job already exists for the given idempotency key"
	}

	// Auto-Start GPU Pod if enabled and no active GPU workers are reporting heartbeats
	if h.cfg.AutoStartGPU && h.runpodClient != nil && h.runpodClient.IsConfigured() {
		workers, err := h.jobRepo.GetActiveWorkers(c.Request().Context(), 45*time.Second)
		if err != nil || len(workers) == 0 {
			log.Printf("[Auto-Start GPU] No active GPU worker detected. Asynchronously resuming RunPod GPU Pod %s...", h.cfg.RunPodPodID)
			go func() {
				bgCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
				defer cancel()
				if err := h.runpodClient.StartPod(bgCtx); err != nil {
					log.Printf("[Auto-Start GPU Error] Failed to resume RunPod GPU Pod: %v", err)
				}
			}()
		}
	}

	// Return immediate HTTP 202 Accepted
	return c.JSON(http.StatusAccepted, models.GenericAPIResponse{
		Success: true,
		Data: models.SubmissionData{
			TranscriptionID: job.ID.String(),
			Status:          job.Status,
		},
		Message: msg,
	})
}

// GetStatus GET /api/v1/transcribe/:id/status
func (h *Handler) GetStatus(c echo.Context) error {
	idStr := c.Param("id")
	jobID, err := uuid.Parse(idStr)
	if err != nil {
		return c.JSON(http.StatusBadRequest, models.GenericAPIResponse{
			Success: false,
			Message: "Invalid UUID format",
			Error: &models.APIError{
				Code:    "INVALID_ID",
				Message: "The transcription_id must be a valid UUID",
			},
		})
	}

	job, err := h.jobRepo.GetJobByID(c.Request().Context(), jobID)
	if err != nil {
		if errors.Is(err, repository.ErrJobNotFound) {
			return c.JSON(http.StatusNotFound, models.GenericAPIResponse{
				Success: false,
				Message: "Transcription job not found",
				Error: &models.APIError{
					Code:    "NOT_FOUND",
					Message: "No transcription found with the specified ID",
				},
			})
		}
		return c.JSON(http.StatusInternalServerError, models.GenericAPIResponse{
			Success: false,
			Message: "Failed to fetch status",
			Error: &models.APIError{
				Code:    "INTERNAL_ERROR",
				Message: "An internal error occurred",
			},
		})
	}

	resultAvailable := job.Status == models.StatusCompleted && job.Result != nil
	errMsg := ""
	if job.ErrorMessage != nil {
		errMsg = *job.ErrorMessage
	}

	// Direct JSON format matching existing client expectations
	return c.JSON(http.StatusOK, models.StatusData{
		Progress:        job.Progress,
		ResultAvailable: resultAvailable,
		Stage:           job.Stage,
		Status:          job.Status,
		TranscriptionID: job.ID.String(),
		ErrorMessage:    errMsg,
	})
}

// GetResult GET /api/v1/transcribe/:id/result
func (h *Handler) GetResult(c echo.Context) error {
	idStr := c.Param("id")
	jobID, err := uuid.Parse(idStr)
	if err != nil {
		return c.JSON(http.StatusBadRequest, models.GenericAPIResponse{
			Success: false,
			Message: "Invalid UUID format",
			Error: &models.APIError{
				Code:    "INVALID_ID",
				Message: "The transcription_id must be a valid UUID",
			},
		})
	}

	job, err := h.jobRepo.GetJobByID(c.Request().Context(), jobID)
	if err != nil {
		if errors.Is(err, repository.ErrJobNotFound) {
			return c.JSON(http.StatusNotFound, models.GenericAPIResponse{
				Success: false,
				Message: "Transcription job not found",
				Error: &models.APIError{
					Code:    "NOT_FOUND",
					Message: "No transcription found with the specified ID",
				},
			})
		}
		return c.JSON(http.StatusInternalServerError, models.GenericAPIResponse{
			Success: false,
			Message: "Failed to retrieve transcription result",
		})
	}

	if job.Status == models.StatusPending || job.Status == models.StatusQueued || job.Status == models.StatusProcessing {
		return c.JSON(http.StatusAccepted, models.GenericAPIResponse{
			Success: false,
			Message: "Transcription is still in progress",
			Data: models.SubmissionData{
				TranscriptionID: job.ID.String(),
				Status:          job.Status,
			},
		})
	}

	if job.Status == models.StatusFailed {
		errMsg := "Transcription failed"
		if job.ErrorMessage != nil {
			errMsg = *job.ErrorMessage
		}
		return c.JSON(http.StatusUnprocessableEntity, models.GenericAPIResponse{
			Success: false,
			Message: errMsg,
			Error: &models.APIError{
				Code:    "TRANSCRIPTION_FAILED",
				Message: errMsg,
			},
		})
	}

	if job.Status == models.StatusCancelled {
		return c.JSON(http.StatusGone, models.GenericAPIResponse{
			Success: false,
			Message: "Transcription job was cancelled",
		})
	}

	// Parse stored JSONB result
	var transcriptionData models.TranscriptionData
	if job.Result != nil {
		_ = json.Unmarshal(*job.Result, &transcriptionData)
	}

	procTime := 0.0
	if job.ProcessingTime != nil {
		procTime = *job.ProcessingTime
	}

	return c.JSON(http.StatusOK, models.GenericAPIResponse{
		Success: true,
		Message: "Transcription result retrieved successfully",
		Data: models.ResultData{
			TranscriptionID: job.ID.String(),
			Status:          job.Status,
			Result:          &transcriptionData,
			ProcessingTime:  procTime,
		},
	})
}

// DeleteJob DELETE /api/v1/transcribe/:id
func (h *Handler) DeleteJob(c echo.Context) error {
	idStr := c.Param("id")
	jobID, err := uuid.Parse(idStr)
	if err != nil {
		return c.JSON(http.StatusBadRequest, models.GenericAPIResponse{
			Success: false,
			Message: "Invalid UUID format",
		})
	}

	cancelled, err := h.jobRepo.CancelJob(c.Request().Context(), jobID)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, models.GenericAPIResponse{
			Success: false,
			Message: "Failed to delete/cancel job",
		})
	}

	if !cancelled {
		return c.JSON(http.StatusNotFound, models.GenericAPIResponse{
			Success: false,
			Message: "Job not found or already in terminal state",
		})
	}

	return c.JSON(http.StatusOK, models.GenericAPIResponse{
		Success: true,
		Message: "Transcription job cancelled successfully",
	})
}
