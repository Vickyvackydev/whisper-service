package models

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
)

// Job Statuses
const (
	StatusPending    = "pending"
	StatusQueued     = "queued"
	StatusProcessing = "processing"
	StatusCompleted  = "completed"
	StatusFailed     = "failed"
	StatusCancelled  = "cancelled"
)

// Job Stages
const (
	StageQueued        = "queued"
	StageDownloading   = "downloading"
	StagePreprocessing = "preprocessing"
	StageTranscribing  = "transcribing"
	StageAligning      = "aligning"
	StageDiarizing     = "diarizing"
	StageFinalizing    = "finalizing"
	StageCompleted     = "completed"
	StageFailed        = "failed"
)

// API Request Payloads

type TranscriptionRequest struct {
	AudioURL            string  `json:"audio_url"`
	EnableDiarization   bool    `json:"enable_diarization"`
	EnableSummarization bool    `json:"enable_summarization"`
	Language            *string `json:"language,omitempty"`
	TranscriptionMode   string  `json:"transcription_mode"`
	CallbackURL         *string `json:"callback_url,omitempty"`
	SummaryFormat       string  `json:"summary_format"`
	IdempotencyKey      *string `json:"idempotency_key,omitempty"`
}

func (r *TranscriptionRequest) UnmarshalJSON(data []byte) error {
	type Alias TranscriptionRequest
	aux := &struct {
		EnableDiarization  *bool `json:"enable_diarization"`
		DiarizationEnabled *bool `json:"diarization_enabled"`
		Diarize            *bool `json:"diarize"`
		*Alias
	}{
		Alias: (*Alias)(r),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	if aux.EnableDiarization != nil {
		r.EnableDiarization = *aux.EnableDiarization
	} else if aux.DiarizationEnabled != nil {
		r.EnableDiarization = *aux.DiarizationEnabled
	} else if aux.Diarize != nil {
		r.EnableDiarization = *aux.Diarize
	} else {
		// Default to TRUE so standard client requests without explicit flag get diarization automatically
		r.EnableDiarization = true
	}
	return nil
}

// API Response Payloads

type GenericAPIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Message string      `json:"message"`
	Error   *APIError   `json:"error,omitempty"`
}

type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

type SubmissionData struct {
	TranscriptionID string `json:"transcription_id"`
	Status          string `json:"status"`
}

type StatusData struct {
	Progress        int    `json:"progress"`
	ResultAvailable bool   `json:"result_available"`
	Stage           string `json:"stage"`
	Status          string `json:"status"`
	TranscriptionID string `json:"transcription_id"`
	ErrorMessage    string `json:"error_message,omitempty"`
}

type ResultData struct {
	TranscriptionID string             `json:"transcription_id"`
	Status          string             `json:"status"`
	Result          *TranscriptionData `json:"result"`
	ProcessingTime  float64            `json:"processing_time"`
}

type TranscriptionData struct {
	Segments              []Segment `json:"segments"`
	Language              string    `json:"language"`
	TranscribedText       string    `json:"transcribed_text"`
	SourceTranscribedText *string   `json:"source_transcribed_text"`
	TranscriptionMode     string    `json:"transcription_mode"`
	ModelUsed             string    `json:"model_used"`
	Duration              float64   `json:"duration"`
	DurationFormatted     string    `json:"duration_formatted"`
	NumSpeakers           int       `json:"num_speakers"`
	ProcessingTime        float64   `json:"processing_time"`
	ProcessingTimeFormatted string  `json:"processing_time_formatted"`
	WordCount             int       `json:"word_count"`
	TargetLanguage        *string   `json:"target_language"`
	TranslationMethod     *string   `json:"translation_method"`
}

type Segment struct {
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Text       string  `json:"text"`
	SourceText *string `json:"source_text"`
	Speaker    string  `json:"speaker"`
	Words      []Word  `json:"words"`
}

type Word struct {
	Word        string   `json:"word"`
	Start       float64  `json:"start"`
	End         float64  `json:"end"`
	Score       *float64 `json:"score"`
	Speaker     string   `json:"speaker"`
	SourceWord  *string  `json:"source_word"`
	MappingType *string  `json:"mapping_type"`
}

// Health & Ready Responses

type HealthData struct {
	Status    string    `json:"status"`
	Timestamp time.Time `json:"timestamp"`
	Version   string    `json:"version"`
	Database  string    `json:"database"`
}

type ReadyData struct {
	Ready              bool             `json:"ready"`
	DatabaseConnected  bool             `json:"database_connected"`
	ActiveWorkersCount int              `json:"active_workers_count"`
	Workers            []WorkerSummary  `json:"workers"`
	QueueMetrics       QueueMetricsInfo `json:"queue_metrics"`
}

type WorkerSummary struct {
	WorkerID          string    `json:"worker_id"`
	GPUAvailable      bool      `json:"gpu_available"`
	GPUName           *string   `json:"gpu_name"`
	GPUMemoryTotalMB  *int      `json:"gpu_memory_total_mb"`
	GPUMemoryUsedMB   *int      `json:"gpu_memory_used_mb"`
	ModelLoaded       *string   `json:"model_loaded"`
	DiarizationLoaded bool      `json:"diarization_loaded"`
	Status            string    `json:"status"`
	LastHeartbeatAt   time.Time `json:"last_heartbeat_at"`
}

type QueueMetricsInfo struct {
	QueuedJobsCount     int `json:"queued_jobs_count"`
	ProcessingJobsCount int `json:"processing_jobs_count"`
}

// Database Entities

type Job struct {
	ID                  uuid.UUID        `json:"id"`
	IdempotencyKey      *string          `json:"idempotency_key"`
	Status              string           `json:"status"`
	Stage               string           `json:"stage"`
	Progress            int              `json:"progress"`
	AudioURL            string           `json:"audio_url"`
	EnableDiarization   bool             `json:"enable_diarization"`
	EnableSummarization bool             `json:"enable_summarization"`
	Language            *string          `json:"language"`
	TranscriptionMode   string           `json:"transcription_mode"`
	SummaryFormat       string           `json:"summary_format"`
	CallbackURL         *string          `json:"callback_url"`
	RetryCount          int              `json:"retry_count"`
	MaxRetries          int              `json:"max_retries"`
	WorkerID            *string          `json:"worker_id"`
	WorkerHeartbeatAt   *time.Time       `json:"worker_heartbeat_at"`
	ErrorMessage        *string          `json:"error_message"`
	ErrorDetails        *json.RawMessage `json:"error_details"`
	Result              *json.RawMessage `json:"result"`
	AudioDuration       *float64         `json:"audio_duration"`
	ProcessingTime      *float64         `json:"processing_time"`
	StartedAt           *time.Time       `json:"started_at"`
	CompletedAt         *time.Time       `json:"completed_at"`
	CreatedAt           time.Time        `json:"created_at"`
	UpdatedAt           time.Time        `json:"updated_at"`
}
