package webhook

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"time"

	"whisper-service/internal/config"
	"whisper-service/internal/repository"
)

type Dispatcher struct {
	cfg     *config.Config
	jobRepo *repository.JobRepository
	client  *http.Client
}

func NewDispatcher(cfg *config.Config, jobRepo *repository.JobRepository) *Dispatcher {
	return &Dispatcher{
		cfg:     cfg,
		jobRepo: jobRepo,
		client: &http.Client{
			Timeout: cfg.WebhookTimeout,
		},
	}
}

// Start begins the background webhook delivery loop
func (d *Dispatcher) Start(ctx context.Context) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	log.Println("[Webhook] Dispatcher background service started")

	for {
		select {
		case <-ctx.Done():
			log.Println("[Webhook] Dispatcher shutting down")
			return
		case <-ticker.C:
			d.processPendingWebhooks(ctx)
		}
	}
}

func (d *Dispatcher) processPendingWebhooks(ctx context.Context) {
	tasks, err := d.jobRepo.GetPendingWebhooks(ctx, 10)
	if err != nil {
		log.Printf("[Webhook] Error fetching pending webhooks: %v\n", err)
		return
	}

	for _, task := range tasks {
		go d.deliverWebhook(ctx, task)
	}
}

func (d *Dispatcher) deliverWebhook(ctx context.Context, task repository.WebhookTask) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, task.CallbackURL, bytes.NewReader(task.Payload))
	if err != nil {
		d.handleFailure(ctx, task, 0, "", fmt.Sprintf("Failed to construct HTTP request: %v", err))
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "WhisperService-Webhook-Dispatcher/1.0")

	startTime := time.Now()
	resp, err := d.client.Do(req)
	duration := time.Since(startTime)

	if err != nil {
		d.handleFailure(ctx, task, 0, "", fmt.Sprintf("HTTP delivery error after %v: %v", duration, err))
		return
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 2048)) // read up to 2KB response
	respBody := string(bodyBytes)

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		// Successful delivery
		log.Printf("[Webhook] Successfully delivered webhook for Job %s to %s (status: %d)\n",
			task.JobID, task.CallbackURL, resp.StatusCode)
		_ = d.jobRepo.UpdateWebhookDelivery(ctx, task.ID, "success", resp.StatusCode, respBody, "", nil)
	} else {
		// Non-2xx response status code
		d.handleFailure(ctx, task, resp.StatusCode, respBody, fmt.Sprintf("Received non-2xx status code: %d", resp.StatusCode))
	}
}

func (d *Dispatcher) handleFailure(ctx context.Context, task repository.WebhookTask, statusCode int, respBody string, errMsg string) {
	log.Printf("[Webhook] Delivery failed for Job %s (Attempt %d/%d): %s\n",
		task.JobID, task.Attempt, task.MaxAttempts, errMsg)

	if task.Attempt < task.MaxAttempts {
		// Calculate exponential backoff: 2s, 4s, 8s, 16s, 32s
		backoffSeconds := math.Pow(2, float64(task.Attempt))
		nextRetry := time.Now().Add(time.Duration(backoffSeconds) * time.Second)
		_ = d.jobRepo.UpdateWebhookDelivery(ctx, task.ID, "retrying", statusCode, respBody, errMsg, &nextRetry)
	} else {
		// Max attempts exhausted
		log.Printf("[Webhook] Max attempts reached for Job %s. Marking webhook delivery as failed.\n", task.JobID)
		_ = d.jobRepo.UpdateWebhookDelivery(ctx, task.ID, "failed", statusCode, respBody, errMsg, nil)
	}
}
