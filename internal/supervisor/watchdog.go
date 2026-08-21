package supervisor

import (
	"context"
	"log"
	"time"

	"whisper-service/internal/config"
	"whisper-service/internal/repository"
)

type Watchdog struct {
	cfg     *config.Config
	jobRepo *repository.JobRepository
}

func NewWatchdog(cfg *config.Config, jobRepo *repository.JobRepository) *Watchdog {
	return &Watchdog{
		cfg:     cfg,
		jobRepo: jobRepo,
	}
}

// Start begins periodic stale job detection and automatic recovery
func (w *Watchdog) Start(ctx context.Context) {
	ticker := time.NewTicker(w.cfg.WatchdogInterval)
	defer ticker.Stop()

	log.Printf("[Watchdog] Stale job supervisor started (interval: %v, stale threshold: %v)\n",
		w.cfg.WatchdogInterval, w.cfg.StaleJobThreshold)

	for {
		select {
		case <-ctx.Done():
			log.Println("[Watchdog] Supervisor shutting down")
			return
		case <-ticker.C:
			w.checkStaleJobs(ctx)
		}
	}
}

func (w *Watchdog) checkStaleJobs(ctx context.Context) {
	recoveredCount, err := w.jobRepo.RecoverStaleJobs(ctx, w.cfg.StaleJobThreshold, w.cfg.MaxRetries)
	if err != nil {
		log.Printf("[Watchdog] Error checking stale jobs: %v\n", err)
		return
	}

	if recoveredCount > 0 {
		log.Printf("[Watchdog] Successfully recovered/updated %d stale or crashed worker jobs\n", recoveredCount)
	}
}
