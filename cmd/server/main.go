package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"whisper-service/internal/api"
	"whisper-service/internal/config"
	"whisper-service/internal/repository"
	"whisper-service/internal/supervisor"
	"whisper-service/internal/webhook"
)

func main() {
	log.Println("==================================================")
	log.Println("  Whisper Asynchronous Transcription Service API  ")
	log.Println("==================================================")

	// Load configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v\n", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize Database Pool
	log.Printf("Connecting to PostgreSQL at: %s\n", sanitizeDBURL(cfg.DatabaseURL))
	db, err := repository.NewDB(ctx, cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v\n", err)
	}
	defer db.Close()
	log.Println("PostgreSQL connection established successfully.")

	// Execute Schema Migration
	log.Println("Verifying database schema...")
	if err := db.RunMigrations(ctx, repository.InitialSchemaSQL); err != nil {
		log.Fatalf("Database schema migration failed: %v\n", err)
	}
	log.Println("Database schema is up-to-date.")

	// Initialize Repositories
	jobRepo := repository.NewJobRepository(db)

	// Start Background Webhook Dispatcher
	dispatcher := webhook.NewDispatcher(cfg, jobRepo)
	go dispatcher.Start(ctx)

	// Start Stale Job Watchdog
	watchdog := supervisor.NewWatchdog(cfg, jobRepo)
	go watchdog.Start(ctx)

	// Setup Echo HTTP Server
	e := api.SetupRouter(cfg, jobRepo, db)

	// Start HTTP Server in goroutine
	serverAddr := ":" + cfg.Port
	go func() {
		log.Printf("Whisper Service API running on %s\n", serverAddr)
		if err := e.Start(serverAddr); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("Echo server error: %v\n", err)
		}
	}()

	// Graceful shutdown handling
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt, syscall.SIGTERM)
	<-quit

	log.Println("Shutdown signal received, gracefully terminating services...")
	cancel() // Stop background workers

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	if err := e.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP server shutdown error: %v\n", err)
	}

	log.Println("Whisper Service API stopped cleanly.")
}

func sanitizeDBURL(rawURL string) string {
	// Simple sanitizer to avoid logging passwords
	if len(rawURL) > 30 {
		return rawURL[:12] + "..." + rawURL[len(rawURL)-12:]
	}
	return "postgres://..."
}
