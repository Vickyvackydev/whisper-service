package config

import (
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

type Config struct {
	Port               string
	DatabaseURL        string
	APITokens          []string
	AuthEnabled        bool
	MaxAudioSizeBytes  int64
	JobTimeout         time.Duration
	WatchdogInterval   time.Duration
	StaleJobThreshold  time.Duration
	MaxRetries         int
	WebhookMaxAttempts int
	WebhookTimeout     time.Duration
	Environment        string
}

func LoadConfig() (*Config, error) {
	// Attempt to load .env if present
	_ = godotenv.Load()

	port := getEnv("PORT", "8080")
	dbURL := getEnv("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/whisper_service?sslmode=disable")
	
	tokensStr := getEnv("API_TOKENS", "")
	if tokensStr == "" {
		tokensStr = getEnv("API_KEY", "")
	}
	var tokens []string
	if tokensStr != "" {
		for _, t := range strings.Split(tokensStr, ",") {
			trimmed := strings.TrimSpace(t)
			if trimmed != "" {
				tokens = append(tokens, trimmed)
			}
		}
	}

	authEnabled := getEnvAsBool("AUTH_ENABLED", true)
	maxAudioBytes := getEnvAsInt64("MAX_AUDIO_SIZE_BYTES", 1024*1024*1024*2) // 2 GB
	maxRetries := getEnvAsInt("MAX_RETRIES", 3)
	webhookMaxAttempts := getEnvAsInt("WEBHOOK_MAX_ATTEMPTS", 5)

	return &Config{
		Port:               port,
		DatabaseURL:        dbURL,
		APITokens:          tokens,
		AuthEnabled:        authEnabled,
		MaxAudioSizeBytes:  maxAudioBytes,
		JobTimeout:         time.Duration(getEnvAsInt("JOB_TIMEOUT_SECONDS", 1800)) * time.Second,
		WatchdogInterval:   time.Duration(getEnvAsInt("WATCHDOG_INTERVAL_SECONDS", 10)) * time.Second,
		StaleJobThreshold:  time.Duration(getEnvAsInt("STALE_JOB_THRESHOLD_SECONDS", 45)) * time.Second,
		MaxRetries:         maxRetries,
		WebhookMaxAttempts: webhookMaxAttempts,
		WebhookTimeout:     time.Duration(getEnvAsInt("WEBHOOK_TIMEOUT_SECONDS", 15)) * time.Second,
		Environment:        getEnv("ENVIRONMENT", "development"),
	}, nil
}

func getEnv(key, defaultVal string) string {
	if val, exists := os.LookupEnv(key); exists && strings.TrimSpace(val) != "" {
		return val
	}
	return defaultVal
}

func getEnvAsInt(key string, defaultVal int) int {
	valStr := getEnv(key, "")
	if val, err := strconv.Atoi(valStr); err == nil {
		return val
	}
	return defaultVal
}

func getEnvAsInt64(key string, defaultVal int64) int64 {
	valStr := getEnv(key, "")
	if val, err := strconv.ParseInt(valStr, 10, 64); err == nil {
		return val
	}
	return defaultVal
}

func getEnvAsBool(key string, defaultVal bool) bool {
	valStr := getEnv(key, "")
	if val, err := strconv.ParseBool(valStr); err == nil {
		return val
	}
	return defaultVal
}
