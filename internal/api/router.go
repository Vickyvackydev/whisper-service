package api

import (
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"whisper-service/internal/config"
	"whisper-service/internal/repository"
)

func SetupRouter(cfg *config.Config, jobRepo *repository.JobRepository, db *repository.DB) *echo.Echo {
	e := echo.New()
	e.HideBanner = true

	// Standard middleware
	e.Use(middleware.Recover())
	e.Use(middleware.RequestID())
	e.Use(middleware.LoggerWithConfig(middleware.LoggerConfig{
		Format: `{"time":"${time_rfc3339}","id":"${id}","remote_ip":"${remote_ip}",` +
			`"method":"${method}","uri":"${uri}","status":${status},"latency_ms":${latency_ms}}` + "\n",
	}))
	e.Use(CORS())

	handler := NewHandler(cfg, jobRepo, db)

	// API v1 Group
	v1 := e.Group("/api/v1")

	// Public Health & Readiness endpoints
	v1.GET("/health", handler.HealthCheck)
	v1.GET("/ready", handler.ReadyCheck)
	v1.GET("/transcribe/modes", handler.GetModes)
	v1.GET("/transcribe/languages", handler.GetLanguages)

	// Protected Transcription endpoints
	transcribeGroup := v1.Group("/transcribe")
	transcribeGroup.Use(AuthMiddleware(cfg))

	transcribeGroup.POST("", handler.SubmitTranscription)
	transcribeGroup.GET("/:id/status", handler.GetStatus)
	transcribeGroup.GET("/:id/result", handler.GetResult)
	transcribeGroup.DELETE("/:id", handler.DeleteJob)

	return e
}
