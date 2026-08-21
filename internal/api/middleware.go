package api

import (
	"crypto/subtle"
	"net/http"
	"strings"

	"github.com/labstack/echo/v4"
	"whisper-service/internal/config"
	"whisper-service/internal/models"
)

// AuthMiddleware authenticates requests using Bearer token or X-API-Key header
func AuthMiddleware(cfg *config.Config) echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			if !cfg.AuthEnabled {
				return next(c)
			}

			// Extract token from Authorization header or X-API-Key
			token := ""
			authHeader := c.Request().Header.Get("Authorization")
			if authHeader != "" {
				parts := strings.SplitN(authHeader, " ", 2)
				if len(parts) == 2 && strings.EqualFold(parts[0], "Bearer") {
					token = strings.TrimSpace(parts[1])
				}
			}

			if token == "" {
				token = strings.TrimSpace(c.Request().Header.Get("X-API-Key"))
			}

			if token == "" {
				return c.JSON(http.StatusUnauthorized, models.GenericAPIResponse{
					Success: false,
					Message: "Missing authorization credentials",
					Error: &models.APIError{
						Code:    "UNAUTHORIZED",
						Message: "An API token is required via Authorization: Bearer <TOKEN> or X-API-Key header",
					},
				})
			}

			// Constant time token comparison against configured tokens
			valid := false
			for _, configuredToken := range cfg.APITokens {
				if subtle.ConstantTimeCompare([]byte(token), []byte(configuredToken)) == 1 {
					valid = true
					break
				}
			}

			if !valid {
				return c.JSON(http.StatusUnauthorized, models.GenericAPIResponse{
					Success: false,
					Message: "Invalid API token",
					Error: &models.APIError{
						Code:    "INVALID_CREDENTIALS",
						Message: "The provided API token is not valid",
					},
				})
			}

			return next(c)
		}
	}
}

// CORS headers for client integration
func CORS() echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			c.Response().Header().Set("Access-Control-Allow-Origin", "*")
			c.Response().Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			c.Response().Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, X-Idempotency-Key")

			if c.Request().Method == http.MethodOptions {
				return c.NoContent(http.StatusOK)
			}

			return next(c)
		}
	}
}
