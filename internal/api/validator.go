package api

import (
	"errors"
	"strings"

	"whisper-service/internal/models"
	"whisper-service/internal/security"
)

var (
	ValidTranscriptionModes = map[string]bool{
		"fast":     true,
		"accurate": true,
		"balanced": true,
	}

	ValidSummaryFormats = map[string]bool{
		"plain_text": true,
		"markdown":   true,
		"bullet":     true,
		"json":       true,
	}
)

func ValidateTranscriptionRequest(req *models.TranscriptionRequest) error {
	req.AudioURL = strings.TrimSpace(req.AudioURL)
	if req.AudioURL == "" {
		return errors.New("audio_url is required")
	}

	// Validate audio_url for SSRF & format
	if err := security.ValidateURL(req.AudioURL); err != nil {
		return err
	}

	// Mode validation
	req.TranscriptionMode = strings.ToLower(strings.TrimSpace(req.TranscriptionMode))
	if req.TranscriptionMode == "" {
		req.TranscriptionMode = "fast"
	}
	if !ValidTranscriptionModes[req.TranscriptionMode] {
		return errors.New("invalid transcription_mode: must be 'fast', 'accurate', or 'balanced'")
	}

	// Summary format validation
	req.SummaryFormat = strings.ToLower(strings.TrimSpace(req.SummaryFormat))
	if req.SummaryFormat == "" {
		req.SummaryFormat = "plain_text"
	}
	if !ValidSummaryFormats[req.SummaryFormat] {
		return errors.New("invalid summary_format: must be 'plain_text', 'markdown', 'bullet', or 'json'")
	}

	// Callback URL validation if supplied
	if req.CallbackURL != nil {
		trimmedCallback := strings.TrimSpace(*req.CallbackURL)
		if trimmedCallback != "" {
			if err := security.ValidateURL(trimmedCallback); err != nil {
				return errors.New("invalid callback_url: " + err.Error())
			}
			req.CallbackURL = &trimmedCallback
		} else {
			req.CallbackURL = nil
		}
	}

	// Language code trimming
	if req.Language != nil {
		trimmedLang := strings.ToLower(strings.TrimSpace(*req.Language))
		if trimmedLang != "" {
			req.Language = &trimmedLang
		} else {
			req.Language = nil
		}
	}

	// Target language code trimming
	if req.TargetLanguage != nil {
		trimmedTarget := strings.ToLower(strings.TrimSpace(*req.TargetLanguage))
		if trimmedTarget != "" {
			req.TargetLanguage = &trimmedTarget
			// If target_language is explicitly provided, auto-enable translation if target differs or is en
			if req.Language == nil || *req.Language != trimmedTarget {
				req.EnableTranslation = true
			}
		} else {
			req.TargetLanguage = nil
		}
	}

	return nil
}
