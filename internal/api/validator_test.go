package api

import (
	"testing"
	"whisper-service/internal/models"
)

func TestValidateTranscriptionRequest(t *testing.T) {
	validURL := "https://actions.google.com/sounds/v1/speech/hello_there.ogg"
	invalidSSRF := "http://127.0.0.1:8080/audio.wav"

	tests := []struct {
		name        string
		req         models.TranscriptionRequest
		expectError bool
	}{
		{
			name: "Valid request default mode",
			req: models.TranscriptionRequest{
				AudioURL:          validURL,
				EnableDiarization: true,
			},
			expectError: false,
		},
		{
			name: "Valid request with mode accurate",
			req: models.TranscriptionRequest{
				AudioURL:          validURL,
				TranscriptionMode: "accurate",
			},
			expectError: false,
		},
		{
			name: "Invalid mode",
			req: models.TranscriptionRequest{
				AudioURL:          validURL,
				TranscriptionMode: "super_fast_invalid",
			},
			expectError: true,
		},
		{
			name: "Empty audio_url",
			req: models.TranscriptionRequest{
				AudioURL: "",
			},
			expectError: true,
		},
		{
			name: "SSRF audio_url rejected",
			req: models.TranscriptionRequest{
				AudioURL: invalidSSRF,
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateTranscriptionRequest(&tt.req)
			if (err != nil) != tt.expectError {
				t.Errorf("ValidateTranscriptionRequest() error = %v, expectError = %v", err, tt.expectError)
			}
		})
	}
}
