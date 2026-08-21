package security

import (
	"testing"
)

func TestValidateURL_SSRF(t *testing.T) {
	tests := []struct {
		name        string
		url         string
		expectError bool
	}{
		{"Empty URL", "", true},
		{"Invalid scheme ftp", "ftp://example.com/audio.mp3", true},
		{"Invalid scheme file", "file:///etc/passwd", true},
		{"Localhost literal", "http://localhost/audio.mp3", true},
		{"Localhost uppercase", "http://LOCALHOST:8080/audio.mp3", true},
		{"127.0.0.1 loopback", "http://127.0.0.1/audio.mp3", true},
		{"10.0.0.1 private", "http://10.0.0.1/audio.mp3", true},
		{"192.168.1.1 private", "http://192.168.1.1/audio.mp3", true},
		{"172.16.0.1 private", "http://172.16.0.1/audio.mp3", true},
		{"169.254.169.254 AWS metadata", "http://169.254.169.254/latest/meta-data/", true},
		{"Valid public HTTPS URL", "https://actions.google.com/sounds/v1/speech/hello_there.ogg", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateURL(tt.url)
			if (err != nil) != tt.expectError {
				t.Errorf("ValidateURL(%q) error = %v, expectError = %v", tt.url, err, tt.expectError)
			}
		})
	}
}
