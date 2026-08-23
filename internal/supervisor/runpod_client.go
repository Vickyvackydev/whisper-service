package supervisor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

type RunPodClient struct {
	apiKey     string
	podID      string
	httpClient *http.Client
}

func NewRunPodClient(apiKey string, podID string) *RunPodClient {
	return &RunPodClient{
		apiKey: apiKey,
		podID:  podID,
		httpClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

func (c *RunPodClient) IsConfigured() bool {
	return c.apiKey != "" && c.podID != ""
}

// StartPod sends a REST API v2 request to start/resume the GPU pod
func (c *RunPodClient) StartPod(ctx context.Context) error {
	if !c.IsConfigured() {
		return fmt.Errorf("runpod API key or pod ID not configured")
	}

	url := fmt.Sprintf("https://api.runpod.io/v2/pods/%s/action", c.podID)
	reqBody := map[string]string{
		"action": "start",
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal runpod start request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return fmt.Errorf("failed to create runpod request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute runpod start API call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("runpod REST API v2 returned non-200 status code: %d", resp.StatusCode)
	}

	log.Printf("[RunPod Client] Successfully sent REST API v2 start action for Pod ID: %s", c.podID)
	return nil
}
