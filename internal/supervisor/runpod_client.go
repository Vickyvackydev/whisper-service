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

// StartPod sends a REST API request to start/resume the GPU pod
func (c *RunPodClient) StartPod(ctx context.Context) error {
	if !c.IsConfigured() {
		return fmt.Errorf("runpod API key or pod ID not configured")
	}

	// Method 1: RunPod REST API v1 endpoint (POST https://rest.runpod.io/v1/pods/{podId}/start)
	v1URL := fmt.Sprintf("https://rest.runpod.io/v1/pods/%s/start", c.podID)
	req, err := http.NewRequestWithContext(ctx, "POST", v1URL, nil)
	if err == nil {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.httpClient.Do(req)
		if err == nil {
			defer resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				log.Printf("[RunPod Client] Successfully resumed Pod ID: %s via REST API v1", c.podID)
				return nil
			}
			log.Printf("[RunPod Client] REST API v1 returned status %d. Retrying with REST v2 API...", resp.StatusCode)
		}
	}

	// Method 2: RunPod REST API v2 action endpoint
	v2URL := fmt.Sprintf("https://api.runpod.io/v2/pods/%s/action", c.podID)
	reqBody := map[string]string{"action": "start"}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal runpod start request: %w", err)
	}

	req2, err := http.NewRequestWithContext(ctx, "POST", v2URL, bytes.NewBuffer(jsonBody))
	if err != nil {
		return fmt.Errorf("failed to create runpod request: %w", err)
	}

	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp2, err := c.httpClient.Do(req2)
	if err != nil {
		return fmt.Errorf("failed to execute runpod start API call: %w", err)
	}
	defer resp2.Body.Close()

	if resp2.StatusCode < 200 || resp2.StatusCode >= 300 {
		return fmt.Errorf("runpod REST API returned status code: %d (check GPU availability or pod ID)", resp2.StatusCode)
	}

	log.Printf("[RunPod Client] Successfully sent start action for Pod ID: %s", c.podID)
	return nil
}

