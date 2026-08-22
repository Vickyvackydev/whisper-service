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

// StartPod sends a GraphQL mutation to RunPod API to resume/start the GPU pod
func (c *RunPodClient) StartPod(ctx context.Context) error {
	if !c.IsConfigured() {
		return fmt.Errorf("runpod API key or pod ID not configured")
	}

	mutation := fmt.Sprintf(`
		mutation {
			podResume(input: {podId: "%s"}) {
				id
				desiredStatus
			}
		}
	`, c.podID)

	reqBody := map[string]string{
		"query": mutation,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal runpod start request: %w", err)
	}

	url := fmt.Sprintf("https://api.runpod.io/graphql?api_key=%s", c.apiKey)
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
		return fmt.Errorf("runpod API returned non-200 status code: %d", resp.StatusCode)
	}

	log.Printf("[RunPod Client] Successfully sent podResume signal for Pod ID: %s", c.podID)
	return nil
}
