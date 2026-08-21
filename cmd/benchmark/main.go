package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"whisper-service/internal/models"
)

func main() {
	baseURL := flag.String("url", "http://localhost:8080", "Base URL of Whisper Service API")
	token := flag.String("token", "dev-secret-token-change-in-production", "API Token")
	audioURL := flag.String("audio", "https://actions.google.com/sounds/v1/speech/hello_there.ogg", "Audio URL to test")
	enableDiarize := flag.Bool("diarize", true, "Enable speaker diarization")
	mode := flag.String("mode", "fast", "Transcription mode (fast, accurate, balanced)")
	flag.Parse()

	fmt.Println("==================================================")
	fmt.Println("    Whisper Service End-to-End API Benchmark     ")
	fmt.Println("==================================================")
	fmt.Printf("API Endpoint: %s\n", *baseURL)
	fmt.Printf("Audio URL:    %s\n", *audioURL)
	fmt.Printf("Mode:         %s\n", *mode)
	fmt.Printf("Diarization:  %v\n\n", *enableDiarize)

	client := &http.Client{Timeout: 30 * time.Second}

	// 1. Health check
	fmt.Print("[1/4] Testing GET /api/v1/health ... ")
	resp, err := client.Get(*baseURL + "/api/v1/health")
	if err != nil {
		fmt.Printf("FAILED: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	fmt.Printf("OK (Status %d)\n", resp.StatusCode)

	// 2. Submit transcription
	fmt.Print("[2/4] Submitting POST /api/v1/transcribe ... ")
	subReq := models.TranscriptionRequest{
		AudioURL:          *audioURL,
		EnableDiarization: *enableDiarize,
		TranscriptionMode: *mode,
	}
	reqBody, _ := json.Marshal(subReq)

	httpReq, _ := http.NewRequest(http.MethodPost, *baseURL+"/api/v1/transcribe", bytes.NewReader(reqBody))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+*token)

	submitStart := time.Now()
	resp, err = client.Do(httpReq)
	if err != nil {
		fmt.Printf("FAILED: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("FAILED (Status %d): %s\n", resp.StatusCode, string(body))
		os.Exit(1)
	}

	var subResp models.GenericAPIResponse
	_ = json.NewDecoder(resp.Body).Decode(&subResp)
	dataBytes, _ := json.Marshal(subResp.Data)
	var subData models.SubmissionData
	_ = json.Unmarshal(dataBytes, &subData)

	fmt.Printf("OK (Accepted in %v, Job ID: %s)\n", time.Since(submitStart), subData.TranscriptionID)

	// 3. Poll status
	fmt.Println("[3/4] Polling GET /api/v1/transcribe/:id/status ...")
	pollStart := time.Now()
	var finalResult models.GenericAPIResponse

	for {
		time.Sleep(1 * time.Second)

		statusReq, _ := http.NewRequest(http.MethodGet, fmt.Sprintf("%s/api/v1/transcribe/%s/status", *baseURL, subData.TranscriptionID), nil)
		statusReq.Header.Set("Authorization", "Bearer "+*token)
		sResp, err := client.Do(statusReq)
		if err != nil {
			fmt.Printf("Polling error: %v\n", err)
			continue
		}

		var sData models.StatusData
		_ = json.NewDecoder(sResp.Body).Decode(&sData)
		sResp.Body.Close()

		fmt.Printf("  -> Elapsed: %4.1fs | Stage: %-13s | Progress: %3d%% | Status: %s\n",
			time.Since(pollStart).Seconds(), sData.Stage, sData.Progress, sData.Status)

		if sData.Status == models.StatusCompleted {
			break
		}
		if sData.Status == models.StatusFailed {
			fmt.Printf("\nJob failed! Reason: %s\n", sData.ErrorMessage)
			os.Exit(1)
		}
	}

	// 4. Retrieve result
	fmt.Print("\n[4/4] Retrieving final GET /api/v1/transcribe/:id/result ... ")
	resReq, _ := http.NewRequest(http.MethodGet, fmt.Sprintf("%s/api/v1/transcribe/%s/result", *baseURL, subData.TranscriptionID), nil)
	resReq.Header.Set("Authorization", "Bearer "+*token)
	rResp, err := client.Do(resReq)
	if err != nil {
		fmt.Printf("FAILED: %v\n", err)
		os.Exit(1)
	}
	defer rResp.Body.Close()

	_ = json.NewDecoder(rResp.Body).Decode(&finalResult)
	fmt.Printf("OK (Status %d)\n\n", rResp.StatusCode)

	rBytes, _ := json.Marshal(finalResult.Data)
	var resData models.ResultData
	_ = json.Unmarshal(rBytes, &resData)

	if resData.Result != nil {
		fmt.Println("==================================================")
		fmt.Println("                TRANSCRIPTION RESULT              ")
		fmt.Println("==================================================")
		fmt.Printf("Text:              \"%s\"\n", resData.Result.TranscribedText)
		fmt.Printf("Language:          %s\n", resData.Result.Language)
		fmt.Printf("Audio Duration:    %.2fs (%s)\n", resData.Result.Duration, resData.Result.DurationFormatted)
		fmt.Printf("Processing Time:   %.2fs (%s)\n", resData.Result.ProcessingTime, resData.Result.ProcessingTimeFormatted)
		fmt.Printf("Real-Time Factor:  %.4f (Lower is better)\n", resData.Result.ProcessingTime/resData.Result.Duration)
		fmt.Printf("Number of Speakers:%d\n", resData.Result.NumSpeakers)
		fmt.Printf("Word Count:        %d\n", resData.Result.WordCount)
		fmt.Printf("Total Segments:    %d\n", len(resData.Result.Segments))
		fmt.Println("==================================================")
	}
}
