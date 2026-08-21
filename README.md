# Whisper Asynchronous Transcription Service

A production-ready, high-throughput, asynchronous speech-to-text inference microservice. Built specifically for high accuracy and low cost, sitting underneath transcription client applications like **Exscripts.ai** and **LexScriptsAI**.

---

## Key Highlights

- **Core ML Pipeline**: `faster-whisper` (**CTranslate2 `large-v3`**) + `pyannote.audio` 3.1 Neural Speaker Diarization.
- **Architecture**: Decoupled Go Orchestrator (**Echo v4**) + Python GPU ML Worker.
- **Zero Redis Dependency**: Pure **PostgreSQL** queue engine using atomic `FOR UPDATE SKIP LOCKED` transactions. Eliminates Redis infrastructure cost and operational overhead.
- **Exact API Contract Compatibility**: Identical JSON structures for submission (`202 Accepted`), status polling, stage updates, and final transcription segments.
- **High Concurrency & Low Latency**: Go API handles rapid polling and resilient webhook retries without blocking ML GPU inference loops.
- **Production Resilience**: Worker heartbeat tracking, stale-job auto-recovery, GPU OOM safety guards, SSRF protection filters, and immediate scratch audio disk cleanup.

---

## System Architecture

```
                                  Clients
                       (Exscripts.ai, LexScriptsAI)
                                     │
                ┌────────────────────┴────────────────────┐
                │ POST /transcribe  GET /status  GET /result
                ▼                                         ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             Go API & Orchestration (Echo v4)                │
   │  - SSRF URL verification & safe download bounds             │
   │  - Auth (Bearer / X-API-Key) & Idempotency key handling     │
   │  - Asynchronous Webhook Dispatcher (Exponential Backoff)    │
   │  - Stale Job Watchdog (Auto re-queue crashed worker tasks)  │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │               PostgreSQL Database & Queue                   │
   │  - Atomic Job Leasing: SELECT ... FOR UPDATE SKIP LOCKED    │
   │  - Durable JSONB Results & Stage Tracking                   │
   │  - Webhook Delivery Ledger & Worker Telemetry Heartbeats    │
   └──────────────────────────────▲──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │              Python GPU ML Worker (Isolated)                │
   │  - Models loaded ONCE at startup (Whisper large-v3)         │
   │  - ffmpeg 16kHz mono normalization                          │
   │  - Word-level timestamps & VAD speech segmentation          │
   │  - Pyannote 3.1 Speaker Diarization & Word/Segment mapping   │
   │  - Real-time VRAM telemetry & 5s Heartbeat loop             │
   └─────────────────────────────────────────────────────────────┘
```

---

## API Reference

### 1. Health & Readiness Check
- `GET /api/v1/health`
  - Returns `200 OK` if the Go API process and DB connection are healthy.
- `GET /api/v1/ready`
  - Returns `200 OK` (or `503 Unavailable`) checking PostgreSQL connectivity, active GPU worker count, VRAM usage, and loaded models.

### 2. Supported Modes & Languages
- `GET /api/v1/transcribe/modes`
  - Returns `fast`, `accurate`, `balanced`.
- `GET /api/v1/transcribe/languages`
  - Returns supported ISO-639-1 language codes.

### 3. Submit Transcription Job
`POST /api/v1/transcribe`
```json
{
  "audio_url": "https://example.com/interview.mp3",
  "enable_diarization": true,
  "enable_summarization": false,
  "language": "en",
  "transcription_mode": "fast",
  "callback_url": "https://example.com/webhooks/transcription",
  "summary_format": "plain_text"
}
```
**Response (`202 Accepted`):**
```json
{
  "success": true,
  "data": {
    "transcription_id": "8fa85ec1-6a95-460d-85ce-bfe9f563604f",
    "status": "queued"
  },
  "message": "Transcription job submitted successfully"
}
```

### 4. Poll Job Status
`GET /api/v1/transcribe/{transcription_id}/status`
```json
{
  "progress": 45,
  "result_available": false,
  "stage": "transcribing",
  "status": "processing",
  "transcription_id": "8fa85ec1-6a95-460d-85ce-bfe9f563604f"
}
```
*Stages:* `queued` -> `downloading` -> `preprocessing` -> `transcribing` -> `diarizing` -> `aligning` -> `finalizing` -> `completed`.

### 5. Retrieve Final Result
`GET /api/v1/transcribe/{transcription_id}/result`
```json
{
  "success": true,
  "data": {
    "transcription_id": "8fa85ec1-6a95-460d-85ce-bfe9f563604f",
    "status": "completed",
    "result": {
      "segments": [
        {
          "start": 3.558,
          "end": 4.600,
          "text": "That ain't your dog for real.",
          "source_text": null,
          "speaker": "SPEAKER_00",
          "words": [
            {
              "word": "That",
              "start": 3.558,
              "end": 3.738,
              "score": 0.622,
              "speaker": "SPEAKER_00",
              "source_word": null,
              "mapping_type": null
            }
          ]
        }
      ],
      "language": "en",
      "transcribed_text": "That ain't your dog for real.",
      "source_transcribed_text": null,
      "transcription_mode": "fast",
      "model_used": "large-v3",
      "duration": 147.893,
      "duration_formatted": "2m 28s",
      "num_speakers": 2,
      "processing_time": 5.897,
      "processing_time_formatted": "5.9s",
      "word_count": 185,
      "target_language": null,
      "translation_method": null
    },
    "processing_time": 5.897
  },
  "message": "Transcription result retrieved successfully"
}
```

---

## Quickstart & Local Setup

### Prerequisites
1. **Go 1.22+**
2. **Python 3.10 / 3.11** + **CUDA 11.8 or 12.x** (with NVIDIA GPU drivers)
3. **ffmpeg** installed on system PATH (`ffmpeg -version`)
4. **PostgreSQL 14+** running locally or remotely

### 1. Database Setup
Create database in PostgreSQL:
```bash
createdb -U postgres whisper_service
```
*(The Go server automatically applies embedded migrations on startup).*

### 2. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Update your `DATABASE_URL` and `HF_TOKEN` (HuggingFace token for pyannote diarization).

### 3. Run Go API Server
```bash
# Download Go modules and run
go mod download
go run cmd/server/main.go
```
The API server starts on `http://localhost:8080`.

### 4. Run Python ML GPU Worker
In a separate terminal:
```bash
# Create and activate virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies (with PyTorch CUDA)
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r ml_worker/requirements.txt

# Start ML Worker
python -m ml_worker.worker
```

---

## Benchmarking & Verification

### 1. Standalone ML Engine Benchmark (RTF & VRAM)
```bash
python -m ml_worker.benchmark https://actions.google.com/sounds/v1/speech/hello_there.ogg --mode fast
```
Outputs Real-Time Factor (RTF), processing speed, VRAM memory peak, and word counts.

### 2. End-to-End API Integration Benchmark
```bash
go run cmd/benchmark/main.go --url http://localhost:8080 --audio https://actions.google.com/sounds/v1/speech/hello_there.ogg
```

---

## GPU Sizing & Cost Optimization Analysis

To achieve the most cost-effective production deployment for this microservice:

### 1. GPU Hardware Sizing Table

| GPU Model | VRAM | RTF (faster-whisper large-v3) | Concurrent Jobs | Approx. Cloud Cost / Hr | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA RTX 3060 / 4060** | 12 GB | ~0.04 - 0.05 | 1 - 2 | ~$0.15 - $0.22 / hr | **High Value Dedicated Entry** |
| **NVIDIA RTX 4070 Ti / 4080** | 16 GB | ~0.02 - 0.03 | 2 - 3 | ~$0.28 - $0.35 / hr | **Best Price/Performance** |
| **NVIDIA RTX 4090** | 24 GB | ~0.015 - 0.02 | 3 - 4 | ~$0.45 - $0.60 / hr | **Ultra Fast / High Throughput** |
| **NVIDIA A10G (AWS G5)** | 24 GB | ~0.025 - 0.035 | 2 - 3 | ~$1.00 / hr (On-Demand) | Enterprise AWS VPC |
| **NVIDIA T4 (AWS G4dn)** | 16 GB | ~0.08 - 0.10 | 1 | ~$0.52 / hr | Budget AWS legacy |

### 2. Cost Calculation Example
- An RTF of **0.03** means **1 hour of audio takes only ~108 seconds** of GPU compute.
- On a **$0.30/hour RTX 4070 instance**, 1 hour of audio costs approximately:
  $$\text{Cost per Audio Hour} = \frac{108\text{ s}}{3600\text{ s}} \times \$0.30 \approx \mathbf{\$0.009}\text{ (Less than 1 cent!)}$$
- Compared to third-party transcription APIs ($0.36 - $1.20 per audio hour), this self-hosted service yields **95%+ cost reduction**.

### 3. Recommended Cloud Providers
1. **RunPod / Lambda Labs (Recommended for pure GPU cost)**: Community or Secure Cloud instances (RTX 4090 or A4000 at $0.20 - $0.50/hr).
2. **Hetzner GPU / Scaleway**: Dedicated European bare-metal GPU servers with zero egress fees.
3. **AWS EC2 (g5.xlarge)**: If client services already live in AWS and require private VPC connectivity.
