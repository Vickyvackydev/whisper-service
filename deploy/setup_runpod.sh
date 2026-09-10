#!/bin/bash
# deploy/setup_runpod.sh
# Automated setup script for RunPod GPU ML Worker

set -e

echo "=================================================="
echo "  Whisper Service: RunPod GPU Worker Installer    "
echo "=================================================="

if [ -z "$1" ]; then
  echo "Error: Missing PostgreSQL Database URL."
  echo "Usage: ./deploy/setup_runpod.sh 'postgres://postgres:password@DROPLET_IP:5432/whisper_service?sslmode=disable'"
  exit 1
fi

DB_URL="$1"
HF_TOKEN="${2:-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}"

# 1. Install System Dependencies
echo "[1/4] Installing system packages & ffmpeg..."
apt-get update -y && apt-get install -y ffmpeg git python3-pip python3-venv

# 2. Setup Python Virtual Environment
echo "[2/4] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch & Python Dependencies
echo "[3/4] Installing PyTorch (CUDA 12.1) & ML dependencies..."
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r ml_worker/requirements.txt

# 4. Generate Worker Environment
echo "[4/4] Generating ML Worker environment configuration..."
cat <<EOF > .env
DATABASE_URL=${DB_URL}
WORKER_ID=gpu-worker-runpod-1
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8_float16
HF_TOKEN=${HF_TOKEN}
ENABLE_DIARIZATION=true
IDLE_SHUTDOWN_MINUTES=0
AUTO_START_GPU=false
EOF

echo "=================================================="
echo "  RUNPOD WORKER INSTALLED SUCCESSFULLY!          "
echo "=================================================="
echo "Starting ML Worker loop with pre-warmed VRAM..."
python -m ml_worker.worker
