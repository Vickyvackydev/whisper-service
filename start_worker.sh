#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "Stopping any existing ml_worker.worker process..."
pkill -f "ml_worker.worker" || true
sleep 2

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "/workspace/venv/bin/activate" ]; then
    source /workspace/venv/bin/activate
fi

echo "Starting GPU ML Worker in background (logging to worker.log)..."
nohup python3 -m ml_worker.worker > worker.log 2>&1 &
sleep 1
echo "Worker started! Tailing worker.log (Press Ctrl+C to exit log view; worker will stay running):"
tail -f worker.log

