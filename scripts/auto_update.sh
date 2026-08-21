#!/bin/bash
# scripts/auto_update.sh
# Checks GitHub repository every 30 seconds.
# If changes are pushed to main, it automatically pulls and restarts both services.

REPO_DIR="/whisper-service"
BRANCH="main"
CHECK_INTERVAL_SEC=30

cd "$REPO_DIR" || exit 1

echo "[$(date)] Auto-update daemon started for branch: $BRANCH"

while true; do
    git fetch origin "$BRANCH" --quiet 2>/dev/null
    
    LOCAL_HASH=$(git rev-parse HEAD)
    REMOTE_HASH=$(git rev-parse origin/"$BRANCH")
    
    if [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
        echo "[$(date)] New update detected on GitHub! ($LOCAL_HASH -> $REMOTE_HASH)"
        echo "[$(date)] Pulling latest changes..."
        git pull origin "$BRANCH"
        
        echo "[$(date)] Restarting API and ML Worker services..."
        pkill -f "cmd/server/main.go"
        pkill -f "main.go"
        pkill -f "ml_worker.worker"
        sleep 2
        
        nohup go run cmd/server/main.go > api.log 2>&1 &
        nohup python -m ml_worker.worker > worker.log 2>&1 &
        
        echo "[$(date)] Services successfully restarted with new changes!"
    fi
    
    sleep "$CHECK_INTERVAL_SEC"
done
