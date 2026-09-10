#!/bin/bash
# scripts/auto_update.sh
# Checks GitHub repository every 30 seconds.
# Supports both Public and Private GitHub repositories (via SSH deploy keys or credential helper).
# Automatically pulls changes, rebuilds Go API, and restarts systemd service.

REPO_DIR="${REPO_DIR:-/opt/whisper-service}"
BRANCH="${BRANCH:-main}"
CHECK_INTERVAL_SEC=30

cd "$REPO_DIR" || exit 1

echo "[$(date)] Auto-update daemon started for branch: $BRANCH in $REPO_DIR"

while true; do
    # Fetch latest remote changes (works for private repos if SSH deploy key or PAT is stored)
    git fetch origin "$BRANCH" --quiet 2>/dev/null
    
    LOCAL_HASH=$(git rev-parse HEAD 2>/dev/null || echo "")
    REMOTE_HASH=$(git rev-parse origin/"$BRANCH" 2>/dev/null || echo "")
    
    if [ -n "$LOCAL_HASH" ] && [ -n "$REMOTE_HASH" ] && [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
        echo "[$(date)] New update detected on GitHub! ($LOCAL_HASH -> $REMOTE_HASH)"
        echo "[$(date)] Pulling latest changes..."
        git pull origin "$BRANCH"
        
        echo "[$(date)] Rebuilding Go API binary..."
        go build -o bin/server cmd/server/main.go
        
        echo "[$(date)] Restarting whisper-api systemd service..."
        if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet whisper-api; then
            sudo systemctl restart whisper-api
            echo "[$(date)] whisper-api systemctl restart completed!"
        else
            pkill -f "cmd/server/main.go" || true
            pkill -f "bin/server" || true
            sleep 1
            nohup bin/server > api.log 2>&1 &
            echo "[$(date)] Process restarted in background!"
        fi
    fi
    
    sleep "$CHECK_INTERVAL_SEC"
done

