#!/bin/bash
# deploy/setup_droplet_systemd.sh
# Native Systemd + PostgreSQL Installer for DigitalOcean Droplet (No Docker required)

set -e

echo "=================================================="
echo "  Whisper Service: Systemd Native Droplet Installer"
echo "=================================================="

# 1. Update & Install Prerequisites (PostgreSQL, Go, Git, UFW)
echo "[1/5] Installing native packages (PostgreSQL, Go, Git)..."
sudo apt-get update -y
sudo apt-get install -y postgresql postgresql-contrib golang-go git ufw curl

# Start PostgreSQL service
sudo systemctl enable --now postgresql

# 2. Database Setup
echo "[2/5] Setting up PostgreSQL database & credentials..."
read -sp "Enter secure password for PostgreSQL 'postgres' user: " DB_PASS
echo ""

sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD '${DB_PASS}';"
sudo -u postgres psql -c "CREATE DATABASE whisper_service;" || true

# Allow PostgreSQL remote connections from RunPod GPU worker
PG_VER=$(ls /etc/postgresql/ 2>/dev/null | head -n 1 || echo "14")
CONF_FILE="/etc/postgresql/${PG_VER}/main/postgresql.conf"
HBA_FILE="/etc/postgresql/${PG_VER}/main/pg_hba.conf"

if [ -f "$CONF_FILE" ]; then
    sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/g" "$CONF_FILE"
    sudo sed -i "s/listen_addresses = 'localhost'/listen_addresses = '*'/g" "$CONF_FILE"
fi

if [ -f "$HBA_FILE" ]; then
    if ! grep -q "0.0.0.0/0" "$HBA_FILE"; then
        echo "host    whisper_service     all             0.0.0.0/0               scram-sha-256" | sudo tee -a "$HBA_FILE"
    fi
fi

sudo systemctl restart postgresql

# 3. Build Go API Binary
echo "[3/5] Building Go API server binary..."
mkdir -p bin
go build -o bin/server cmd/server/main.go
chmod +x bin/server

# 4. Generate Production .env File
echo "[4/5] Creating /opt/whisper-service/.env configuration..."
INSTALL_DIR=$(pwd)

cat <<EOF > .env
PORT=8080
ENVIRONMENT=production
DATABASE_URL=postgres://postgres:${DB_PASS}@localhost:5432/whisper_service?sslmode=disable
AUTH_ENABLED=true
API_TOKENS=whp_live_9f83a7c6e4b2d109f582bc194a0d8e72fa91cb45
AUTO_START_GPU=false
EOF

# 5. Install Systemd Service
echo "[5/5] Installing whisper-api.service into systemd..."
cat <<EOF | sudo tee /etc/systemd/system/whisper-api.service
[Unit]
Description=Whisper Transcription Service - Go API & Orchestrator
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${INSTALL_DIR}/.env
ExecStart=${INSTALL_DIR}/bin/server
Restart=always
RestartSec=5s
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now whisper-api

# 6. Configure UFW Firewall
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8080/tcp    # Go API Server
sudo ufw allow 5432/tcp    # PostgreSQL for RunPod Worker
sudo ufw --force enable || true

DROPLET_IP=$(curl -s ifconfig.me || hostname -I | awk '{print $1}')

echo "=================================================="
echo "  NATIVE SYSTEMD DEPLOYMENT SUCCESSFUL!          "
echo "=================================================="
echo "API Endpoint URL : http://${DROPLET_IP}:8080"
echo "PostgreSQL URL   : postgres://postgres:${DB_PASS}@${DROPLET_IP}:5432/whisper_service?sslmode=disable"
echo "Live API Token   : whp_live_9f83a7c6e4b2d109f582bc194a0d8e72fa91cb45"
echo ""
echo "Service Commands:"
echo "  Check status  : systemctl status whisper-api"
echo "  View logs     : journalctl -u whisper-api -f"
echo "  Restart API   : systemctl restart whisper-api"
echo "=================================================="
