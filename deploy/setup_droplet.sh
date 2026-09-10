#!/bin/bash
# deploy/setup_droplet.sh
# One-liner automated setup script for DigitalOcean Droplet

set -e

echo "=================================================="
echo "  Whisper Service: DigitalOcean Droplet Installer "
echo "=================================================="

# 1. Update and install prerequisites
echo "[1/4] Updating system packages & installing Docker..."
sudo apt-get update -y
sudo apt-get install -y curl git docker.io docker-compose-v2 ufw

# Enable and start Docker
sudo systemctl enable docker
sudo systemctl start docker

# 2. Configure Firewall (UFW)
echo "[2/4] Configuring UFW Firewall..."
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8080/tcp    # Go API
sudo ufw allow 5432/tcp    # PostgreSQL DB for RunPod worker
sudo ufw --force enable || true

# 3. Create docker-compose environment
echo "[3/4] Generating production environment configuration..."
read -sp "Enter secure PostgreSQL Password for database: " DB_PASS
echo ""

cat <<EOF > deploy/.env.droplet
POSTGRES_PASSWORD=${DB_PASS}
API_TOKENS=whp_live_9f83a7c6e4b2d109f582bc194a0d8e72fa91cb45
PORT=8080
ENVIRONMENT=production
EOF

# 4. Start Services
echo "[4/4] Starting PostgreSQL & Go API containers..."
docker compose -f deploy/docker-compose.droplet.yml --env-file deploy/.env.droplet up -d --build

DROPLET_IP=$(curl -s ifconfig.me || hostname -I | awk '{print $1}')

echo "=================================================="
echo "  DEPLOYMENT SUCCESSFUL!                         "
echo "=================================================="
echo "API Endpoint URL : http://${DROPLET_IP}:8080"
echo "PostgreSQL URL   : postgres://postgres:${DB_PASS}@${DROPLET_IP}:5432/whisper_service?sslmode=disable"
echo "Live API Token   : whp_live_9f83a7c6e4b2d109f582bc194a0d8e72fa91cb45"
echo ""
echo "Next Step: Use the PostgreSQL URL above when launching your RunPod GPU worker!"
echo "=================================================="
