#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/AdamHess/squirrel-capture.git"
INSTALL_DIR="/opt/squirrel-capture"
SERVICE_USER="pi"

echo "=== Installing squirrel-capture pipeline ==="

# Install system dependencies
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

# Clone repo
sudo git clone "$REPO_URL" "$INSTALL_DIR"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# Create venv and install Python deps
cd "$INSTALL_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Copy service file and enable
sudo cp deploy/squirrel-capture.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable squirrel-capture
sudo systemctl start squirrel-capture

echo ""
echo "=== Done ==="
echo "Edit config:  sudo -u $SERVICE_USER nano $INSTALL_DIR/config.yaml"
echo "View logs:    sudo journalctl -fu squirrel-capture"
echo "Restart:      sudo systemctl restart squirrel-capture"
echo "Stop:         sudo systemctl stop squirrel-capture"
