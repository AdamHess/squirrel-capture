#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# HANDOFF: Squirrel Capture Pipeline Setup
# This script completes the deployment on the Linux server.
# Run as root on the target server.
#
# Instructions:
#   1. Copy this script to the server
#   2. chmod +x handoff.sh
#   3. ./handoff.sh
# ============================================================

REPO_URL="https://github.com/AdamHess/squirrel-capture.git"
INSTALL_DIR="/opt/squirrel-capture"

# Camera credentials (update if different)
CAMERA_IP="192.168.1.204"
CAMERA_USER="admin"
CAMERA_PASS="PeriodCake2024!"

echo "=== 1. Installing system dependencies ==="
apt update
apt install -y python3 python3-pip python3-venv curl git
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env 2>/dev/null || true

echo "=== 2. Cloning repo ==="
git clone "$REPO_URL" "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "=== 3. Installing Python deps with uv ==="
uv sync

echo "=== 4. Writing config.yaml ==="
cat > config.yaml << 'CONFIGEOF'
---
camera:
  rtsp_url: "rtsp://admin:PeriodCake2024!@192.168.1.204:554/Preview_01_main"
  reconnect_interval: 5
  timeout: 10
  decode_every_n: 5

motion:
  enabled: true
  method: "mog2"
  min_area: 3000
  threshold: 25
  cooldown: 1.0

detection:
  model: "deploy/nyc-backyard-v2.pt"
  conf_threshold: 0.25
  iou_threshold: 0.45
  target_classes: [0]
  device: "cuda:0"

tracker:
  enabled: true
  method: "bytetrack"
  max_lost: 30
  iou_threshold: 0.3

capture:
  output_dir: "data"
  label_format: "yolo"
  save_raw: true
  save_labeled: true
  save_annotated: false
  min_confidence: 0.3
  track_cooldown: 3
  quality:
    min_blur: 50
    min_box_area: 2000
    max_edge_margin: 10

export:
  split_ratio: [0.8, 0.2, 0.0]
  include_augmented: false
CONFIGEOF

# Test the config won't break YAML parsing
python3 -c "import yaml; yaml.safe_load(open('config.yaml')); print('Config OK')"

echo "=== 5. Copying trained model weights ==="
# After training on desktop, upload best.pt to deploy/:
#   scp runs/detect/runs/nyc-backyard-v2/weights/best.pt root@192.168.1.251:/opt/squirrel-capture/deploy/nyc-backyard-v2.pt
mkdir -p deploy
if [ ! -f deploy/nyc-backyard-v2.pt ]; then
    echo "WARNING: deploy/nyc-backyard-v2.pt not found. Download from GitHub releases or run:"
    echo "  scp runs/detect/runs/nyc-backyard-v2/weights/best.pt root@192.168.1.251:/opt/squirrel-capture/deploy/nyc-backyard-v2.pt"
    echo "Falling back to yolo11n.pt for initial setup."
    sed -i 's|deploy/nyc-backyard-v2.pt|yolo11n.pt|' config.yaml
fi

echo "=== 6. Installing systemd service ==="
# Determine the non-root user to run as (use the first regular user)
SERVICE_USER=$(who am i | awk '{print $1}' || echo "root")
if [ "$SERVICE_USER" = "root" ]; then
    SERVICE_USER="pi"
    # Create pi user if it doesn't exist
    id -u pi &>/dev/null || useradd -m -s /bin/bash pi
fi

chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

cat > /etc/systemd/system/squirrel-capture.service << SERVICEEOF
[Unit]
Description=Squirrel capture pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/.venv/bin/python pipeline.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl daemon-reload
systemctl enable squirrel-capture

echo "=== 7. Starting pipeline ==="
systemctl start squirrel-capture

echo ""
echo "============================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================"
echo ""
echo "  Status:       systemctl status squirrel-capture"
echo "  Logs:         journalctl -fu squirrel-capture"
echo "  Restart:      systemctl restart squirrel-capture"
echo "  Stop:         systemctl stop squirrel-capture"
echo ""
echo "  Config:       nano $INSTALL_DIR/config.yaml"
echo "  Data:         $INSTALL_DIR/data/labeled/"
echo ""
echo "  After collecting data, rsync to desktop:"
echo "  rsync -av $INSTALL_DIR/data/labeled/ user@desktop-ip:~/squirrel-capture/data/labeled/"
echo ""

# Quick health check (wait 10s then check status)
sleep 10
systemctl is-active --quiet squirrel-capture && echo "Service is RUNNING" || echo "Service FAILED - check 'journalctl -u squirrel-capture -n 50 --no-pager'"
