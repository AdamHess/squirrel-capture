# Handoff: Squirrel Capture Pipeline Deployment

## For the agent on the Linux server

Run `deploy/handoff.sh` as root on the target server (192.168.1.251). It will:

1. Install Python, git, and deps
2. Clone the repo to `/opt/squirrel-capture`
3. Create a venv and install Python packages
4. Write `config.yaml` with the Reolink camera at 192.168.1.204 (admin / PeriodCake2024!)
5. Enable GPU inference (`device: cuda:0`) for the GT 1030
6. Install a systemd service (`squirrel-capture`) that auto-starts on boot
7. Start the pipeline

## After deployment

- **Logs**: `journalctl -fu squirrel-capture`
- **Config**: `/opt/squirrel-capture/config.yaml`
- **Data**: `/opt/squirrel-capture/data/labeled/` -- images + YOLO labels

## Model registry

Trained models are tracked in `models/registry.yaml` on the desktop.
Each entry stores date, dataset, classes, and metrics.
Models are referenced by name (e.g. `nyc-backyard-v1`) in config and
automatically resolved to the local .pt path.

To list registered models:
```
uv run python -c "from models.registry import list_models; print(list_models())"
```

To register a new model after training (automatic if using `training/train.py`):
```python
from models.registry import register_model
register_model("my-model", "runs/.../best.pt", ...)
```

Available prelabeled models:

| Name | Classes | Images | mAP50 | Base |
|------|---------|--------|-------|------|
| `nyc-backyard-v1` | Squirrel, Bird, Raccoon, Cat, Mouse, Person | 1,435 | 0.700 | yolo11s |

## Training loop (on Windows desktop)

### Prerequisites
```powershell
$env:CUDA_VISIBLE_DEVICES="0"
cd C:\Users\Adam\squirrel-capture
```

### Finetune from a prelabeled model on your own data
```powershell
uv run python -m training.train ^
  --model nyc-backyard-v1 ^
  --data data/labeled/data.yaml ^
  --epochs 50 --batch 16 --device 0 ^
  --name finetune-v1 --model-type finetuned
```

### Train a new prelabeled model from a public dataset
```powershell
# Download classes from OpenImages (skipping Rat -- not in boxable subset)
uv run oi_download_dataset --base_dir data/openimages --labels Squirrel Bird Raccoon Cat Mouse Person --format darknet --limit 300 --csv_dir data/oid_csv

# Consolidate into YOLO train/val
uv run python training/consolidate_openimages.py

# Train
uv run python -m training.train --data data/exports/nyc-backyard-v1/data.yaml --model yolo11s.pt --epochs 100 --batch 16 --device 0 --name my-prelabeled-v1 --model-type prelabeled
```

### Add nighttime augmentations (squirrel focus)
```powershell
uv run python training/augment_night.py --src data/exports/nyc-backyard-v1 --dst data/exports/nyc-backyard-night-v1 --factor 0.5

uv run python -m training.train --data data/exports/nyc-backyard-night-v1/data.yaml --model nyc-backyard-v1 --epochs 50 --device 0 --name nyc-backyard-night-v1 --model-type finetuned
```

### Deploy weights to server
```powershell
scp runs/detect/runs/nyc-backyard-night-v1/weights/best.pt root@192.168.1.251:/opt/squirrel-capture/deploy/nyc-backyard-v1.pt

# On server:
ssh root@192.168.1.251
sed -i 's|model:.*|model: "deploy/nyc-backyard-v1.pt"|' /opt/squirrel-capture/config.yaml
systemctl restart squirrel-capture
```

## Configuration

Key settings in `config.yaml`:

```yaml
camera:
  rtsp_url: "rtsp://admin:PeriodCake2024!@192.168.1.204:554/Preview_01_main"

detection:
  model: "nyc-backyard-v1"    # name in models/registry.yaml, or path to .pt
  conf_threshold: 0.25
  target_classes: [0]         # [0] = Squirrel (our class 0), [] = all
  device: "cpu"               # "cuda:0" on server, "0" on desktop

tracker:
  enabled: true
  method: "bytetrack"
  max_lost: 30
```

## Credentials

| Resource | Details |
|----------|---------|
| Server | root / 11215 @ 192.168.1.251:22 |
| Camera | admin / PeriodCake2024! @ 192.168.1.204:554 |
| GitHub | https://github.com/AdamHess/squirrel-capture |
