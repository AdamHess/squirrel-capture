# Squirrel Capture Pipeline

Auto-labeling pipeline for collecting squirrel training data from a Reolink camera. Runs capture/inference on a **Linux server** near the camera, trains on a **Windows desktop** with a GTX 2080.

## Quick Start (Linux Server - Capture)

```bash
pip install -r requirements.txt
```

Edit `config.yaml` with your camera's RTSP URL, then:

```bash
python pipeline.py
```

Only motion-triggered frames with YOLO detections are saved. Images and YOLO-format labels go to `data/labeled/`.

## Data Format

```
data/labeled/
  images/           # JPEG frames
  labels/           # YOLO .txt labels (class_id cx cy w h, normalized 0-1)
```

Label example (`9 0.45 0.38 0.12 0.21`):
- Class 9 = squirrel (COCO)
- cx, cy, w, h are normalized to [0, 1]

## Training (Windows Desktop)

Sync `data/labeled/` from the server, then:

```bash
python training/export_dataset.py --input data/labeled --output data/exports/dataset
python training/train.py --data data/exports/dataset --epochs 100 --device 0
```

Best weights saved to `runs/squirrel/weights/best.pt`. Copy back to the server and update `config.yaml`'s `detection.model` path.

## Config

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `camera` | `rtsp_url` | (required) | Reolink RTSP URL |
| `motion` | `min_area` | 3000 | Min contour px to trigger |
| `detection` | `model` | `yolo11n.pt` | Pretrained model path |
| `detection` | `target_classes` | `[9]` | COCO classes to detect (`[]` = all) |
| `capture` | `max_images_per_hour` | 120 | Rate limit |

## Workflow

1. **Server**: `pipeline.py` reads RTSP, detects motion, runs YOLO, saves labeled images
2. **Server -> Desktop**: sync `data/labeled/` (rsync, SMB, etc.)
3. **Desktop**: `export_dataset.py` + `train.py` fine-tunes YOLO on the 2080
4. **Desktop -> Server**: copy `best.pt` back, update config
5. Repeat: accuracy improves with each cycle
