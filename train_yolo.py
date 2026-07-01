"""
CLI wrapper for Ultralytics YOLO training.

Usage:
  python train_yolo.py --model best.pt --name my-run --batch 6 --lr 0.003 --epochs 50
  python train_yolo.py --model yolo11s.pt --name test --epochs 10 --imgsz 640
  python train_yolo.py --model last.pt --name resume-me --resume
  python train_yolo.py --model best.pt --name v5 --freeze 10 --cos-lr --register --deploy
  python train_yolo.py --model yolo11s.pt --name distill-v1 --distill runs/trapper-v3/weights/best.pt --batch 16 --lr 0.001 --epochs 100

Defaults:
  - data: D:/squrrel_training/finetune/data.yaml
  - imgsz: 960
  - workers: 2
  - patience: 15
  - optimizer: SGD
  - project: runs
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent
DATA_YAML = "D:/squrrel_training/finetune/data.yaml"
REGISTRY = REPO_ROOT / "models" / "registry.yaml"
DEPLOY_DIR = REPO_ROOT / "deploy"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model")
    parser.add_argument("--model", required=True, help="Path to .pt weights")
    parser.add_argument("--name", required=True, help="Run directory name")
    parser.add_argument("--data", default=DATA_YAML, help=f"Dataset YAML (default: {DATA_YAML})")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size (default: 960)")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer (default: SGD)")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze first N layers")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM for faster training")
    parser.add_argument("--single-cls", action="store_true", help="Single class (overrides model nc)")
    parser.add_argument("--register", action="store_true", help="Add to registry.yaml after training")
    parser.add_argument("--deploy", action="store_true", help="Copy best.pt to deploy/ after training")
    parser.add_argument("--distill", default=None, help="Path to teacher model for distillation")
    return parser.parse_args()


def register_in_registry(name: str, path: str, metrics: dict, args, notes: str = ""):
    """Append a model entry to registry.yaml."""
    registry_path = Path(REGISTRY)
    if not registry_path.exists():
        print(f"[WARN] Registry not found at {registry_path}, skipping registration")
        return

    import yaml
    try:
        with open(registry_path) as f:
            reg = yaml.safe_load(f) or {}
    except Exception:
        reg = {}

    entry = {
        "type": "finetuned",
        "date": __import__("datetime").date.today().isoformat(),
        "path": str(path),
        "classes": ["squirrel"],
        "num_classes": 1,
        "base_model": str(args.model),
        "dataset": args.data,
        "num_images": None,  # could parse from data.yaml
        "metrics": {
            "mAP50": round(metrics.get("mAP50", 0), 3),
            "mAP50-95": round(metrics.get("mAP50-95", 0), 3),
            "precision": round(metrics.get("precision", 0), 3),
            "recall": round(metrics.get("recall", 0), 3),
        },
        "notes": notes,
    }
    reg[name] = entry

    with open(registry_path, "w") as f:
        yaml.dump(reg, f, default_flow_style=False, sort_keys=False)
    print(f"[REGISTRY] Added '{name}' to {registry_path}")


def main():
    args = parse_args()

    # Build kwargs from args, skipping None defaults
    kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": 2,
        "patience": 15,
        "optimizer": args.optimizer,
        "lr0": args.lr,
        "cos_lr": args.cos_lr,
        "device": 0,
        "project": "runs",
        "name": args.name,
        "save": True,
        "save_period": 10,
        "cache": args.cache,
        "pretrained": True,
        "amp": not args.no_amp,
    }
    if args.freeze is not None:
        kwargs["freeze"] = args.freeze
    if args.single_cls:
        kwargs["single_cls"] = True
    if args.distill:
        kwargs["distill_model"] = args.distill

    print(f"{'='*60}")
    print(f"  Training: {args.name}")
    print(f"  Model:    {args.model}")
    print(f"  Batch:    {args.batch} | LR: {args.lr} | Epochs: {args.epochs} | ImgSz: {args.imgsz}")
    print(f"  Opt:      {args.optimizer} | CosLR: {args.cos_lr} | Freeze: {args.freeze or 0}")
    if args.distill:
        print(f"  Distill:  teacher={args.distill}")
    print(f"{'='*60}")

    model = YOLO(args.model)
    results = model.train(**kwargs)

    r = results.box
    mAP50 = r.map50
    mAP50_95 = r.map
    precision = float(r.p.mean()) if hasattr(r.p, 'mean') else float(r.p)
    recall = float(r.r.mean()) if hasattr(r.r, 'mean') else float(r.r)
    print(f"\n{'='*60}")
    print(f"  Results: mAP50={mAP50:.4f}  mAP50-95={mAP50_95:.4f}")
    print(f"           P={precision:.4f}  R={recall:.4f}")
    print(f"{'='*60}")

    # Locate best.pt
    run_dir = REPO_ROOT / "runs" / args.name
    if not run_dir.exists():
        run_dir = REPO_ROOT / "runs" / "detect" / "runs" / args.name
    best_pt = run_dir / "weights" / "best.pt"

    metrics = {"mAP50": mAP50, "mAP50-95": mAP50_95, "precision": precision, "recall": recall}

    if args.register and best_pt.exists():
        register_in_registry(
            name=args.name,
            path=str(best_pt.relative_to(REPO_ROOT)),
            metrics=metrics,
            args=args,
            notes=f"batch={args.batch}, lr={args.lr}, epochs={args.epochs}, imgsz={args.imgsz}",
        )

    if args.deploy and best_pt.exists():
        deploy_path = DEPLOY_DIR / f"{args.name}.pt"
        shutil.copy2(best_pt, deploy_path)
        print(f"[DEPLOY] Copied to {deploy_path}")


if __name__ == "__main__":
    main()
