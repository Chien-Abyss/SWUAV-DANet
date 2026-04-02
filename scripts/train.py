from __future__ import annotations

import argparse
from pathlib import Path
import sys
import site
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
for p in site.getsitepackages():
    if p not in sys.path:
        sys.path.append(p)

import swuav_dan
from ultralytics import YOLO
DEFAULT_CFG = ROOT / "configs" / "DANet.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SWUAV-DANet model with Ultralytics YOLO.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG, help="Model YAML to load.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional pretrained weights to warm start from.")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default per paper).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Letterboxed image size.")
    parser.add_argument("--device", type=str, default="auto", help="Device string accepted by Ultralytics (e.g. '0', 'cpu').")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer choice (defaults to SGD).")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate factor override.")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum.")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--project", type=Path, default=Path("runs/train"), help="Training output directory.")
    parser.add_argument("--name", type=str, default="SWUAV-DANet", help="Experiment name.")
    parser.add_argument("--resume", action="store_true", help="Resume from the most recent checkpoint in the project/name folder.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP training.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"
    model_source = args.weights if args.weights is not None else args.config
    model = YOLO(model_source, task="detect")
    model.overrides["pretrained"] = False

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "pretrained": False,
        "cache": True,
        "project": str(args.project),
        "name": args.name,
        "resume": args.resume,
        "patience": args.patience,
        "amp": not args.no_amp,
        "seed": args.seed,
    }
    train_kwargs["lr0"] = args.lr0
    train_kwargs["momentum"] = args.momentum
    train_kwargs["weight_decay"] = args.weight_decay
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf

    results = model.train(**train_kwargs)
    print(results)


if __name__ == "__main__":
    main()
