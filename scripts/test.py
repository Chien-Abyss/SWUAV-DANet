from __future__ import annotations

import argparse
from pathlib import Path
import sys
import site

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
for p in site.getsitepackages():
    if p not in sys.path:
        sys.path.append(p)

import swuav_dan
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SWUAV-DANet checkpoint.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained weights (.pt).")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML used for evaluation.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate (val/test).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--device", type=str, default="auto", help="Device string accepted by Ultralytics.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for metrics.")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS.")
    parser.add_argument("--project", type=Path, default=Path("runs/val"), help="Evaluation output directory.")
    parser.add_argument("--name", type=str, default="SWUAV-DANet-val", help="Experiment name for evaluation.")
    parser.add_argument("--save-txt", action="store_true", help="Save predictions to *.txt files.")
    parser.add_argument("--save-json", action="store_true", help="Save predictions to COCO json.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    results = model.val(
        data=args.data,
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=str(args.project),
        name=args.name,
        save_txt=args.save_txt,
        save_json=args.save_json,
        plots=not args.no_plots,
    )
    print(results)


if __name__ == "__main__":
    main()
