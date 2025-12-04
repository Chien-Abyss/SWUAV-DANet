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

import swuav_dan  # noqa: F401  # register custom modules
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize inference results with SWUAV-DANet.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained weights (.pt).")
    parser.add_argument("--source", type=str, required=True, help="Image/video/glob or directory to run inference on.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS.")
    parser.add_argument("--device", type=str, default="auto", help="Execution device.")
    parser.add_argument("--project", type=Path, default=Path("runs/visualize"), help="Output directory.")
    parser.add_argument("--name", type=str, default="SWUAV-DANet-vis", help="Run name.")
    parser.add_argument("--show", action="store_true", help="Display results in a window.")
    parser.add_argument("--save-txt", action="store_true", help="Save detections to *.txt (YOLO format).")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped detections.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(args.project),
        name=args.name,
        save=True,
        show=args.show,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
    )
    if args.show:
        # When --show is on, stream prints minimal info to avoid clutter
        for r in results:
            print(r)
    else:
        print(f"Visualization saved to {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
