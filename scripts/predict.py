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
    parser = argparse.ArgumentParser(description="Run inference with a SWUAV-DANet checkpoint.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained weights (.pt).")
    parser.add_argument("--source", type=str, required=True, help="Image/video/glob or directory to run inference on.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS.")
    parser.add_argument("--device", type=str, default="auto", help="Execution device.")
    parser.add_argument("--project", type=Path, default=Path("runs/predict"), help="Output directory.")
    parser.add_argument("--name", type=str, default="SWUAV-DANet-predict", help="Run name.")
    parser.add_argument("--save-txt", action="store_true", help="Save results as *.txt.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences in *.txt.")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped image patches of detections.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Video frame stride.")
    parser.add_argument("--stream", action="store_true", help="Enable Python generator streaming predictions.")
    parser.add_argument("--show", action="store_true", help="Display results in a window.")
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
        save=args.show is False,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        vid_stride=args.vid_stride,
        stream=args.stream,
        show=args.show,
    )
    if args.stream:
        for r in results:
            print(r)
    else:
        print(results)


if __name__ == "__main__":
    main()
