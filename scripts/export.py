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

EXPORT_FORMATS = (
    "onnx",
    "torchscript",
    "engine",
    "openvino",
    "coreml",
    "tflite",
    "edgetpu",
    "tfjs",
    "paddle",
    "ncnn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SWUAV-DANet checkpoints to various backends.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained weights (.pt).")
    parser.add_argument("--format", type=str, default="onnx", choices=EXPORT_FORMATS, help="Export backend format.")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640], help="Image size (square or hw).")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export.")
    parser.add_argument("--half", action="store_true", help="Export half precision model if supported.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes if supported.")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph (requires onnx-simplifier).")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version override.")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization (if backend supports).")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace size (GB) when exporting engine.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    imgsz = args.imgsz if len(args.imgsz) > 1 else args.imgsz[0]
    export_kwargs = dict(
        format=args.format,
        imgsz=imgsz,
        device=args.device,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        int8=args.int8,
        workspace=args.workspace,
    )
    if args.opset is not None:
        export_kwargs["opset"] = args.opset

    artifact = model.export(**export_kwargs)
    print(f"Exported artifact: {artifact}")


if __name__ == "__main__":
    main()
