# SWUAV-DANet

SWUAV-DANet is a standalone implementation of DANet with two custom modules:

- `C2ACT` (renamed from `C2TSSA_DYT`)
- `Detect_DAAH` (renamed from `Detect_TADDH`)

The goal is model-level equivalence with `yolo11-C2TSSA-DYT-TADHH.yaml`, while keeping this repository independent.

## Repository Layout

- `configs/DANet.yaml`: model YAML that uses `C2ACT` and `Detect_DAAH`
- `swuav_dan/modules/c2act.py`: C2ACT wrapper
- `swuav_dan/heads/daah.py`: Detect_DAAH head
- `swuav_dan/vendor/extra_transformer.py`: AttentionTSSA + C2PSA/C2ACT support blocks
- `swuav_dan/vendor/extra_head.py`: DyDCNv2 + head helper blocks
- `swuav_dan/vendor/c3k2.py`: C3k2 implementation used by DANet YAML
- `swuav_dan/registry.py`: runtime registration and parser patch
- `scripts/`: train/val/predict/export/visualize entry scripts

## Installation

```bash
git clone <repo-url> SWUAV-DANet
cd SWUAV-DANet
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Notes:

- The package sets `YOLO_CONFIG_DIR` to `<repo>/.yolo_cfg` by default to avoid permission issues writing Ultralytics settings.
- `registry.py` can auto-detect a sibling `ultralytics-yolo11-main` folder if present.

## Training

```bash
python scripts/train.py \
  --data path/to/data.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 640
```

Defaults:

- Optimizer: SGD
- `lr0=0.01`, `momentum=0.937`, `weight_decay=0.0005`
- `epochs=200`, `batch=16`, `imgsz=640`

## Validation

```bash
python scripts/test.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --data path/to/data.yaml \
  --split val
```

## Inference

```bash
python scripts/predict.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --source path/to/images_or_video \
  --conf 0.25
```

## Export

```bash
python scripts/export.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --format onnx \
  --imgsz 640
```

## Dataset YAML Example

```yaml
path: data/SWUAV
train: images/train
val: images/val
test: images/test

nc: 5
names: [car, truck, bus, van, freight car]
```

## Model Zoo

| Model | Split | Epoch | AP (test) | Config | Download |
| --- | --- | --- | --- | --- | --- |
| SWUAV-DANet | SWUAV test | 200 | 46.9 | `configs/DANet.yaml` | [Baidu](https://pan.baidu.com/s/1IqAYTN8bfNN6ak7oqD-1qg?pwd=yupp) |

## Citation

```bibtex
@article{zhang2025swuavdanet,
  title   = {SWUAV-DANet: A Severe-Weather UAV Dataset and Dynamic AlignAir Network for Robust Aerial Vehicle Detection},
  author  = {Zhang, Longze and Guo, Keying},
  journal = {arXiv preprint},
  year    = {2025}
}
```
