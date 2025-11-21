# SWUAV-DANet

SWUAV-DANet: A Severe-Weather UAV Dataset and Dynamic AlignAir Network for Robust Aerial Vehicle Detection.

## Repository Structure

- `configs/DANet.yaml`: DANet model YAML 
- `swuav_dan/modules/c2act.py`: C2ACT implementation.
- `swuav_dan/heads/daah.py`: Detect_DAAH implementation.
- `swuav_dan/registry.py`: register.
- `requirements.txt`
- `scripts/`: train/val/predict/export helpers.
- `LICENSE` / `CONTRIBUTING.md` / `CHANGELOG.md`: project meta.

## Installation & Training
```bash
git clone <repo-url> SWUAV-DANet
cd SWUAV-DANet
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Training
```bash
python scripts/train.py \
  --data path/to/data.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 640
```
Defaults:
- Optimizer SGD, `lr0=0.001`, momentum `0.937`, weight decay `0.0005`
- Epochs `200`, batch `16`
- Images letterboxed to `640x640`
- Other hypers follow Ultralytics defaults

Common options:
- `--weights path/to/weights.pt` warm start
- `--project` / `--name` control outputs
- `--resume` continue from `project/name`

### Validation / Test
```bash
python scripts/test.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --data path/to/data.yaml \
  --split val
```
Optional: `--save-txt`, `--save-json`, `--no-plots`, etc.

### Inference
```bash
python scripts/predict.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --source path/to/images_or_video \
  --conf 0.25
```
Supports `--save-txt`, `--save-crop`, `--show`, and works on images/folders/videos/camera streams.

### Model Export
```bash
python scripts/export.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --format onnx \
  --imgsz 640
```
Formats: `onnx/torchscript/engine/openvino/...`; toggles: `--half`, `--dynamic`, `--simplify`, etc.
