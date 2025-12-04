# SWUAV-DANet

SWUAV-DANet: Severe-Weather UAV dataset and Dynamic AlignAir Network for robust aerial vehicle detection.

## Repository Structure
- `configs/DANet.yaml`: model YAML
- `swuav_dan/`: CACT/DAAH modules and registry
- `scripts/`: train / test / predict / export / visualize helpers
- `requirements.txt`
- `LICENSE`, `CONTRIBUTING.md`, `CHANGELOG.md`

## Installation & Training
```bash
git clone <repo-url> SWUAV-DANet
cd SWUAV-DANet
python -m venv .venv && .\.venv\Scripts\activate   # or use conda
pip install -r requirements.txt
pip install -e .
```

### Train
```bash
python scripts/train.py \
  --data path/to/data.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 640
```
Defaults: SGD (`lr0=0.001`, momentum `0.937`, weight decay `0.0005`), epochs 200, batch 16, imgsz 640. Common options: `--weights path/to/weights.pt`, `--project/--name`, `--resume`, `--no-amp`, `--device 0`.

### Validate / Test
```bash
python scripts/test.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --data path/to/data.yaml \
  --split val
```

### Inference
```bash
python scripts/predict.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --source path/to/images_or_video \
  --conf 0.25
```

### Export
```bash
python scripts/export.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --format onnx \
  --imgsz 640
```
Formats: onnx/torchscript/engine/openvino/...; toggles `--half`, `--dynamic`, `--simplify`.

## Environment & Hardware
- CUDA: 11.8 or 12.1 recommended (matches PyTorch 2.2.x/2.3.x). Use matching wheels if different.
- GPU memory: single 12GB can run `batch=16, imgsz=640`; if tight, lower `--batch`/`--imgsz` or add `--no-amp`.

## Dataset Access
To request the SWUAV dataset, fill the table below and email to `chien_abyss@hainanu.edu.cn`; we will reply with download instructions.

| Name | Institution | Email | Intended use | Public/Internal |
| --- | --- | --- | --- | --- |
|     |     |     |     |     |

### Dataset Layout Example
```
data/
  SWUAV/
    images/
      train/xxx.jpg
      val/xxx.jpg
      test/xxx.jpg
    labels/
      train/xxx.txt
      val/xxx.txt
      test/xxx.txt
    data.yaml   # can be here or parent; adjust paths accordingly
```
Labels use YOLO txt: `class x_center y_center width height` (normalized to [0,1]); class order should match `nc`/`names`, e.g. `['car','truck','bus','van','freight car']`.

Sample `data.yaml`:
```yaml
path: data/SWUAV
train: images/train
val: images/val
test: images/test

nc: 5
names: [car, truck, bus, van, freight car]
```

## Model Zoo
| Model | Split | Epoch | AP (paper) | Config | Download |
| :---: | :---: | :---: | :--------: | :----: | :------: |
| SWUAV-DANet | SWUAV test | 200 | 46.9 | [configs/DANet.yaml](configs/DANet.yaml) | [Baidu](https://pan.baidu.com/s/1IqAYTN8bfNN6ak7oqD-1qg?pwd=yupp) (pwd: `yupp`) |

## Visualization Demo
Run and save boxed results:
```bash
python scripts/visualize.py \
  --weights path/to/best.pt \
  --source path/to/images_or_dir \
  --imgsz 640 \
  --conf 0.25
```
Outputs in `runs/visualize/SWUAV-DANet-vis`; add `--show` to preview.

## Citation
If this repo or SWUAV-DANet helps your research, please cite:
```
@article{zhang2025swuavdanet,
  title   = {SWUAV-DANet: A Severe-Weather UAV Dataset and Dynamic AlignAir Network for Robust Aerial Vehicle Detection},
  author  = {Zhang, Longze and Guo, Keying},
  journal = {arXiv preprint},
  year    = {2025}
}
```

## Acknowledgements
- Built on Ultralytics YOLO; thanks for the training/inference/export toolchain.
- Thanks to the open-source community (YOLO series, DETR series, AOD-Net, TransWeather, etc.) for foundational ideas and baselines.
