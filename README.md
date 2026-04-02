# SWUAV-DANet

SWUAV-DANet is a severe-weather UAV vehicle detection project, including:
- SWUAV dataset support
- Dynamic AlignAir Network (DANet)
- Reproducible training, evaluation, inference, export, and visualization scripts

## Repository Layout

| Path | Description |
| --- | --- |
| `configs/DANet.yaml` | DANet model configuration |
| `swuav_dan/` | CACT/DAAH modules and runtime registry |
| `scripts/` | Entry scripts for train/test/predict/export/visualize |
| `requirements.txt` | Python dependencies |
| `LICENSE`, `CONTRIBUTING.md`, `CHANGELOG.md` | Project metadata |

## Installation

```bash
git clone <repo-url> SWUAV-DANet
cd SWUAV-DANet
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Training and Evaluation

### Train

```bash
python scripts/train.py \
  --data path/to/data.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 640
```

Default training settings:
- Optimizer: `SGD`
- `lr0=0.001`, `momentum=0.937`, `weight_decay=0.0005`
- `epochs=200`, `batch=16`, `imgsz=640`

Common options:
- `--weights path/to/weights.pt`
- `--project` / `--name`
- `--resume`
- `--no-amp`
- `--device 0`

### Validate / Test

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

Supported formats include `onnx`, `torchscript`, `engine`, and `openvino`. Common switches: `--half`, `--dynamic`, `--simplify`.

## Environment Notes

- CUDA `11.8` or `12.1` is recommended (aligned with PyTorch 2.2.x/2.3.x).
- A single 12 GB GPU can typically run `batch=16`, `imgsz=640`.
- If memory is tight, reduce `--batch` or `--imgsz`, or use `--no-amp`.

## Dataset Access

To request the SWUAV dataset, fill the table below and email it to `chien_abyss@hainanu.edu.cn`. Download instructions will be provided by reply.

| Name | Institution | Email | Intended use | Public/Internal |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Dataset Layout Example

```text
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
    data.yaml
```

Label format follows YOLO TXT: `class x_center y_center width height` (normalized to `[0,1]`).
Class order should match `nc` and `names`, e.g. `['car', 'truck', 'bus', 'van', 'freight car']`.

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

| Model | Split | Epoch | AP (test) | Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| SWUAV-DANet | SWUAV test | 200 | 46.9 | [configs/DANet.yaml](configs/DANet.yaml) | [Baidu](https://pan.baidu.com/s/1IqAYTN8bfNN6ak7oqD-1qg?pwd=yupp) (pwd: `yupp`) |

## Visualization

```bash
python scripts/visualize.py \
  --weights path/to/best.pt \
  --source path/to/images_or_dir \
  --imgsz 640 \
  --conf 0.25
```

Outputs are saved to `runs/visualize/SWUAV-DANet-vis`. Add `--show` for on-screen preview.

## Citation

If this project helps your research, please cite:

```bibtex
@article{zhang2025swuavdanet,
  title   = {SWUAV-DANet: A Severe-Weather UAV Dataset and Dynamic AlignAir Network for Robust Aerial Vehicle Detection},
  author  = {Zhang, Longze and Guo, Keying},
  journal = {arXiv preprint},
  year    = {2025}
}
```

## Acknowledgements

- Built on Ultralytics YOLO.
- Thanks to the open-source community (YOLO series, DETR series, AOD-Net, TransWeather, etc.) for foundational ideas and baselines.
