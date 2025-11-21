# SWUAV-DANet

SWUAV-DANet 提供 DAN 模型中自定义的两个核心模块：骨干的 **C2ACT**（原 `C2TSSA_DYT`）与检测头 **Detect_DAAH**（原 `Detect_TADDH`）。

## 仓库结构

- `configs/DANet.yaml`：DAN 模型 YAML
- `swuav_dan/modules/c2act.py`：完整的 C2ACT实现。
- `swuav_dan/heads/daah.py`：完整的 Detect_DAAH 实现。
- `swuav_dan/registry.py`：注册。
- `requirements.txt`：依赖。
- `scripts/`：训练、验证、推理、导出工具脚本。
- `LICENSE` / `CONTRIBUTING.md` / `CHANGELOG.md`：基础工程文件。

## 安装与训练

```bash
git clone <repo-url> SWUAV-DANet
cd SWUAV-DANet
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 训练

```bash
python scripts/train.py \
  --data path/to/data.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 640
```

默认训练策略：
- 优化器：SGD，初始学习率 `0.001`，动量 `0.937`，权值衰减 `0.0005`
- 训练轮数 `200`，批大小 `16`
- 输入图像 letterbox 到 `640×640`
- 其余超参沿用 Ultralytics 默认设置  

常用可选项：
- `--weights path/to/weights.pt`：用已有权重热启动。
- `--project` / `--name`：指定输出目录。
- `--resume`：在 `project/name` 目录下继续最近一次训练。

### 验证/测试

```bash
python scripts/test.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --data path/to/data.yaml \
  --split val
```

可附加 `--save-txt`、`--save-json`、`--no-plots` 等参数按需输出。

### 推理

```bash
python scripts/predict.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --source path/to/images_or_video \
  --conf 0.25
```

支持 `--save-txt`、`--save-crop`、`--show` 等参数，可对图片、文件夹、视频或摄像头流批量推理。

### 模型导出

```bash
python scripts/export.py \
  --weights runs/train/SWUAV-DANet/weights/best.pt \
  --format onnx \
  --imgsz 640
```

`--format` 可选 `onnx/torchscript/engine/openvino/...`，并提供 `--half`、`--dynamic`、`--simplify` 等常用开关。

## 说明

- 由于只保留 DAN 相关模块，仓库体积显著减小，无关文件已移除。
- 其余训练/推理逻辑全部复用官方 Ultralytics，实现与原始工程一致。
- 如果需要在其他项目中复用模块，只需 `pip install -e .` 并在脚本开头 `import swuav_dan`。
