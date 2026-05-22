# YOLO26 使用指南

本目录为 YOLO26 提供独立的命令行脚本，结构参考 `ultralytics/models/yolo/11`，支持检测和分割任务的训练、验证、预测与导出。

## 文件结构

```text
ultralytics/models/yolo/26/
├── train.py                 # 检测训练脚本
├── val.py                   # 检测验证脚本
├── detect.py                # 检测预测脚本
├── export.py                # 通用模型导出脚本
├── segment/
│   ├── train.py             # 分割训练脚本
│   ├── val.py               # 分割验证脚本
│   ├── predict.py           # 分割预测脚本
│   └── export.py            # 分割模型导出脚本
└── README.md                # 本说明文档
```

## 检测任务

### 训练检测模型

```bash
# 使用预训练权重训练
python ultralytics/models/yolo/26/train.py --img 640 --batch-size 16 --epochs 100 \
    --weights yolo26n.pt \
    --data coco128.yaml \
    --project runs/train \
    --name yolo26_detect

# 使用配置文件从头训练
python ultralytics/models/yolo/26/train.py --img 640 --batch-size 16 --epochs 100 \
    --cfg ultralytics/cfg/models/26/yolo26.yaml \
    --data coco128.yaml \
    --project runs/train \
    --name yolo26_detect_scratch
```

### 验证检测模型

```bash
python ultralytics/models/yolo/26/val.py --img 640 --batch-size 32 \
    --weights runs/train/yolo26_detect/weights/best.pt \
    --data coco128.yaml \
    --project runs/val \
    --name yolo26_detect
```

### 检测预测

```bash
python ultralytics/models/yolo/26/detect.py \
    --weights runs/train/yolo26_detect/weights/best.pt \
    --source /path/to/images \
    --project runs/predict \
    --name yolo26_detect
```

### 导出检测模型

```bash
python ultralytics/models/yolo/26/export.py \
    --weights runs/train/yolo26_detect/weights/best.pt \
    --format onnx \
    --imgsz 640 \
    --dynamic \
    --simplify
```

## 分割任务

### 训练分割模型

```bash
# 使用预训练分割权重训练
python ultralytics/models/yolo/26/segment/train.py --img 640 --batch-size 16 --epochs 100 \
    --weights yolo26n-seg.pt \
    --data coco128-seg.yaml \
    --project runs/train-seg \
    --name yolo26_seg

# 使用分割配置文件从头训练
python ultralytics/models/yolo/26/segment/train.py --img 640 --batch-size 16 --epochs 100 \
    --cfg ultralytics/cfg/models/26/yolo26-seg.yaml \
    --data coco128-seg.yaml \
    --project runs/train-seg \
    --name yolo26_seg_scratch
```

### 验证分割模型

```bash
python ultralytics/models/yolo/26/segment/val.py --img 640 --batch-size 32 \
    --weights runs/train-seg/yolo26_seg/weights/best.pt \
    --data coco128-seg.yaml \
    --project runs/val-seg \
    --name yolo26_seg
```

### 分割预测

```bash
python ultralytics/models/yolo/26/segment/predict.py \
    --weights runs/train-seg/yolo26_seg/weights/best.pt \
    --source /path/to/images \
    --retina-masks \
    --project runs/predict-seg \
    --name yolo26_seg
```

### 导出分割模型

```bash
python ultralytics/models/yolo/26/segment/export.py \
    --weights runs/train-seg/yolo26_seg/weights/best.pt \
    --format engine \
    --imgsz 640 \
    --device 0 \
    --half
```

## 主要参数说明

- `--weights`: 模型权重文件路径，例如 `yolo26n.pt`、`yolo26n-seg.pt` 或训练输出的 `best.pt`。
- `--cfg`: 模型配置文件路径，例如 `ultralytics/cfg/models/26/yolo26.yaml` 或 `ultralytics/cfg/models/26/yolo26-seg.yaml`。
- `--data`: 数据集配置文件路径，检测可用 `coco128.yaml`，分割可用 `coco128-seg.yaml`。
- `--epochs`: 训练轮数。
- `--batch-size`: 训练或验证批大小。
- `--imgsz`/`--img`: 输入图像尺寸。
- `--device`: 运行设备，例如 `cpu`、`0`、`0,1`。
- `--project` 和 `--name`: 输出目录。
- `--exist-ok`: 允许复用已有输出目录。
- `--conf-thres`: 预测或验证时的置信度阈值。
- `--iou-thres`: NMS IoU 阈值。
- `--retina-masks`: 分割预测时输出高分辨率掩码。
- `--overlap-mask`: 分割训练或验证时是否允许实例掩码重叠。
- `--mask-ratio`: 分割掩码下采样比例。

## 模型配置

YOLO26 的模型配置文件位于 `ultralytics/cfg/models/26/`：

- `yolo26.yaml`: 检测模型
- `yolo26-seg.yaml`: 分割模型
- `yolo26-pose.yaml`: 姿态模型
- `yolo26-obb.yaml`: 旋转框模型
- `yolo26-cls.yaml`: 分类模型
- `yolo26-p2.yaml`: P2 检测模型
- `yolo26-p6.yaml`: P6 检测模型

支持的常见权重命名包括 `yolo26n`、`yolo26s`、`yolo26m`、`yolo26l`、`yolo26x`，以及对应的 `-seg` 分割权重。
