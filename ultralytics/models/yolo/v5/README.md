# YOLOv5 使用指南

本目录为 YOLOv5 提供独立的命令行脚本，结构参考 `ultralytics/models/yolo/11`，包含检测和分割任务的训练、验证、预测与导出入口。

## 文件结构

```text
ultralytics/models/yolo/v5/
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
# 使用 YOLOv5u 预训练权重训练
python ultralytics/models/yolo/v5/train.py --img 640 --batch-size 16 --epochs 100 \
    --weights yolov5nu.pt \
    --data coco128.yaml \
    --project runs/train \
    --name yolov5_detect

# 使用配置文件从头训练
python ultralytics/models/yolo/v5/train.py --img 640 --batch-size 16 --epochs 100 \
    --cfg ultralytics/cfg/models/v5/yolov5.yaml \
    --data coco128.yaml \
    --project runs/train \
    --name yolov5_detect_scratch
```

### 验证检测模型

```bash
python ultralytics/models/yolo/v5/val.py --img 640 --batch-size 32 \
    --weights runs/train/yolov5_detect/weights/best.pt \
    --data coco128.yaml \
    --project runs/val \
    --name yolov5_detect
```

### 检测预测

```bash
python ultralytics/models/yolo/v5/detect.py \
    --weights runs/train/yolov5_detect/weights/best.pt \
    --source /path/to/images \
    --project runs/predict \
    --name yolov5_detect
```

### 导出检测模型

```bash
python ultralytics/models/yolo/v5/export.py \
    --weights runs/train/yolov5_detect/weights/best.pt \
    --format onnx \
    --imgsz 640 \
    --dynamic \
    --simplify
```

## 分割任务

当前仓库内置的 YOLOv5 YAML 主要是检测配置：`ultralytics/cfg/models/v5/yolov5.yaml` 和 `yolov5-p6.yaml`。分割脚本已按 `11/segment` 的结构补齐，使用时请传入可用的 YOLOv5 分割权重或自定义分割 YAML。

### 训练分割模型

```bash
# 使用可用的分割权重训练
python ultralytics/models/yolo/v5/segment/train.py --img 640 --batch-size 16 --epochs 100 \
    --weights yolov5nu-seg.pt \
    --data coco128-seg.yaml \
    --project runs/train-seg \
    --name yolov5_seg

# 使用自定义分割配置文件从头训练
python ultralytics/models/yolo/v5/segment/train.py --img 640 --batch-size 16 --epochs 100 \
    --cfg /path/to/yolov5-seg.yaml \
    --data coco128-seg.yaml \
    --project runs/train-seg \
    --name yolov5_seg_scratch
```

### 验证分割模型

```bash
python ultralytics/models/yolo/v5/segment/val.py --img 640 --batch-size 32 \
    --weights runs/train-seg/yolov5_seg/weights/best.pt \
    --data coco128-seg.yaml \
    --project runs/val-seg \
    --name yolov5_seg
```

### 分割预测

```bash
python ultralytics/models/yolo/v5/segment/predict.py \
    --weights runs/train-seg/yolov5_seg/weights/best.pt \
    --source /path/to/images \
    --retina-masks \
    --project runs/predict-seg \
    --name yolov5_seg
```

### 导出分割模型

```bash
python ultralytics/models/yolo/v5/segment/export.py \
    --weights runs/train-seg/yolov5_seg/weights/best.pt \
    --format engine \
    --imgsz 640 \
    --device 0 \
    --half
```

## 主要参数说明

- `--weights`: 模型权重文件路径，例如 `yolov5nu.pt` 或训练输出的 `best.pt`。
- `--cfg`: 模型配置文件路径，例如 `ultralytics/cfg/models/v5/yolov5.yaml`。
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

YOLOv5 的内置模型配置文件位于 `ultralytics/cfg/models/v5/`：

- `yolov5.yaml`: P3-P5 检测模型
- `yolov5-p6.yaml`: P3-P6 检测模型

推荐检测权重使用 YOLOv5u 命名，例如 `yolov5nu.pt`、`yolov5su.pt`、`yolov5mu.pt`、`yolov5lu.pt`、`yolov5xu.pt`。旧权重名如 `yolov5n.pt` 会由项目工具提示替换为 `yolov5nu.pt`。
