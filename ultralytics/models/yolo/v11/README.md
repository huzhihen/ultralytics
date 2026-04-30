# YOLOv11 使用指南

本目录包含了YOLOv11模型的训练、验证和预测脚本，支持检测和分割任务。

## 文件结构

```
models/yolo/v11/
├── __init__.py              # 检测模块初始化
├── train.py                 # 检测训练脚本
├── val.py                   # 检测验证脚本
├── detect.py                # 检测预测脚本
├── segment/                 # 分割模块
│   ├── __init__.py          # 分割模块初始化
│   ├── train.py             # 分割训练脚本
│   ├── val.py               # 分割验证脚本
│   └── predict.py           # 分割预测脚本
└── README.md                # 本说明文档
```

## 使用方法

### 检测任务

#### 训练检测模型
```bash
# 使用配置文件训练
python models/yolo/v11/train.py --img 960 --batch 8 --epochs 80 \
    --cfg cfg/models/11/yolo11.yaml \
    --data data/coco128.yaml \
    --hyp data/hyps/hyp.scratch-high.yaml \
    --project runs/train \
    --name yolov11_detect \
    --weights yolo11n.pt --cache

# 使用预训练权重继续训练
python models/yolo/v11/train.py --img 960 --batch 8 --epochs 80 \
    --data data/coco128.yaml \
    --project runs/train \
    --name yolov11_detect_continue \
    --weights runs/train/yolov11_detect/weights/best.pt
```

#### 验证检测模型
```bash
python models/yolo/v11/val.py --img 960 --batch 8 \
    --weights runs/train/yolov11_detect/weights/best.pt \
    --data data/coco128.yaml \
    --project runs/val \
    --name yolov11_detect
```

#### 检测预测
```bash
python models/yolo/v11/detect.py \
    --weights runs/train/yolov11_detect/weights/best.pt \
    --data data/coco128.yaml \
    --source /path/to/images \
    --project runs/predict \
    --name yolov11_detect
```

### 分割任务

#### 训练分割模型
```bash
# 使用配置文件训练
python models/yolo/v11/segment/train.py --img 960 --batch 8 --epochs 80 \
    --cfg cfg/models/11/yolo11-seg.yaml \
    --data data/coco128-seg.yaml \
    --hyp data/hyps/hyp.scratch-high.yaml \
    --project runs/train-seg \
    --name yolov11_seg \
    --weights yolo11n-seg.pt --cache

# 使用预训练权重继续训练
python models/yolo/v11/segment/train.py --img 960 --batch 8 --epochs 80 \
    --data data/coco128-seg.yaml \
    --project runs/train-seg \
    --name yolov11_seg_continue \
    --weights runs/train-seg/yolov11_seg/weights/best.pt
```

#### 验证分割模型
```bash
python models/yolo/v11/segment/val.py --img 960 --batch 8 \
    --weights runs/train-seg/yolov11_seg/weights/best.pt \
    --data data/coco128-seg.yaml \
    --project runs/val-seg \
    --name yolov11_seg
```

#### 分割预测
```bash
python models/yolo/v11/segment/predict.py \
    --weights runs/train-seg/yolov11_seg/weights/best.pt \
    --data data/coco128-seg.yaml \
    --source /path/to/images \
    --project runs/predict-seg \
    --name yolov11_seg
```

## 主要参数说明

### 通用参数
- `--weights`: 模型权重文件路径
- `--cfg`: 模型配置文件路径
- `--data`: 数据集配置文件路径
- `--hyp`: 超参数配置文件路径
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--img`: 输入图像尺寸
- `--device`: 设备选择 (GPU/CPU)
- `--project`: 项目保存目录
- `--name`: 实验名称
- `--exist-ok`: 允许覆盖现有实验目录

### 检测特定参数
- `--conf-thres`: 置信度阈值
- `--iou-thres`: NMS IoU阈值
- `--max-det`: 每张图像最大检测数量

### 分割特定参数
- `--overlap-mask`: 训练时掩码是否重叠
- `--mask-ratio`: 掩码下采样比例
- `--retina-masks`: 使用高分辨率掩码

## 注意事项

1. 确保数据集配置文件格式正确
2. 训练前检查GPU内存是否足够
3. 可以根据需要调整超参数
4. 建议使用预训练权重进行迁移学习
5. 分割任务需要相应的分割标注数据

## 模型配置

YOLOv11的模型配置文件位于：
- 检测模型: `cfg/models/11/yolo11.yaml`
- 分割模型: `cfg/models/11/yolo11-seg.yaml`

支持的不同尺寸模型：
- `yolo11n`: 最小模型 (最快)
- `yolo11s`: 小模型
- `yolo11m`: 中等模型
- `yolo11l`: 大模型
- `yolo11x`: 超大模型 (最准确) 