#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv8 分割训练脚本（仿照 v11 版本）。

详细使用说明：
1) 基础训练（使用预训练权重）：
   python segment/train.py --data coco-seg.yaml --weights yolov8n-seg.pt --epochs 100 --batch-size 16 --imgsz 640
2) 从模型配置文件开始训练：
   python segment/train.py --data coco-seg.yaml --cfg ultralytics/cfg/models/v8/yolov8-seg.yaml --epochs 100
3) 指定显卡与输出目录：
   python segment/train.py --data custom-seg.yaml --weights yolov8s-seg.pt --device 0 --project runs/train-seg --name exp_v8_seg
4) 断点续训：
   python segment/train.py --resume
5) 常用参数：
   --data        分割数据集配置文件（必填）
   --weights     预训练权重路径（与 --cfg 二选一）
   --cfg         模型结构配置文件（与 --weights 二选一）
   --epochs      训练轮数
   --batch-size  批大小
   --imgsz       输入图像尺寸
   --overlap-mask 训练时是否允许 mask 重叠
   --mask-ratio   mask 下采样比例
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import ultralytics
FILE = Path(__file__).resolve()
ROOT = FILE.parents[5]  # repository root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # add ROOT to PATH

from ultralytics import YOLO
from ultralytics.models.yolo._script_utils import resolve_project
from ultralytics.utils import LOGGER


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default="", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default="", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch-size", "--batch", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume from last checkpoint")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable autoanchor check")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="--cache images for faster training (ram/disk)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="runs/train-seg", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine learning rate scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP parameter, do not modify")
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box eval interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="version of dataset artifact to use")

    # YOLOv8 segmentation specific arguments
    parser.add_argument("--task", type=str, default="segment", help="task type")
    parser.add_argument("--overlap-mask", action="store_true", help="masks should overlap during training")
    parser.add_argument("--mask-ratio", type=int, default=4, help="mask downsample ratio")

    return parser.parse_args()


def main(opt):
    """Main training function."""
    LOGGER.info(f"Starting YOLOv8 segmentation training with arguments: {opt}")

    # Initialize model
    if opt.cfg and opt.weights:
        model = YOLO(opt.cfg).load(opt.weights)
        LOGGER.info(f"Created segmentation model from config {opt.cfg} and loaded weights: {opt.weights}")
    elif opt.cfg:
        model = YOLO(opt.cfg)
        LOGGER.info(f"Created new segmentation model from config: {opt.cfg}")
    elif opt.weights:
        model = YOLO(opt.weights)
        LOGGER.info(f"Loaded pretrained segmentation model: {opt.weights}")
    else:
        model = YOLO("yolov8n-seg.pt")
        LOGGER.info("Using default YOLOv8n-seg model")

    train_args = {
        "data": opt.data,
        "epochs": opt.epochs,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "device": opt.device,
        "workers": opt.workers,
        "project": resolve_project(opt.project),
        "name": opt.name,
        "exist_ok": opt.exist_ok,
        "pretrained": True,
        "optimizer": opt.optimizer,
        "verbose": True,
        "seed": 0,
        "deterministic": True,
        "single_cls": opt.single_cls,
        "rect": opt.rect,
        "cos_lr": opt.cos_lr,
        "close_mosaic": 10,
        "resume": opt.resume,
        "amp": True,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "label_smoothing": opt.label_smoothing,
        "nbs": 64,
        "overlap_mask": opt.overlap_mask,
        "mask_ratio": opt.mask_ratio,
        "dropout": 0.0,
        "val": True,
        "plots": not opt.noplots,
        "save": not opt.nosave,
        "save_period": opt.save_period,
        "cache": opt.cache,
        "patience": opt.patience,
    }

    if opt.hyp:
        train_args["hyp"] = opt.hyp

    try:
        results = model.train(**train_args)
        LOGGER.info(f"Segmentation training completed successfully. Results saved to {model.trainer.save_dir}")
        return results
    except Exception as e:
        LOGGER.error(f"Segmentation training failed with error: {e}")
        raise


if __name__ == "__main__":
    options = parse_opt()
    main(options)
