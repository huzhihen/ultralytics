#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLO26 检测训练脚本。

详细使用说明：
1. 使用官方预训练权重训练检测模型：
   python ultralytics/models/yolo/26/train.py --weights yolo26n.pt --data coco128.yaml --epochs 100 --img 640
2. 使用模型配置从头训练：
   python ultralytics/models/yolo/26/train.py --cfg ultralytics/cfg/models/26/yolo26.yaml --data coco128.yaml
3. 从已有训练结果继续训练：
   python ultralytics/models/yolo/26/train.py --weights runs/train/exp/weights/best.pt --data coco128.yaml
4. 指定 GPU、批大小和输出目录：
   python ultralytics/models/yolo/26/train.py --weights yolo26n.pt --data coco128.yaml --device 0 --batch-size 16 \
       --project runs/train --name yolo26_detect
5. 常用参数：
   --weights     初始权重路径，不传时默认使用 yolo26n.pt
   --cfg         模型 YAML 配置路径，适合从头训练
   --data        数据集 YAML 配置路径
   --epochs      训练轮数，默认 100
   --batch-size  批大小，默认 16
   --imgsz       训练/验证图片尺寸，默认 640
   --resume      恢复训练
"""

import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]  # repository root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
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
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="--cache images for faster training (ram/disk)")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="runs/train", help="save to project/name")
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
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment"], help="task type")
    return parser.parse_args()


def main(opt):
    """Run YOLO26 detection training."""
    LOGGER.info(f"Starting YOLO26 training with arguments: {opt}")

    if opt.cfg:
        model = YOLO(opt.cfg)
        LOGGER.info(f"Created new model from config: {opt.cfg}")
        if opt.weights:
            model.load(opt.weights)
            LOGGER.info(f"Loaded initial weights: {opt.weights}")
    elif opt.weights:
        model = YOLO(opt.weights)
        LOGGER.info(f"Loaded pretrained model: {opt.weights}")
    else:
        model = YOLO("yolo26n.pt")
        LOGGER.info("Using default YOLO26n model")

    train_args = {
        "data": opt.data,
        "epochs": opt.epochs,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "device": opt.device,
        "workers": opt.workers,
        "project": opt.project,
        "name": opt.name,
        "exist_ok": opt.exist_ok,
        "pretrained": True,
        "optimizer": opt.optimizer,
        "verbose": True,
        "seed": 0,
        "deterministic": True,
        "single_cls": opt.single_cls,
        "image_weights": opt.image_weights,
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
        "fl_gamma": 0.0,
        "label_smoothing": opt.label_smoothing,
        "nbs": 64,
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.0,
        "val": not opt.noval,
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
        LOGGER.info(f"Training completed successfully. Results saved to {model.trainer.save_dir}")
        return results
    except Exception as e:
        LOGGER.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    options = parse_opt()
    main(options)
