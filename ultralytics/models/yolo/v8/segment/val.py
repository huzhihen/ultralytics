#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv8 分割验证脚本（仿照 v11 版本）。

详细使用说明：
1) 使用官方分割权重验证：
   python segment/val.py --weights yolov8n-seg.pt --data coco-seg.yaml --imgsz 640
2) 使用训练产出的 best.pt 验证：
   python segment/val.py --weights runs/train-seg/exp/weights/best.pt --data custom-seg.yaml
3) 常用参数：
   --overlap-mask  验证时 mask 是否重叠
   --mask-ratio    mask 下采样比例
   --save-json     导出 COCO JSON
   --plots         输出验证可视化图
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import ultralytics
FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]  # ultralytics root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov8n-seg.pt", help="model.pt path(s)")
    parser.add_argument("--data", type=str, default="coco128-seg.yaml", help="dataset.yaml path")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default="runs/val-seg", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--plots", action="store_true", help="save plots for train/val")

    # YOLOv8 segmentation specific arguments
    parser.add_argument("--overlap-mask", action="store_true", help="masks should overlap during validation")
    parser.add_argument("--mask-ratio", type=int, default=4, help="mask downsample ratio")
    return parser.parse_args()


def main(opt):
    """Main validation function."""
    LOGGER.info(f"Starting YOLOv8 segmentation validation with arguments: {opt}")

    model = YOLO(opt.weights[0])
    LOGGER.info(f"Loaded segmentation model: {opt.weights[0]}")

    val_args = {
        "data": opt.data,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "device": opt.device,
        "workers": opt.workers,
        "project": opt.project,
        "name": opt.name,
        "exist_ok": opt.exist_ok,
        "verbose": opt.verbose,
        "save_txt": opt.save_txt,
        "save_hybrid": opt.save_hybrid,
        "save_conf": opt.save_conf,
        "save_json": opt.save_json,
        "plots": opt.plots,
        "conf": opt.conf_thres,
        "iou": opt.iou_thres,
        "single_cls": opt.single_cls,
        "augment": opt.augment,
        "half": opt.half,
        "dnn": opt.dnn,
        "overlap_mask": opt.overlap_mask,
        "mask_ratio": opt.mask_ratio,
    }

    try:
        results = model.val(**val_args)
        LOGGER.info(f"Segmentation validation completed successfully. Results saved to {opt.project}/{opt.name}")
        return results
    except Exception as e:
        LOGGER.error(f"Segmentation validation failed with error: {e}")
        raise


if __name__ == "__main__":
    options = parse_opt()
    main(options)
