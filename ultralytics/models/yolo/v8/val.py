#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv8 检测验证脚本（仿照 v11 版本）。

详细使用说明：
1) 使用官方权重验证：
   python val.py --weights yolov8n.pt --data coco.yaml --imgsz 640
2) 使用训练产出的 best.pt 验证：
   python val.py --weights runs/train/exp/weights/best.pt --data custom.yaml
3) 常用参数：
   --batch-size  验证批大小（默认 32）
   --conf-thres  置信度阈值（默认 0.001）
   --iou-thres   NMS IoU 阈值（默认 0.6）
   --save-json   导出 COCO JSON 结果
   --plots       保存验证可视化图
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path so we can import ultralytics
FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]  # repository root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # add ROOT to PATH

from ultralytics import YOLO
from ultralytics.models.yolo._script_utils import resolve_project
from ultralytics.utils import LOGGER


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov8n.pt", help="model.pt path(s)")
    parser.add_argument("--data", type=str, default="coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--batch-size", "--batch", type=int, default=32, help="batch size")
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
    parser.add_argument("--project", default="runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--plots", action="store_true", help="save plots for train/val")
    return parser.parse_args()


def main(opt):
    """Main validation function."""
    LOGGER.info(f"Starting YOLOv8 validation with arguments: {opt}")

    model = YOLO(opt.weights[0])
    LOGGER.info(f"Loaded model: {opt.weights[0]}")

    val_args = {
        "data": opt.data,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "device": opt.device,
        "workers": opt.workers,
        "project": resolve_project(opt.project),
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
    }

    try:
        results = model.val(**val_args)
        save_dir = getattr(results, "save_dir", None) or f"{opt.project}/{opt.name}"
        LOGGER.info(f"Validation completed successfully. Results saved to {save_dir}")
        return results
    except Exception as e:
        LOGGER.error(f"Validation failed with error: {e}")
        raise


if __name__ == "__main__":
    options = parse_opt()
    main(options)
