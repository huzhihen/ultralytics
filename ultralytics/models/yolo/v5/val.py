#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv5 检测验证脚本。

详细使用说明：
1. 验证 YOLOv5u 官方检测模型：
   python ultralytics/models/yolo/v5/val.py --weights yolov5nu.pt --data coco128.yaml --img 640
2. 验证训练后的最佳权重：
   python ultralytics/models/yolo/v5/val.py --weights runs/train/exp/weights/best.pt --data coco128.yaml
3. 指定批大小、设备和输出目录：
   python ultralytics/models/yolo/v5/val.py --weights yolov5nu.pt --data coco128.yaml --batch-size 32 --device 0 \
       --project runs/val --name yolov5_detect
4. 保存预测标签或 COCO JSON：
   python ultralytics/models/yolo/v5/val.py --weights yolov5nu.pt --data coco128.yaml --save-txt --save-json
5. 常用参数：
   --weights     模型权重路径
   --data        数据集 YAML 配置路径
   --batch-size  验证批大小，默认 32
   --imgsz       验证图片尺寸，默认 640
   --conf-thres  置信度阈值，默认 0.001
   --iou-thres   NMS IoU 阈值，默认 0.6
"""

import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]  # repository root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo._script_utils import resolve_project
from ultralytics.utils import LOGGER


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov5nu.pt", help="model.pt path(s)")
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
    """Run YOLOv5 detection validation."""
    LOGGER.info(f"Starting YOLOv5 validation with arguments: {opt}")

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
