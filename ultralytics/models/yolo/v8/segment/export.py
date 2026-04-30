#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv8 分割模型导出脚本（仿照 v11 版本）。

详细使用说明：
1) 导出 TensorRT engine：
   python ultralytics/models/yolo/v8/segment/export.py --weights yolov8n-seg.pt --format engine --imgsz 640
2) 导出 ONNX：
   python ultralytics/models/yolo/v8/segment/export.py --weights yolov8s-seg.pt --format onnx --imgsz 640 --dynamic --simplify
3) 常用参数：
   --format     导出格式（engine/onnx/torchscript）
   --half       半精度导出
   --int8       INT8 量化
   --workspace  TensorRT 工作空间大小（GB）
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
    parser.add_argument("--weights", type=str, default="yolov8n-seg.pt", help="model weights path")
    parser.add_argument("--format", type=str, default="engine", help="export format (engine, onnx, torchscript)")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="image size (pixels)")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dynamic", action="store_true", help="dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="simplify model")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="add NMS to model")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    return parser.parse_args()


def main(opt):
    """Main export function."""
    LOGGER.info(f"Starting YOLOv8 segmentation model export with arguments: {opt}")

    model = YOLO(opt.weights)
    LOGGER.info(f"Loaded segmentation model: {opt.weights}")

    export_args = {
        "format": opt.format,
        "imgsz": opt.imgsz,
        "batch": opt.batch,
        "device": opt.device,
        "half": opt.half,
        "dynamic": opt.dynamic,
        "simplify": opt.simplify,
        "opset": opt.opset,
        "workspace": opt.workspace,
        "nms": opt.nms,
        "int8": opt.int8,
        "verbose": opt.verbose,
    }

    try:
        results = model.export(**export_args)
        LOGGER.info("Segmentation model export completed successfully.")
        LOGGER.info(f"Exported model saved to: {results}")
        return results
    except Exception as e:
        LOGGER.error(f"Segmentation model export failed with error: {e}")
        raise


if __name__ == "__main__":
    options = parse_opt()
    main(options)
