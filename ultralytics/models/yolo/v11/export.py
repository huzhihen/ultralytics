#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv11 模型导出脚本（默认导出 ONNX，支持 detect/segment/pose 等任务）。

详细使用说明：
1) 导出官方检测模型到 ONNX：
   python ultralytics/models/yolo/v11/export.py --weights yolo11n.pt --format onnx --imgsz 640
2) 导出你自己的模型到 ONNX：
   python ultralytics/models/yolo/v11/export.py --weights /path/to/best.pt --format onnx --imgsz 640
3) 导出动态尺寸 ONNX（推荐部署时使用）：
   python ultralytics/models/yolo/v11/export.py --weights /path/to/best.pt --dynamic --simplify
4) 指定设备和半精度：
   python ultralytics/models/yolo/v11/export.py --weights /path/to/best.pt --device 0 --half
5) 常用参数说明：
   --weights   模型权重路径（.pt）
   --format    导出格式，默认 onnx（也支持 engine/torchscript 等）
   --imgsz     导出输入尺寸，默认 640
   --dynamic   是否导出动态输入维度
   --simplify  是否简化 ONNX 图
   --opset     ONNX opset 版本，默认 12
"""

import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # ultralytics root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="model weights path (.pt)")
    parser.add_argument("--format", type=str, default="onnx", help="export format, e.g. onnx, engine, torchscript")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="image size (pixels)")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision export")
    parser.add_argument("--dynamic", action="store_true", help="export with dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="simplify ONNX graph")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="add NMS to exported model if supported")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    return parser.parse_args()


def main(opt):
    """Main export function."""
    LOGGER.info(f"Starting YOLOv11 export with arguments: {opt}")

    model = YOLO(opt.weights)
    LOGGER.info(f"Loaded model: {opt.weights}")

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
        result = model.export(**export_args)
        LOGGER.info("Model export completed successfully.")
        LOGGER.info(f"Exported file: {result}")
        return result
    except Exception as e:
        LOGGER.error(f"Model export failed with error: {e}")
        raise


if __name__ == "__main__":
    options = parse_opt()
    main(options)
