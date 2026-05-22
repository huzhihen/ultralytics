#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv5 分割模型导出脚本，默认导出 TensorRT engine，也可导出 ONNX、TorchScript 等格式。

详细使用说明：
1. 导出分割模型到 TensorRT engine：
   python ultralytics/models/yolo/v5/segment/export.py --weights yolov5nu-seg.pt --format engine --imgsz 640 --device 0
2. 导出分割模型到 ONNX：
   python ultralytics/models/yolo/v5/segment/export.py --weights yolov5nu-seg.pt --format onnx --imgsz 640 --dynamic
3. 导出自定义训练权重：
   python ultralytics/models/yolo/v5/segment/export.py --weights runs/train-seg/exp/weights/best.pt --format engine
4. 半精度导出：
   python ultralytics/models/yolo/v5/segment/export.py --weights yolov5nu-seg.pt --format engine --device 0 --half
5. 常用参数：
   --weights   分割模型权重路径
   --format    导出格式，默认 engine
   --imgsz     导出输入尺寸，默认 640
   --batch     导出批大小，默认 1
   --device    运行设备，TensorRT 通常需要指定 GPU，例如 0
   --dynamic   是否导出动态输入维度
   --opset     ONNX opset 版本，默认 11
"""

import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[5]  # repository root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5nu-seg.pt", help="model weights path")
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
    """Run YOLOv5 segmentation model export."""
    LOGGER.info(f"Starting YOLOv5 segmentation model export with arguments: {opt}")

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
