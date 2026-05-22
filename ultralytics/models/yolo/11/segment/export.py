#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Export YOLOv11 segmentation model to TensorRT engine format.

Usage:
    $ python ultralytics/models/yolo/v11/segment/export.py --weights yolo11n-seg.pt --format engine --imgsz 640
    $ python ultralytics/models/yolo/v11/segment/export.py --weights yolo11s-seg.pt --format engine --imgsz 640 --half
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import ultralytics
FILE = Path(__file__).resolve()
ROOT = FILE.parents[5]  # repository root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # add ROOT to PATH

from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo11n-seg.pt', help='model weights path')
    parser.add_argument('--format', type=str, default='engine', help='export format (engine, onnx, torchscript)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='image size (pixels)')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dynamic', action='store_true', help='dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='simplify model')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--workspace', type=float, default=4.0, help='TensorRT workspace size (GB)')
    parser.add_argument('--nms', action='store_true', help='add NMS to model')
    parser.add_argument('--int8', action='store_true', help='INT8 quantization')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    
    opt = parser.parse_args()
    return opt


def main(opt):
    """Main export function."""
    LOGGER.info(f"Starting YOLOv11 segmentation model export with arguments: {opt}")
    
    # Initialize model
    model = YOLO(opt.weights)
    LOGGER.info(f"Loaded segmentation model: {opt.weights}")
    
    # Prepare export arguments
    export_args = {
        'format': opt.format,
        'imgsz': opt.imgsz,
        'batch': opt.batch,
        'device': opt.device,
        'half': opt.half,
        'dynamic': opt.dynamic,
        'simplify': opt.simplify,
        'opset': opt.opset,
        'workspace': opt.workspace,
        'nms': opt.nms,
        'int8': opt.int8,
        'verbose': opt.verbose,
    }
    
    # Start export
    try:
        results = model.export(**export_args)
        LOGGER.info(f"Segmentation model export completed successfully.")
        LOGGER.info(f"Exported model saved to: {results}")
        return results
    except Exception as e:
        LOGGER.error(f"Segmentation model export failed with error: {e}")
        raise


if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 