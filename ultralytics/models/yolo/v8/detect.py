#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv8 检测推理脚本（仿照 v11 版本）。

详细使用说明：
1) 摄像头实时推理：
   python detect.py --weights yolov8n.pt --source 0
2) 图片/视频推理：
   python detect.py --weights yolov8n.pt --source img.jpg
   python detect.py --weights yolov8n.pt --source video.mp4
3) 目录或通配符推理：
   python detect.py --weights yolov8n.pt --source path/
   python detect.py --weights yolov8n.pt --source "path/*.jpg"
4) 常用参数：
   --conf-thres  置信度阈值（默认 0.25）
   --iou-thres   NMS IoU 阈值（默认 0.45）
   --imgsz       推理尺寸（默认 640）
   --device      设备，如 0 / 0,1 / cpu
   --save-txt    保存标签到 txt
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
    parser.add_argument("--weights", nargs="+", type=str, default="yolov8n.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default="", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default="runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    """Main detection function."""
    LOGGER.info(f"Starting YOLOv8 detection with arguments: {opt}")

    model = YOLO(opt.weights[0])
    LOGGER.info(f"Loaded model: {opt.weights[0]}")

    predict_args = {
        "source": opt.source,
        "data": opt.data,
        "imgsz": opt.imgsz,
        "device": opt.device,
        "project": resolve_project(opt.project),
        "name": opt.name,
        "exist_ok": opt.exist_ok,
        "conf": opt.conf_thres,
        "iou": opt.iou_thres,
        "max_det": opt.max_det,
        "classes": opt.classes,
        "agnostic_nms": opt.agnostic_nms,
        "augment": opt.augment,
        "visualize": opt.visualize,
        "save_txt": opt.save_txt,
        "save_conf": opt.save_conf,
        "save_crop": opt.save_crop,
        "save": not opt.nosave,
        "half": opt.half,
        "dnn": opt.dnn,
        "vid_stride": opt.vid_stride,
        "line_width": opt.line_thickness,
        "show_labels": not opt.hide_labels,
        "show_conf": not opt.hide_conf,
    }

    try:
        results = model.predict(**predict_args)
        LOGGER.info(f"Detection completed successfully. Results saved to {opt.project}/{opt.name}")
        return results
    except Exception as e:
        LOGGER.error(f"Detection failed with error: {e}")
        raise


if __name__ == "__main__":
    options = parse_opt()
    main(options)
