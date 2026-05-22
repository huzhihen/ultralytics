#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv5 检测预测脚本。

详细使用说明：
1. 摄像头实时检测：
   python ultralytics/models/yolo/v5/detect.py --weights yolov5nu.pt --source 0
2. 单张图片检测：
   python ultralytics/models/yolo/v5/detect.py --weights yolov5nu.pt --source img.jpg
3. 视频检测：
   python ultralytics/models/yolo/v5/detect.py --weights yolov5nu.pt --source video.mp4
4. 文件夹或通配符检测：
   python ultralytics/models/yolo/v5/detect.py --weights yolov5nu.pt --source path/to/images
   python ultralytics/models/yolo/v5/detect.py --weights yolov5nu.pt --source "path/to/images/*.jpg"
5. 网络流检测：
   python ultralytics/models/yolo/v5/detect.py --weights yolov5nu.pt --source "rtsp://example.com/media.mp4"
6. 常用参数：
   --imgsz       推理尺寸，默认 640
   --conf-thres  置信度阈值，默认 0.25
   --iou-thres   NMS IoU 阈值，默认 0.45
   --device      运行设备，例如 cpu、0、0,1
   --save-txt    保存 YOLO txt 标签
   --nosave      不保存可视化图片或视频
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
    parser.add_argument("--weights", nargs="+", type=str, default="yolov5nu.pt", help="model path or triton URL")
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
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


def main(opt):
    """Run YOLOv5 detection prediction."""
    LOGGER.info(f"Starting YOLOv5 detection with arguments: {opt}")

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
