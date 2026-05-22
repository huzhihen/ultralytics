#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv5 分割预测脚本。

详细使用说明：
1. 摄像头实时分割：
   python ultralytics/models/yolo/v5/segment/predict.py --weights yolov5nu-seg.pt --source 0
2. 单张图片分割：
   python ultralytics/models/yolo/v5/segment/predict.py --weights yolov5nu-seg.pt --source img.jpg
3. 使用自定义训练权重预测：
   python ultralytics/models/yolo/v5/segment/predict.py --weights runs/train-seg/exp/weights/best.pt --source img.jpg
4. 文件夹或通配符分割：
   python ultralytics/models/yolo/v5/segment/predict.py --weights yolov5nu-seg.pt --source path/to/images
   python ultralytics/models/yolo/v5/segment/predict.py --weights yolov5nu-seg.pt --source "path/to/images/*.jpg"
5. 高分辨率掩码输出：
   python ultralytics/models/yolo/v5/segment/predict.py --weights yolov5nu-seg.pt --source img.jpg --retina-masks
6. 常用参数：
   --weights       分割模型权重路径
   --source        输入源，支持图片、视频、文件夹、通配符、摄像头和网络流
   --imgsz         推理尺寸，默认 640
   --conf-thres    置信度阈值，默认 0.25
   --retina-masks  使用高分辨率掩码
   --save-txt      保存预测标签
"""

import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[5]  # repository root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo._script_utils import resolve_project
from ultralytics.utils import LOGGER


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov5nu-seg.pt", help="model path or triton URL")
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
    parser.add_argument("--project", default="runs/predict-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--retina-masks", action="store_true", help="use high-resolution masks")
    parser.add_argument("--overlap-mask", action="store_true", help="masks should overlap during prediction")

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


def main(opt):
    """Run YOLOv5 segmentation prediction."""
    LOGGER.info(f"Starting YOLOv5 segmentation prediction with arguments: {opt}")

    model = YOLO(opt.weights[0])
    LOGGER.info(f"Loaded segmentation model: {opt.weights[0]}")

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
        "retina_masks": opt.retina_masks,
        "overlap_mask": opt.overlap_mask,
    }

    try:
        results = model.predict(**predict_args)
        LOGGER.info(f"Segmentation prediction completed successfully. Results saved to {opt.project}/{opt.name}")
        return results
    except Exception as e:
        LOGGER.error(f"Segmentation prediction failed with error: {e}")
        raise


if __name__ == "__main__":
    options = parse_opt()
    main(options)
