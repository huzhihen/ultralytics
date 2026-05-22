#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLO26 分割自动标注脚本，支持按 `--data` 中的类别过滤可视化和标注输出，并支持 JSON/TXT 两种格式。

详细使用说明：
1. 对图片目录自动生成分割标注：
   python ultralytics/models/yolo/26/autolabeling_segment.py \
       --weights runs/train-seg/yolo26_full_640/weights/best.pt \
       --source /path/to/images \
       --data /path/to/classes.yaml \
       --project runs/autolabel-seg \
       --name yolo26_full_640 \
       --format json txt

2. 只输出 LabelMe JSON：
   python ultralytics/models/yolo/26/autolabeling_segment.py \
       --weights yolo26n-seg.pt \
       --source /path/to/images \
       --data /path/to/classes.yaml \
       --format json

3. 只输出 YOLO TXT 分割标签：
   python ultralytics/models/yolo/26/autolabeling_segment.py \
       --weights yolo26n-seg.pt \
       --source /path/to/images \
       --data /path/to/classes.yaml \
       --format txt \
       --nosaveimage

4. 类别过滤规则：
   `--data` YAML 的 `names` 字段就是允许自动标注的类别列表。模型预测类别名不在该列表中时，
   不会可视化，也不会写入 JSON/TXT。TXT 类别 ID 会按 `--data.names` 顺序重新映射。

5. 输出目录：
   runs/autolabel-seg/<name>/images  保存可视化图
   runs/autolabel-seg/<name>/json    保存 LabelMe JSON
   runs/autolabel-seg/<name>/labels  保存 YOLO TXT
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]  # repository root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_autolabel = importlib.import_module("ultralytics.models.yolo.26.autolabeling_detection")


def parse_opt() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo26n-seg.pt", help="segmentation model weights path")
    parser.add_argument("--source", type=str, default="ultralytics/assets", help="image file/dir/glob")
    parser.add_argument("--data", type=str, required=True, help="dataset YAML with names to auto-label")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", default="runs/autolabel-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--format", nargs="+", default=["json", "txt"], help="output formats: json txt")
    parser.add_argument("--image-name", default="images", help="directory name for saving visualized images")
    parser.add_argument("--json-name", default="json", help="directory name for saving JSON labels")
    parser.add_argument("--txt-name", default="labels", help="directory name for saving TXT labels")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="annotation line thickness")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels on visualized images")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences on visualized images")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--retina-masks", action="store_true", help="use high-resolution masks")
    parser.add_argument("--save-conf", action="store_true", help="append confidence to TXT labels")
    parser.add_argument("--nosaveimage", action="store_true", help="do not save visualized images")
    parser.add_argument("--nosave-empty", action="store_true", help="do not save files for images without kept labels")
    opt = parser.parse_args()
    opt.formats = _autolabel.parse_formats(opt.format)
    return opt


def main(opt: argparse.Namespace) -> None:
    """Run segmentation auto-labeling."""
    _autolabel.run(
        weights=opt.weights,
        source=opt.source,
        data=opt.data,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        max_det=opt.max_det,
        device=opt.device,
        project=opt.project,
        name=opt.name,
        formats=opt.formats,
        image_name=opt.image_name,
        json_name=opt.json_name,
        txt_name=opt.txt_name,
        exist_ok=opt.exist_ok,
        line_thickness=opt.line_thickness,
        hide_labels=opt.hide_labels,
        hide_conf=opt.hide_conf,
        half=opt.half,
        retina_masks=opt.retina_masks,
        save_conf=opt.save_conf,
        nosaveimage=opt.nosaveimage,
        nosave_empty=opt.nosave_empty,
    )


if __name__ == "__main__":
    main(parse_opt())
