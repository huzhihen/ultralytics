#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLO26 自动标注脚本，支持按 `--data` 中的类别过滤可视化和标注输出，并支持 JSON/TXT 两种格式。

详细使用说明：
1. 对图片目录自动标注，同时输出 LabelMe JSON 和 YOLO TXT：
   python ultralytics/models/yolo/26/autolabeling_detection.py \
       --weights runs/train-seg/yolo26_full_640/weights/best.pt \
       --source /path/to/images \
       --data /path/to/classes.yaml \
       --project runs/autolabel \
       --name yolo26_full_640 \
       --format json txt

2. 只输出 JSON，不保存可视化图片：
   python ultralytics/models/yolo/26/autolabeling_detection.py \
       --weights /path/to/best.pt \
       --source /path/to/images \
       --data /path/to/classes.yaml \
       --format json \
       --nosaveimage

3. 只标注 `--data` 文件中声明的类别：
   - `--data` YAML 的 `names` 字段会作为允许输出的类别列表。
   - 模型预测类别名不在 `names` 中时，会被过滤掉，不会可视化，也不会写入 JSON/TXT。
   - TXT 中的类别 ID 会按 `--data.names` 的顺序重新映射，适合直接作为新数据集标签使用。

4. 支持的 `--data` 示例：
   names:
     0: person
     1: car
   或：
   names: [person, car]

5. 输出目录结构：
   runs/autolabel/yolo26_full_640/
   ├── images/    # 可视化图片
   ├── json/      # LabelMe JSON 标注
   └── labels/    # YOLO TXT 标注

6. TXT 格式：
   - 有分割 mask 时输出 YOLO segment 格式：
     class_id x1 y1 x2 y2 ... xn yn
   - 没有分割 mask 时输出 YOLO detect 格式：
     class_id x_center y_center width height
   - 坐标均为 0-1 归一化坐标。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]  # repository root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo._script_utils import resolve_project
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.files import increment_path


IMG_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}


def load_allowed_classes(data: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load class names from dataset YAML and return name-to-id and id-to-name mappings."""
    if not data:
        return {}, {}

    data_dict = YAML.load(data)
    names = data_dict.get("names", {})
    if isinstance(names, dict):
        id_to_name = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        id_to_name = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError(f"Invalid 'names' field in data YAML: {data}")

    name_to_id = {name: idx for idx, name in id_to_name.items()}
    return name_to_id, id_to_name


def normalize_model_names(names: Any) -> dict[int, str]:
    """Normalize model class names to an integer-keyed dictionary."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError(f"Unsupported model names type: {type(names).__name__}")


def is_image_result(path: str | Path) -> bool:
    """Return whether a prediction result path points to an image file."""
    return Path(path).suffix.lower() in IMG_SUFFIXES


def xyxy_to_yolo_line(class_id: int, xyxy: np.ndarray, width: int, height: int, conf: float | None = None) -> str:
    """Convert an xyxy box to a YOLO detection label line."""
    x1, y1, x2, y2 = xyxy.astype(float)
    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    values = [class_id, xc, yc, w, h]
    if conf is not None:
        values.append(conf)
    return " ".join([str(values[0]), *[f"{v:.6f}" for v in values[1:]]])


def polygon_to_yolo_line(class_id: int, polygon_xyn: np.ndarray, conf: float | None = None) -> str | None:
    """Convert a normalized polygon to a YOLO segmentation label line."""
    if polygon_xyn is None or len(polygon_xyn) < 3:
        return None

    coords = np.asarray(polygon_xyn, dtype=float).reshape(-1, 2)
    coords = np.clip(coords, 0.0, 1.0)
    flat = coords.reshape(-1)
    values = [class_id, *flat.tolist()]
    if conf is not None:
        values.append(conf)
    return " ".join([str(values[0]), *[f"{v:.6f}" for v in values[1:]]])


def make_labelme_json(image_path: Path, image_shape: tuple[int, int], shapes: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a LabelMe-compatible JSON payload."""
    height, width = image_shape
    return {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def parse_formats(values: list[str]) -> set[str]:
    """Parse and validate requested output formats."""
    formats = {v.lower() for value in values for v in value.replace(",", " ").split()}
    invalid = formats - {"json", "txt"}
    if invalid:
        raise ValueError(f"Unsupported output format(s): {sorted(invalid)}. Use json and/or txt.")
    return formats or {"json", "txt"}


def run(
    weights: str | Path = "yolo26n.pt",
    source: str | Path = "ultralytics/assets",
    data: str | Path = "",
    imgsz: int | tuple[int, int] = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 1000,
    device: str = "",
    project: str | Path = "runs/autolabel",
    name: str = "exp",
    formats: set[str] | None = None,
    image_name: str = "images",
    json_name: str = "json",
    txt_name: str = "labels",
    exist_ok: bool = False,
    line_thickness: int = 3,
    hide_labels: bool = False,
    hide_conf: bool = False,
    half: bool = False,
    retina_masks: bool = False,
    save_conf: bool = False,
    nosaveimage: bool = False,
    nosave_empty: bool = False,
) -> None:
    """Run YOLO26 auto-labeling with class filtering and JSON/TXT exports."""
    formats = formats or {"json", "txt"}
    allowed_name_to_id, allowed_id_to_name = load_allowed_classes(data)
    if not allowed_name_to_id:
        raise ValueError("--data must point to a dataset YAML with a non-empty 'names' field.")

    save_dir = increment_path(Path(resolve_project(str(project))) / name, exist_ok=exist_ok)
    image_dir = save_dir / image_name
    json_dir = save_dir / json_name
    txt_dir = save_dir / txt_name

    if not nosaveimage:
        image_dir.mkdir(parents=True, exist_ok=True)
    if "json" in formats:
        json_dir.mkdir(parents=True, exist_ok=True)
    if "txt" in formats:
        txt_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    model_names = normalize_model_names(model.names)

    LOGGER.info("YOLO26 自动标注")
    LOGGER.info("=" * 50)
    LOGGER.info(f"模型权重: {weights}")
    LOGGER.info(f"输入源: {source}")
    LOGGER.info(f"类别配置: {data}")
    LOGGER.info(f"允许类别: {', '.join(allowed_id_to_name.values())}")
    LOGGER.info(f"输出格式: {', '.join(sorted(formats))}")
    LOGGER.info(f"保存目录: {save_dir}")

    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        device=device,
        half=half,
        retina_masks=retina_masks,
        stream=True,
        save=False,
        verbose=False,
    )

    seen = 0
    saved_images = 0
    saved_json = 0
    saved_txt = 0
    skipped_by_class = 0
    total_kept = 0
    class_counts = {name: 0 for name in allowed_id_to_name.values()}

    for result in results:
        seen += 1
        image_path = Path(result.path)
        if not is_image_result(image_path):
            LOGGER.warning(f"跳过非图片结果: {image_path}")
            continue

        orig_img = result.orig_img.copy()
        height, width = orig_img.shape[:2]
        boxes = result.boxes
        masks = result.masks
        shape_items: list[dict[str, Any]] = []
        txt_lines: list[str] = []

        if boxes is not None and len(boxes):
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros(len(cls_ids))
            xyxy_boxes = boxes.xyxy.detach().cpu().numpy()
            polygons_xy = masks.xy if masks is not None else [None] * len(cls_ids)
            polygons_xyn = masks.xyn if masks is not None else [None] * len(cls_ids)

            for idx, cls_id in enumerate(cls_ids):
                model_class_name = model_names.get(int(cls_id), str(cls_id))
                if model_class_name not in allowed_name_to_id:
                    skipped_by_class += 1
                    continue

                output_class_id = allowed_name_to_id[model_class_name]
                conf = float(confs[idx])
                label = model_class_name if hide_conf else f"{model_class_name} {conf:.2f}"
                draw_label = None if hide_labels else label
                color = tuple(int(x) for x in np.random.default_rng(output_class_id).integers(0, 255, size=3))

                polygon_xy = polygons_xy[idx] if idx < len(polygons_xy) else None
                polygon_xyn = polygons_xyn[idx] if idx < len(polygons_xyn) else None

                if polygon_xy is not None and len(polygon_xy) >= 3:
                    points = [[float(x), float(y)] for x, y in np.asarray(polygon_xy, dtype=float)]
                    shape_items.append(
                        {
                            "label": model_class_name,
                            "points": points,
                            "group_id": None,
                            "description": "",
                            "shape_type": "polygon",
                            "flags": {},
                        }
                    )
                    txt_line = polygon_to_yolo_line(
                        output_class_id, np.asarray(polygon_xyn, dtype=float), conf if save_conf else None
                    )
                    if txt_line:
                        txt_lines.append(txt_line)

                    if not nosaveimage:
                        cv2.polylines(orig_img, [np.asarray(points, dtype=np.int32)], True, color, line_thickness)
                        if draw_label:
                            cv2.putText(
                                orig_img,
                                draw_label,
                                tuple(np.asarray(points[0], dtype=int)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                max(line_thickness - 1, 1),
                                cv2.LINE_AA,
                            )
                else:
                    xyxy = xyxy_boxes[idx]
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    shape_items.append(
                        {
                            "label": model_class_name,
                            "points": [[x1, y1], [x2, y2]],
                            "group_id": None,
                            "description": "",
                            "shape_type": "rectangle",
                            "flags": {},
                        }
                    )
                    txt_lines.append(
                        xyxy_to_yolo_line(output_class_id, xyxy, width, height, conf if save_conf else None)
                    )

                    if not nosaveimage:
                        cv2.rectangle(orig_img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
                        if draw_label:
                            cv2.putText(
                                orig_img,
                                draw_label,
                                (int(x1), max(int(y1) - 2, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                max(line_thickness - 1, 1),
                                cv2.LINE_AA,
                            )

                total_kept += 1
                class_counts[model_class_name] += 1

        should_save_empty = not nosave_empty or bool(shape_items)

        if "json" in formats and should_save_empty:
            json_payload = make_labelme_json(image_path, (height, width), shape_items)
            with open(json_dir / f"{image_path.stem}.json", "w", encoding="utf-8") as f:
                json.dump(json_payload, f, ensure_ascii=False, indent=2)
            saved_json += 1

        if "txt" in formats and should_save_empty:
            with open(txt_dir / f"{image_path.stem}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(txt_lines))
                if txt_lines:
                    f.write("\n")
            saved_txt += 1

        if not nosaveimage and should_save_empty:
            cv2.imwrite(str(image_dir / image_path.name), orig_img)
            saved_images += 1

    LOGGER.info("")
    LOGGER.info("自动标注完成")
    LOGGER.info("=" * 50)
    LOGGER.info(f"处理图片数量: {seen}")
    LOGGER.info(f"保留标注数量: {total_kept}")
    LOGGER.info(f"按类别过滤数量: {skipped_by_class}")
    LOGGER.info(f"保存可视化图片: {saved_images}")
    if "json" in formats:
        LOGGER.info(f"保存 JSON 文件: {saved_json} -> {json_dir}")
    if "txt" in formats:
        LOGGER.info(f"保存 TXT 文件: {saved_txt} -> {txt_dir}")
    LOGGER.info("类别统计:")
    for class_name, count in class_counts.items():
        LOGGER.info(f"  {class_name}: {count}")


def parse_opt() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo26n.pt", help="model weights path")
    parser.add_argument("--source", type=str, default="ultralytics/assets", help="image file/dir/glob")
    parser.add_argument("--data", type=str, required=True, help="dataset YAML with names to auto-label")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", default="runs/autolabel", help="save results to project/name")
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
    opt.formats = parse_formats(opt.format)
    return opt


def main(opt: argparse.Namespace) -> None:
    """Run auto-labeling."""
    run(
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
