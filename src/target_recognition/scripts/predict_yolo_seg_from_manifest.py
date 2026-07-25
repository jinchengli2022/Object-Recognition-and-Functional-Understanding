#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO


PALETTE_BGR = [
    (255, 92, 92),
    (75, 192, 255),
    (95, 214, 121),
    (255, 178, 72),
    (196, 112, 255),
    (255, 111, 206),
    (72, 214, 205),
    (126, 146, 255),
]
FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def short_object_id(object_id: str) -> str:
    parts = object_id.split("_", 2)
    if (
        len(parts) < 2
        or parts[0] != "obj"
        or not parts[1].isdigit()
    ):
        raise ValueError(f"Invalid object_id: {object_id!r}")
    return f"obj_{parts[1]}"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_ascii_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def color_for_class(class_id: str) -> tuple[int, int, int]:
    index = zlib.crc32(class_id.encode("utf-8")) % len(PALETTE_BGR)
    return PALETTE_BGR[index]


def add_labels(
    image_bgr: np.ndarray,
    labels: list[
        tuple[str, str, tuple[int, int], tuple[int, int, int]]
    ],
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(canvas)
    font_size = max(18, round(image_bgr.shape[1] / 25))
    chinese_font = load_font(font_size)
    ascii_font = load_ascii_font(font_size)
    for text, suffix, (x, y), color_bgr in labels:
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        text_bbox = draw.textbbox(
            (0, 0),
            text,
            font=chinese_font,
            stroke_width=1,
        )
        text_width = text_bbox[2] - text_bbox[0]
        suffix_bbox = draw.textbbox(
            (0, 0),
            suffix,
            font=ascii_font,
            stroke_width=1,
        )
        suffix_width = suffix_bbox[2] - suffix_bbox[0]
        text_height = max(
            text_bbox[3] - text_bbox[1],
            suffix_bbox[3] - suffix_bbox[1],
        )
        gap = 7 if suffix else 0
        margin = 5
        background = (
            x - margin,
            y - margin,
            x + text_width + gap + suffix_width + margin,
            y + text_height + margin,
        )
        draw.rounded_rectangle(background, radius=5, fill=(18, 18, 18))
        draw.text(
            (x, y),
            text,
            font=chinese_font,
            fill=color_rgb,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
        if suffix:
            draw.text(
                (x + text_width + gap, y),
                suffix,
                font=ascii_font,
                fill=color_rgb,
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def render_mask_overlay(
    image_bgr: np.ndarray,
    items: list[dict[str, object]],
    alpha: float = 0.48,
) -> np.ndarray:
    overlay = image_bgr.copy()
    labels: list[
        tuple[str, str, tuple[int, int], tuple[int, int, int]]
    ] = []
    for item in items:
        mask = np.asarray(item["mask"], dtype=np.uint8) > 0
        if not np.any(mask):
            continue
        class_id = str(item["class_id"])
        color = color_for_class(class_id)
        color_array = np.asarray(color, dtype=np.float32)
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1.0 - alpha)
            + color_array * alpha
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, color, 2, lineType=cv2.LINE_AA)
        x, y, _, _ = cv2.boundingRect(mask.astype(np.uint8))
        labels.append(
            (
                str(item["label"]),
                str(item.get("suffix", "")),
                (x + 4, max(4, y + 4)),
                color,
            )
        )
    return add_labels(overlay, labels)


def save_comparison(
    output_path: Path,
    original_bgr: np.ndarray,
    gt_bgr: np.ndarray,
    prediction_bgr: np.ndarray,
) -> None:
    panels = [original_bgr, gt_bgr, prediction_bgr]
    titles = ["原始图像", "人工标注真值", "模型预测"]
    height, width = original_bgr.shape[:2]
    header_height = max(50, round(height * 0.07))
    canvas = Image.new("RGB", (width * len(panels), height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(22, round(width / 18)))
    for index, (panel, title) in enumerate(zip(panels, titles)):
        panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(panel_rgb), (index * width, header_height))
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        x = index * width + (width - text_width) // 2
        y = max(4, (header_height - (bbox[3] - bbox[1])) // 2 - bbox[1])
        draw.text((x, y), title, fill=(24, 24, 24), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, subsampling=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run YOLO-seg on a target-recognition manifest."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--gt-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument(
        "--visualize-dir",
        type=Path,
        help="Save original/GT/prediction comparison images to this directory.",
    )
    parser.add_argument(
        "--visualize-limit",
        type=int,
        default=0,
        help="Maximum number of comparison images; 0 means all images.",
    )
    args = parser.parse_args()

    root = args.gt_manifest.resolve().parent
    rows = read_tsv(args.gt_manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize_dir:
        args.visualize_dir.mkdir(parents=True, exist_ok=True)

    class_zh = {
        row["category_id"]: row.get("category_zh", "") or row["category_id"]
        for row in rows
    }
    model = YOLO(str(args.weights))
    names = model.names
    if isinstance(names, list):
        idx_to_class = {i: name for i, name in enumerate(names)}
    else:
        idx_to_class = {int(key): value for key, value in names.items()}

    pred_rows: list[dict[str, object]] = []
    visualization_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(
        tqdm(
            rows,
            desc="目标识别推理",
            unit="张",
            dynamic_ncols=True,
            colour="green",
        )
    ):
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = root / img_path
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(img_path)
        height, width = image.shape[:2]
        result = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        prediction_items: list[dict[str, object]] = []
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            order = np.argsort(-confidences)
            for rank, detection_index in enumerate(order):
                mask = (masks[detection_index] > 0.5).astype(np.uint8) * 255
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(
                        mask,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                class_id = idx_to_class[int(classes[detection_index])]
                confidence = float(confidences[detection_index])
                display_class = class_zh.get(class_id, "未知目标")
                prediction_items.append(
                    {
                        "mask": mask,
                        "class_id": class_id,
                        "label": display_class,
                        "suffix": f"{confidence:.2f}",
                    }
                )
                pred_rel = (
                    Path("pred_masks")
                    / row["image_id"]
                    / f"pred_{rank:02d}.png"
                )
                (args.out_dir / pred_rel).parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                cv2.imwrite(str(args.out_dir / pred_rel), mask)
                pred_rows.append(
                    {
                        "image_id": row["image_id"],
                        "pred_instance_id": (
                            f"{row['image_id']}__yolo_pred{rank:02d}"
                        ),
                        "pred_category_id": class_id,
                        "pred_confidence": f"{confidence:.6f}",
                        "pred_mask_path": str(
                            (args.out_dir / pred_rel).resolve()
                        ),
                    }
                )

        should_visualize = args.visualize_dir is not None and (
            args.visualize_limit <= 0
            or len(visualization_rows) < args.visualize_limit
        )
        if should_visualize:
            gt_path = Path(row["mask_path"])
            if not gt_path.is_absolute():
                gt_path = root / gt_path
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                raise FileNotFoundError(gt_path)
            if gt_mask.shape != (height, width):
                gt_mask = cv2.resize(
                    gt_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            gt_overlay = render_mask_overlay(
                image,
                [
                    {
                        "mask": gt_mask,
                        "class_id": row["category_id"],
                        "label": row.get("category_zh", "")
                        or row["category_id"],
                    }
                ],
            )
            pred_overlay = render_mask_overlay(image, prediction_items)
            visualization_path = (
                args.visualize_dir
                / short_object_id(row["object_id"])
                / "target_recognition"
                / f"{row['image_id']}.jpg"
            )
            save_comparison(
                visualization_path,
                image,
                gt_overlay,
                pred_overlay,
            )
            visualization_rows.append(
                {
                    "image_id": row["image_id"],
                    "object_id": row["object_id"],
                    "raw_name": row["raw_name"],
                    "visualization_path": str(visualization_path.resolve()),
                    "prediction_count": len(prediction_items),
                }
            )

    write_tsv(args.out_dir / "predictions.tsv", pred_rows)
    if visualization_rows:
        write_tsv(
            args.out_dir / "visualizations.tsv",
            visualization_rows,
        )
    summary = {
        "input_count": len(rows),
        "prediction_count": len(pred_rows),
        "visualization_count": len(visualization_rows),
        "visualization_dir": (
            str(args.visualize_dir.resolve()) if args.visualize_dir else None
        ),
        "weights": str(args.weights),
        "gt_manifest": str(args.gt_manifest),
    }
    (args.out_dir / "prediction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
