#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if shape is not None and mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def evaluate(gt_manifest: Path, pred_manifest: Path, out_dir: Path) -> dict[str, object]:
    root = gt_manifest.resolve().parent
    gt_rows = read_tsv(gt_manifest)
    pred_rows = read_tsv(pred_manifest)
    preds_by_image: dict[str, list[dict[str, str]]] = {}
    for row in pred_rows:
        preds_by_image.setdefault(row["image_id"], []).append(row)

    result_rows: list[dict[str, object]] = []
    by_category: dict[str, list[float]] = {}
    category_correct = 0

    for gt in gt_rows:
        gt_mask_path = root / gt["mask_path"]
        gt_mask = read_mask(gt_mask_path)
        candidates = preds_by_image.get(gt["image_id"], [])

        best = None
        best_iou = -1.0
        best_class_match_iou = -1.0
        for pred in candidates:
            pred_mask = read_mask(Path(pred["pred_mask_path"]), gt_mask.shape)
            value = iou(gt_mask, pred_mask)
            if value > best_iou:
                best_iou = value
                best = pred
            if pred.get("pred_category_id") == gt["category_id"]:
                best_class_match_iou = max(best_class_match_iou, value)

        if best is None:
            pred_category = ""
            pred_instance_id = ""
            mask_iou = 0.0
            class_match_iou = 0.0
            class_correct = False
        else:
            pred_category = best.get("pred_category_id", "")
            pred_instance_id = best.get("pred_instance_id", "")
            mask_iou = max(0.0, best_iou)
            class_match_iou = max(0.0, best_class_match_iou)
            class_correct = pred_category == gt["category_id"]
            category_correct += int(class_correct)

        by_category.setdefault(gt["category_id"], []).append(class_match_iou)
        result_rows.append(
            {
                "image_id": gt["image_id"],
                "instance_id": gt["instance_id"],
                "object_id": gt["object_id"],
                "raw_name": gt["raw_name"],
                "gt_category_id": gt["category_id"],
                "pred_instance_id": pred_instance_id,
                "pred_category_id": pred_category,
                "mask_iou_best_any_class": f"{mask_iou:.6f}",
                "mask_iou_class_matched": f"{class_match_iou:.6f}",
                "class_correct": int(class_correct),
            }
        )

    per_category_rows = []
    for category_id, values in sorted(by_category.items()):
        per_category_rows.append(
            {
                "category_id": category_id,
                "count": len(values),
                "miou": f"{float(np.mean(values)) if values else 0.0:.6f}",
            }
        )

    miou = float(np.mean([float(row["mask_iou_class_matched"]) for row in result_rows])) if result_rows else 0.0
    any_class_miou = float(np.mean([float(row["mask_iou_best_any_class"]) for row in result_rows])) if result_rows else 0.0
    category_accuracy = category_correct / len(result_rows) if result_rows else 0.0
    summary = {
        "gt_count": len(gt_rows),
        "prediction_count": len(pred_rows),
        "miou_class_matched": miou,
        "miou_best_any_class": any_class_miou,
        "category_accuracy": category_accuracy,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "per_instance_results.tsv", result_rows)
    write_tsv(out_dir / "per_category_miou.tsv", per_category_rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate target recognition masks/classes against GT.")
    parser.add_argument("--gt-manifest", type=Path, default=ROOT / "target_recognition_v1" / "manifest.tsv")
    parser.add_argument("--pred-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "target_recognition_v1" / "eval")
    args = parser.parse_args()

    summary = evaluate(args.gt_manifest, args.pred_manifest, args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
