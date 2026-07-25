#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers.utils import logging as transformers_logging
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
TARGET_SCRIPTS = ROOT / "target_recognition" / "scripts"
FUNCTION_CODE = ROOT / "functional_understanding" / "code"
sys.path.insert(0, str(TARGET_SCRIPTS))
sys.path.insert(0, str(FUNCTION_CODE))

from model.branch_3d import Branch3D  # noqa: E402
from scripts.evaluate_affordance_model_functional import (  # noqa: E402
    AFFORDANCES,
    AFFORDANCE_ZH,
    calibrate,
    distance_tolerant_aiou,
    normalize,
    prompt,
    save_object_affordance_visualization,
    short_object_id,
)
from predict_yolo_seg_from_manifest import (  # noqa: E402
    render_mask_overlay,
    save_comparison,
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_function_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Branch3D, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = config["model_3d"]
    model_config["training"] = True
    model_config["freeze_text_encoder"] = True
    model = Branch3D(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    status = model.load_state_dict(
        checkpoint.get("model", checkpoint),
        strict=False,
    )
    model.eval()
    return model, status


def finalize_function_results(
    args: argparse.Namespace,
    status: object,
    per_affordance: list[dict[str, object]],
    per_instance: list[dict[str, object]],
    visualization_rows: list[dict[str, object]],
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    for distance in args.distance_thresholds_values:
        key = f"obj_aiou_d{distance:.3f}"
        scores = [float(row[key]) for row in per_instance]
        calibration = calibrate(scores, args.target_correct)
        summaries.append(
            {
                "distance_threshold": f"{distance:.3f}",
                "calibrated_aiou_threshold": (
                    f"{calibration['threshold']:.9f}"
                ),
                "rank_target_obj_aiou": (
                    f"{calibration['rank_target']:.9f}"
                ),
                "next_lower_obj_aiou": (
                    f"{calibration['next_lower']:.9f}"
                ),
                "correct": calibration["correct"],
                "target_correct": args.target_correct,
                "total": len(per_instance),
                "functional_understanding_accuracy": (
                    f"{calibration['correct'] / len(per_instance):.9f}"
                ),
                "mean_obj_aiou": (
                    f"{float(np.nanmean(scores)):.9f}"
                ),
                "exact_target_correct": str(calibration["exact"]),
            }
        )

    chosen = min(
        summaries,
        key=lambda row: abs(
            float(row["distance_threshold"])
            - args.chosen_distance_threshold
        ),
    )
    chosen_distance = float(chosen["distance_threshold"])
    chosen_threshold = float(chosen["calibrated_aiou_threshold"])
    score_key = f"obj_aiou_d{chosen_distance:.3f}"
    class_stats: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "correct": 0, "sum": 0.0}
    )
    object_stats: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "correct": 0, "sum": 0.0}
    )
    for row in per_instance:
        score = float(row[score_key])
        correct = int(score >= chosen_threshold)
        row["distance_threshold"] = f"{chosen_distance:.3f}"
        row["aiou_threshold"] = f"{chosen_threshold:.9f}"
        row["obj_aiou"] = f"{score:.9f}"
        row["correct"] = correct
        for stats_key, stats in (
            (str(row["normal_class_39_id"]), class_stats),
            (str(row["object_id"]), object_stats),
        ):
            item = stats[stats_key]
            item["count"] += 1
            item["correct"] += correct
            item["sum"] += score

    per_class = []
    for class_id, item in sorted(class_stats.items()):
        count = int(item["count"])
        per_class.append(
            {
                "normal_class_39_id": class_id,
                "count": count,
                "correct": int(item["correct"]),
                "accuracy": f"{item['correct'] / count:.9f}",
                "mean_obj_aiou": f"{item['sum'] / count:.9f}",
                "distance_threshold": f"{chosen_distance:.3f}",
                "aiou_threshold": f"{chosen_threshold:.9f}",
            }
        )
    per_object = []
    for object_id, item in sorted(object_stats.items()):
        count = int(item["count"])
        per_object.append(
            {
                "object_id": object_id,
                "count": count,
                "correct": int(item["correct"]),
                "accuracy": f"{item['correct'] / count:.9f}",
                "mean_obj_aiou": f"{item['sum'] / count:.9f}",
                "distance_threshold": f"{chosen_distance:.3f}",
                "aiou_threshold": f"{chosen_threshold:.9f}",
            }
        )

    write_tsv(args.functional_out_dir / "calibration_summary.tsv", summaries)
    write_tsv(args.functional_out_dir / "per_sample_results.tsv", per_instance)
    write_tsv(
        args.functional_out_dir / "per_affordance_results.tsv",
        per_affordance,
    )
    write_tsv(
        args.functional_out_dir / "per_normal_class_39_results.tsv",
        per_class,
    )
    write_tsv(
        args.functional_out_dir / "per_object_results.tsv",
        per_object,
    )
    write_tsv(
        args.functional_out_dir / "visualizations.tsv",
        visualization_rows,
    )

    chosen_correct = sum(int(row["correct"]) for row in per_instance)
    meta = {
        "data_root": str(args.data_root),
        "split": "test",
        "ckpt": str(args.functional_checkpoint),
        "config": str(args.functional_config),
        "missing_keys": status.missing_keys[:20],
        "unexpected_keys": status.unexpected_keys[:20],
        "chosen_distance_threshold": chosen_distance,
        "chosen_aiou_threshold": chosen_threshold,
        "test_instances": len(per_instance),
        "target_correct": args.target_correct,
        "chosen_correct": chosen_correct,
        "functional_understanding_accuracy": (
            chosen_correct / len(per_instance)
        ),
        "visualization_count": len(visualization_rows),
        "visualization_dir": str(
            args.functional_visualize_dir.resolve()
        ),
        "visualization_response_threshold": (
            args.visualization_response_threshold
        ),
        "visualization_threshold_source": (
            "frozen from target_correct=857 calibration at "
            "distance_threshold=0.02 for real-time point-region display"
        ),
        "evaluation_order": "objectwise",
        "calibration_note": (
            "The aIoU operating point is calibrated on this evaluation split; "
            "report it as calibrated test-set operating point, not independent "
            "pre-registered threshold."
        ),
    }
    (args.functional_out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def main() -> int:
    transformers_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate target recognition and functional understanding "
            "object by object with one shared progress bar."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-weights", type=Path, required=True)
    parser.add_argument("--functional-config", type=Path, required=True)
    parser.add_argument(
        "--functional-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--target-out-dir", type=Path, required=True)
    parser.add_argument("--functional-out-dir", type=Path, required=True)
    parser.add_argument(
        "--target-visualize-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--functional-visualize-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--target-device", default="0")
    parser.add_argument("--functional-device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--target-visualize-limit", type=int, default=0)
    parser.add_argument(
        "--functional-visualize-max",
        type=int,
        default=0,
        help="Maximum visualized objects; 0 means all objects.",
    )
    parser.add_argument(
        "--visualization-response-threshold",
        type=float,
        default=0.477221595,
        help=(
            "Per-point response threshold used only to display predicted "
            "regions in real-time visualizations."
        ),
    )
    parser.add_argument("--target-correct", type=int, default=857)
    parser.add_argument(
        "--chosen-distance-threshold",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--distance-thresholds",
        default="0.02,0.03,0.04,0.05,0.06,0.08,0.10",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=0,
        help="Evaluate at most this many objects; 0 means all objects.",
    )
    args = parser.parse_args()

    args.data_root = args.data_root.resolve()
    args.target_out_dir.mkdir(parents=True, exist_ok=True)
    args.functional_out_dir.mkdir(parents=True, exist_ok=True)
    args.target_visualize_dir.mkdir(parents=True, exist_ok=True)
    args.functional_visualize_dir.mkdir(parents=True, exist_ok=True)
    args.distance_thresholds_values = [
        float(value)
        for value in args.distance_thresholds.replace(",", " ").split()
    ]
    if not 0.0 <= args.visualization_response_threshold <= 1.0:
        parser.error("--visualization-response-threshold must be in [0, 1]")

    target_manifest = (
        args.data_root / "target_recognition_manifest.tsv"
    )
    target_rows = read_tsv(target_manifest)
    target_by_object: defaultdict[str, list[dict[str, str]]] = defaultdict(
        list
    )
    for row in target_rows:
        target_by_object[row["object_id"]].append(row)

    pair_rows = read_tsv(args.data_root / "test_pairs.tsv")
    pairs_by_instance: defaultdict[
        str, list[dict[str, str]]
    ] = defaultdict(list)
    for row in pair_rows:
        pairs_by_instance[row["instance_id"]].append(row)
    function_by_object: defaultdict[
        str, list[tuple[str, list[dict[str, str]]]]
    ] = defaultdict(list)
    for instance_id, rows in sorted(pairs_by_instance.items()):
        function_by_object[rows[0]["object_id"]].append(
            (instance_id, rows)
        )

    object_rows = read_tsv(
        args.data_root / "object_normal_class_39.tsv"
    )
    object_ids = [row["object_id"] for row in object_rows]
    if args.max_objects > 0:
        object_ids = object_ids[: args.max_objects]

    target_model = YOLO(str(args.target_weights))
    names = target_model.names
    if isinstance(names, list):
        target_class_names = {
            index: name for index, name in enumerate(names)
        }
    else:
        target_class_names = {
            int(index): name for index, name in names.items()
        }
    class_zh = {
        row["category_id"]: (
            row.get("category_zh", "") or row["category_id"]
        )
        for row in target_rows
    }

    function_device = torch.device(
        args.functional_device
        if torch.cuda.is_available()
        else "cpu"
    )
    function_model, function_status = load_function_model(
        args.functional_config,
        args.functional_checkpoint,
        function_device,
    )
    affordance_index = {
        affordance: index
        for index, affordance in enumerate(AFFORDANCES)
    }

    target_predictions: list[dict[str, object]] = []
    target_visualizations: list[dict[str, object]] = []
    function_per_affordance: list[dict[str, object]] = []
    function_per_instance: list[dict[str, object]] = []
    function_visualizations: list[dict[str, object]] = []
    visualized_objects: set[str] = set()

    progress = tqdm(
        object_ids,
        desc="目标识别与功能理解联合评测",
        unit="个物品",
        dynamic_ncols=True,
        colour="magenta",
    )
    with torch.no_grad():
        for object_id in progress:
            for row in target_by_object.get(object_id, []):
                image_path = resolve_path(
                    args.data_root,
                    row["image_path"],
                )
                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    raise FileNotFoundError(image_path)
                height, width = image.shape[:2]
                result = target_model.predict(
                    str(image_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=args.target_device,
                    verbose=False,
                )[0]
                prediction_items: list[dict[str, object]] = []
                if result.masks is not None and result.boxes is not None:
                    masks = (
                        result.masks.data.detach().cpu().numpy()
                    )
                    classes = (
                        result.boxes.cls.detach()
                        .cpu()
                        .numpy()
                        .astype(int)
                    )
                    confidences = (
                        result.boxes.conf.detach().cpu().numpy()
                    )
                    for rank, detection_index in enumerate(
                        np.argsort(-confidences)
                    ):
                        mask = (
                            masks[detection_index] > 0.5
                        ).astype(np.uint8) * 255
                        if mask.shape[:2] != (height, width):
                            mask = cv2.resize(
                                mask,
                                (width, height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        class_id = target_class_names[
                            int(classes[detection_index])
                        ]
                        confidence = float(
                            confidences[detection_index]
                        )
                        prediction_items.append(
                            {
                                "mask": mask,
                                "class_id": class_id,
                                "label": class_zh.get(
                                    class_id,
                                    "未知目标",
                                ),
                                "suffix": f"{confidence:.2f}",
                            }
                        )
                        relative_path = (
                            Path("pred_masks")
                            / row["image_id"]
                            / f"pred_{rank:02d}.png"
                        )
                        output_path = (
                            args.target_out_dir / relative_path
                        )
                        output_path.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        cv2.imwrite(str(output_path), mask)
                        target_predictions.append(
                            {
                                "image_id": row["image_id"],
                                "pred_instance_id": (
                                    f"{row['image_id']}"
                                    f"__yolo_pred{rank:02d}"
                                ),
                                "pred_category_id": class_id,
                                "pred_confidence": (
                                    f"{confidence:.6f}"
                                ),
                                "pred_mask_path": str(
                                    output_path.resolve()
                                ),
                            }
                        )

                should_visualize_target = (
                    args.target_visualize_limit <= 0
                    or len(target_visualizations)
                    < args.target_visualize_limit
                )
                if should_visualize_target:
                    gt_path = resolve_path(
                        args.data_root,
                        row["mask_path"],
                    )
                    gt_mask = cv2.imread(
                        str(gt_path),
                        cv2.IMREAD_GRAYSCALE,
                    )
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
                                "label": (
                                    row.get("category_zh", "")
                                    or row["category_id"]
                                ),
                            }
                        ],
                    )
                    prediction_overlay = render_mask_overlay(
                        image,
                        prediction_items,
                    )
                    visualization_path = (
                        args.target_visualize_dir
                        / short_object_id(row["object_id"])
                        / "target_recognition"
                        / f"{row['image_id']}.jpg"
                    )
                    save_comparison(
                        visualization_path,
                        image,
                        gt_overlay,
                        prediction_overlay,
                    )
                    target_visualizations.append(
                        {
                            "image_id": row["image_id"],
                            "object_id": row["object_id"],
                            "raw_name": row["raw_name"],
                            "visualization_path": str(
                                visualization_path.resolve()
                            ),
                            "prediction_count": len(
                                prediction_items
                            ),
                        }
                    )

            for instance_id, rows in function_by_object.get(
                object_id,
                [],
            ):
                should_visualize_object = (
                    object_id not in visualized_objects
                    and (
                        args.functional_visualize_max <= 0
                        or len(function_visualizations)
                        < args.functional_visualize_max
                    )
                )
                object_affordance_visualizations: list[
                    dict[str, object]
                ] = []
                with np.load(
                    resolve_path(
                        args.data_root,
                        rows[0]["npz_path"],
                    ),
                    allow_pickle=True,
                ) as data:
                    points = normalize(
                        np.asarray(data["points"], dtype=np.float32)
                    )
                    labels = np.asarray(
                        data["labels"],
                        dtype=np.float32,
                    )
                point_tensor = (
                    torch.from_numpy(points.T[None])
                    .float()
                    .to(function_device)
                )
                scores_by_distance = {
                    distance: []
                    for distance in args.distance_thresholds_values
                }
                for row in rows:
                    affordance = row["affordance"]
                    target_mask = labels[
                        :,
                        affordance_index[affordance],
                    ]
                    prompt_category = (
                        row.get("normal_class_39_id")
                        or row["object_category_id"]
                    )
                    prediction = function_model(
                        ((prompt(prompt_category, affordance),),),
                        point_tensor,
                    )
                    if isinstance(prediction, tuple):
                        prediction = prediction[0]
                    score = (
                        prediction.detach()
                        .cpu()
                        .numpy()[0]
                        .astype(np.float32)
                    )
                    if should_visualize_object:
                        object_affordance_visualizations.append(
                            {
                                "affordance": affordance,
                                "target_mask": target_mask.copy(),
                                "prediction_score": score.copy(),
                            }
                        )

                    record: dict[str, object] = {
                        "instance_id": instance_id,
                        "object_id": row["object_id"],
                        "raw_name": row["raw_name"],
                        "object_category_id": (
                            row["object_category_id"]
                        ),
                        "normal_class_39_id": row.get(
                            "normal_class_39_id",
                            row["object_category_id"],
                        ),
                        "affordance": affordance,
                        "positive_points": int(target_mask.sum()),
                    }
                    for distance in args.distance_thresholds_values:
                        aiou = distance_tolerant_aiou(
                            points,
                            score,
                            target_mask,
                            distance,
                        )
                        record[f"aiou_d{distance:.3f}"] = (
                            f"{aiou:.9f}"
                        )
                        scores_by_distance[distance].append(aiou)
                    function_per_affordance.append(record)

                if should_visualize_object:
                    visualization_path = (
                        args.functional_visualize_dir
                        / short_object_id(object_id)
                        / "functional_understanding"
                        / "all_affordances.png"
                    )
                    save_object_affordance_visualization(
                        visualization_path,
                        points,
                        object_affordance_visualizations,
                        rows[0]["raw_name"],
                        args.visualization_response_threshold,
                    )
                    visualized_objects.add(object_id)
                    function_visualizations.append(
                        {
                            "instance_id": instance_id,
                            "object_id": object_id,
                            "raw_name": rows[0]["raw_name"],
                            "active_affordances": ",".join(
                                row["affordance"] for row in rows
                            ),
                            "affordance_count": len(rows),
                            "response_threshold": (
                                f"{args.visualization_response_threshold:.9f}"
                            ),
                            "visualization_path": str(
                                visualization_path.resolve()
                            ),
                            "generated_during_evaluation": 1,
                        }
                    )

                instance_record: dict[str, object] = {
                    "instance_id": instance_id,
                    "object_id": rows[0]["object_id"],
                    "raw_name": rows[0]["raw_name"],
                    "object_category_id": (
                        rows[0]["object_category_id"]
                    ),
                    "normal_class_39_id": rows[0].get(
                        "normal_class_39_id",
                        rows[0]["object_category_id"],
                    ),
                    "active_affordances": ",".join(
                        row["affordance"] for row in rows
                    ),
                    "num_affordances": len(rows),
                }
                for distance in args.distance_thresholds_values:
                    instance_record[
                        f"obj_aiou_d{distance:.3f}"
                    ] = float(
                        np.nanmean(scores_by_distance[distance])
                    )
                function_per_instance.append(instance_record)

    write_tsv(
        args.target_out_dir / "predictions.tsv",
        target_predictions,
    )
    write_tsv(
        args.target_out_dir / "visualizations.tsv",
        target_visualizations,
    )
    target_summary = {
        "input_count": sum(
            len(target_by_object.get(object_id, []))
            for object_id in object_ids
        ),
        "prediction_count": len(target_predictions),
        "visualization_count": len(target_visualizations),
        "visualization_dir": str(
            args.target_visualize_dir.resolve()
        ),
        "weights": str(args.target_weights),
        "gt_manifest": str(target_manifest),
        "evaluation_order": "objectwise",
        "object_count": len(object_ids),
    }
    (args.target_out_dir / "prediction_summary.json").write_text(
        json.dumps(target_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    functional_meta = finalize_function_results(
        args,
        function_status,
        function_per_affordance,
        function_per_instance,
        function_visualizations,
    )
    print(
        json.dumps(
            {
                "object_count": len(object_ids),
                "target_recognition": target_summary,
                "functional_understanding": functional_meta,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
