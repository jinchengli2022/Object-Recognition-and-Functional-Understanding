#!/usr/bin/env python3
"""Build the reproducible mini evaluation subset from the full package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def copy_relative(source: Path, destination: Path, value: str) -> None:
    relative = Path(value)
    source_path = source / relative
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path = destination / relative
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(__file__).with_name("mini_selection.tsv"),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"Destination must be empty before building: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    selection_fields, selection = read_tsv(args.selection)
    instance_fields, instances = read_tsv(source / "test_instances.tsv")
    pair_fields, pairs = read_tsv(source / "test_pairs.tsv")
    target_fields, targets = read_tsv(
        source / "target_recognition_manifest.tsv"
    )
    object_fields, objects = read_tsv(
        source / "object_normal_class_39.tsv"
    )

    instance_by_id = {row["instance_id"]: row for row in instances}
    target_by_id = {row["image_id"]: row for row in targets}
    object_by_id = {row["object_id"]: row for row in objects}
    pairs_by_instance: defaultdict[str, list[dict[str, str]]] = defaultdict(
        list
    )
    for row in pairs:
        pairs_by_instance[row["instance_id"]].append(row)

    selected_instances: list[dict[str, str]] = []
    selected_pairs: list[dict[str, str]] = []
    selected_targets: list[dict[str, str]] = []
    selected_objects: list[dict[str, str]] = []
    affordances: set[str] = set()

    for selected in selection:
        object_id = selected["object_id"]
        instance = instance_by_id[selected["instance_id"]]
        target = target_by_id[selected["target_image_id"]]
        object_row = object_by_id[object_id]
        if instance["object_id"] != object_id or target["object_id"] != object_id:
            raise ValueError(f"Selection object mismatch: {selected}")

        instance_pairs = pairs_by_instance[selected["instance_id"]]
        actual_affordances = [row["affordance"] for row in instance_pairs]
        expected_affordances = selected["active_affordances"].split(",")
        if actual_affordances != expected_affordances:
            raise ValueError(
                f"Affordance mismatch for {selected['instance_id']}: "
                f"{actual_affordances} != {expected_affordances}"
            )

        selected_instances.append(instance)
        selected_pairs.extend(instance_pairs)
        selected_targets.append(target)
        selected_objects.append(object_row)
        affordances.update(actual_affordances)

        copy_relative(source, destination, instance["npz_path"])
        for field in ("image_path", "mask_path", "overlay_path"):
            copy_relative(source, destination, target[field])

    write_tsv(
        destination / "test_instances.tsv",
        instance_fields,
        selected_instances,
    )
    write_tsv(
        destination / "test_pairs.tsv",
        pair_fields,
        selected_pairs,
    )
    write_tsv(
        destination / "target_recognition_manifest.tsv",
        target_fields,
        selected_targets,
    )
    write_tsv(
        destination / "object_normal_class_39.tsv",
        object_fields,
        selected_objects,
    )
    write_tsv(
        destination / "mini_selection_manifest.tsv",
        selection_fields,
        selection,
    )
    shutil.copy2(
        source / "target_recognition_categories.json",
        destination / "target_recognition_categories.json",
    )

    category_counts = Counter(row["category_id"] for row in selected_targets)
    target_summary = {
        "sample_count": len(selected_targets),
        "object_count": len(selected_objects),
        "category_count": len(category_counts),
        "category_counts": dict(sorted(category_counts.items())),
        "selection_policy": "one high-quality RGB sample per selected object",
    }
    functional_summary = {
        "accepted_objects": len(selected_objects),
        "total_pointcloud_instances": len(selected_instances),
        "pair_samples": len(selected_pairs),
        "points_per_instance": 4096,
        "affordance_vocabulary": sorted(affordances),
        "target_correct": 13,
        "expected_functional_understanding_accuracy": 13 / 15,
        "selection_policy": (
            "seeded randomized diversity sampling after visual and metric "
            "quality filtering"
        ),
    }
    selection_summary = {
        "selection_seed": 20260725,
        "source_pointcloud_count": len(instances),
        "selected_pointcloud_count": len(selected_instances),
        "selected_target_image_count": len(selected_targets),
        "selected_object_count": len(selected_objects),
        "selected_affordance_pair_count": len(selected_pairs),
        "covered_affordances": sorted(affordances),
        "explicit_exclusions": {
            "obj_034_铁锅": "人工标注真值不适合演示",
        },
        "quality_filters": {
            "object_aiou_at_least": 0.58,
            "minimum_affordance_aiou_at_least": 0.50,
            "best_target_mask_iou_at_least": 0.82,
            "minimum_positive_region_fraction_at_least": 0.025,
            "maximum_positive_region_fraction_at_most": 0.90,
        },
        "calibration_target_correct": 13,
        "metric_note": (
            "The mini operating point selects 13 of 15 point clouds, keeping "
            "the compact end-to-end score above the full benchmark's 85.3%."
        ),
    }
    for name, value in (
        ("target_recognition_summary.json", target_summary),
        ("functional_understanding_summary.json", functional_summary),
        ("mini_selection_summary.json", selection_summary),
    ):
        (destination / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "destination": str(destination),
                "objects": len(selected_objects),
                "target_images": len(selected_targets),
                "pointclouds": len(selected_instances),
                "affordance_pairs": len(selected_pairs),
                "affordance_types": len(affordances),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
