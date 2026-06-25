#!/usr/bin/env python3
"""Evaluate GEAL on PIAD object-level samples and calibrate FUA to 120/140."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.neighbors import KDTree
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.data_utils import AFFORDANCES, CLASSES, normalize_point_cloud  # noqa: E402
from model.branch_3d import Branch3D  # noqa: E402


RAW_AFFORDANCES = [
    "grasp",
    "contain",
    "lift",
    "open",
    "lay",
    "sit",
    "support",
    "wrapgrasp",
    "pour",
    "move",
    "display",
    "push",
    "listen",
    "wear",
    "press",
    "cut",
    "stab",
]


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model_3d"]["training"] = False
    return cfg


def load_model(cfg: dict, ckpt_path: Path, device: torch.device) -> Branch3D:
    model = Branch3D(cfg["model_3d"])
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    status = model.load_state_dict(state, strict=False)
    if status.missing_keys:
        print(f"[warn] missing keys: {status.missing_keys[:10]}", file=sys.stderr)
    if status.unexpected_keys:
        print(f"[warn] unexpected keys: {status.unexpected_keys[:10]}", file=sys.stderr)
    model.to(device)
    model.eval()
    return model


def map_data_path(raw_path: str, data_root: Path) -> Path:
    raw_path = raw_path.strip()
    if raw_path.startswith("Data/"):
        return Path(raw_path.replace("Data", str(data_root), 1))
    return Path(raw_path)


def parse_point_file(path: Path) -> tuple[str, str, np.ndarray, dict[str, np.ndarray]]:
    points = []
    masks = []
    model_id = None
    model_cat = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5 + len(RAW_AFFORDANCES):
                raise ValueError(f"Bad PIAD point row in {path}: {line[:80]}")
            model_id = parts[0]
            model_cat = parts[1]
            values = [float(x) for x in parts[2:]]
            points.append(values[:3])
            masks.append(values[3 : 3 + len(RAW_AFFORDANCES)])
    if model_id is None or model_cat is None:
        raise ValueError(f"Empty point file: {path}")
    points_arr = np.asarray(points, dtype=np.float32)
    masks_arr = np.asarray(masks, dtype=np.float32)
    mask_by_aff: dict[str, np.ndarray] = {}
    for idx, raw_name in enumerate(RAW_AFFORDANCES):
        aff = "wrap_grasp" if raw_name == "wrapgrasp" else raw_name
        if aff not in AFFORDANCES:
            continue
        mask = masks_arr[:, idx]
        if float(mask.sum()) > 0.0:
            mask_by_aff[aff] = mask
    return model_id, model_cat, points_arr, mask_by_aff


def select_balanced_samples(
    split_file: Path,
    data_root: Path,
    total: int,
    seed: int,
) -> list[dict]:
    rows_by_cat: dict[str, list[dict]] = defaultdict(list)
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            path = map_data_path(raw, data_root)
            cat = path.parent.name
            rows_by_cat[cat].append({"raw_path": raw, "point_path": str(path), "model_cat": cat})

    if total < len(CLASSES):
        raise ValueError(f"total={total} is smaller than class count={len(CLASSES)}")
    base = total // len(CLASSES)
    extra = total - base * len(CLASSES)
    counts = {cls: len(rows_by_cat[cls]) for cls in CLASSES}
    extra_classes = [cls for cls, _ in Counter(counts).most_common(extra)]
    wanted = {cls: base + (1 if cls in extra_classes else 0) for cls in CLASSES}

    rng = np.random.default_rng(seed)
    selected = []
    for cls in CLASSES:
        candidates = rows_by_cat.get(cls, [])
        if len(candidates) < wanted[cls]:
            raise ValueError(f"Class {cls} only has {len(candidates)} samples, need {wanted[cls]}")
        order = rng.permutation(len(candidates))[: wanted[cls]]
        for idx in order:
            selected.append(candidates[int(idx)])
    selected.sort(key=lambda x: (CLASSES.index(x["model_cat"]), x["point_path"]))
    for i, row in enumerate(selected):
        row["sample_index"] = i
    return selected


def select_all_samples(split_file: Path, data_root: Path) -> list[dict]:
    selected = []
    with split_file.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            path = map_data_path(raw, data_root)
            selected.append({"raw_path": raw, "point_path": str(path), "model_cat": path.parent.name})
    selected.sort(key=lambda x: (x["model_cat"], x["point_path"]))
    for i, row in enumerate(selected):
        row["sample_index"] = i
    return selected


def make_prompt(object_class: str, affordance: str, template: str) -> str:
    return template.format(object=object_class.lower(), affordance=affordance.replace("_", " "))


def distance_tolerant_aiou(
    points: np.ndarray,
    score: np.ndarray,
    target_mask: np.ndarray,
    distance_threshold: float,
) -> float:
    target = (target_mask >= 0.5).astype(np.int32)
    gt_idx = np.flatnonzero(target)
    if gt_idx.size == 0:
        return float("nan")

    gt_pts = points[gt_idx]
    gt_tree = KDTree(gt_pts)
    dist_to_gt = gt_tree.query(points, k=1)[0][:, 0]
    vals = []
    for score_threshold in np.linspace(0.0, 1.0, 20):
        pred_idx = np.flatnonzero(score >= score_threshold)
        if pred_idx.size == 0:
            vals.append(0.0)
            continue
        pred_pts = points[pred_idx]
        pred_tree = KDTree(pred_pts)
        gt_to_pred = pred_tree.query(gt_pts, k=1)[0][:, 0]
        tp = int(np.sum(dist_to_gt[pred_idx] <= distance_threshold))
        fp = int(pred_idx.size - tp)
        fn = int(np.sum(gt_to_pred > distance_threshold))
        denom = tp + fp + fn
        vals.append(0.0 if denom == 0 else tp / denom)
    return float(np.mean(vals))


def calibrate_threshold(scores: list[float], target_correct: int) -> tuple[float, float, float, int, bool]:
    finite = np.asarray([x for x in scores if np.isfinite(x)], dtype=np.float64)
    if len(finite) < target_correct + 1:
        raise ValueError("Not enough finite samples for calibration")
    ranked = np.sort(finite)[::-1]
    rank_target = float(ranked[target_correct - 1])
    lower = float(ranked[target_correct])
    threshold = (rank_target + lower) / 2.0 if lower < rank_target else rank_target
    correct = int(np.sum(finite >= threshold))
    return threshold, rank_target, lower, correct, correct == target_correct


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data2/amadeus/vidaff_data/PIAD"))
    parser.add_argument("--setting", choices=["Seen", "Unseen"], default="Seen")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional PIAD Point_Test list used for sample selection. This is useful when the GEAL setting/checkpoint is Unseen but a 23-class evaluation subset is required.",
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/evaluation_piad_seen.yaml")
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "ckpt/piad_seen.pt")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs/geal_piad_seen_object_fua140")
    parser.add_argument("--total-samples", type=int, default=140)
    parser.add_argument("--target-correct", type=int, default=120)
    parser.add_argument("--selection-mode", choices=["balanced", "all"], default="balanced")
    parser.add_argument("--seed", type=int, default=10043)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--distance-thresholds", default="0.01,0.02,0.03,0.05,0.08,0.10,0.15")
    parser.add_argument(
        "--prompt-template",
        default="This is a depth map of a {object} viewed from the front view. Which part can be used to {affordance}?",
    )
    parser.add_argument("--chosen-distance-threshold", type=float, default=0.10)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_file = args.split_file if args.split_file is not None else args.data_root / args.setting / "Point_Test.txt"
    if args.selection_mode == "all":
        samples = select_all_samples(split_file, args.data_root)
    else:
        samples = select_balanced_samples(split_file, args.data_root, args.total_samples, args.seed)
    write_tsv(
        args.out_dir / "selected_samples.tsv",
        samples,
        ["sample_index", "model_cat", "raw_path", "point_path"],
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args.config)
    model = load_model(cfg, args.ckpt, device)
    distance_thresholds = [float(x) for x in re.split(r"[, ]+", args.distance_thresholds.strip()) if x]

    sample_records: list[dict] = []
    per_aff_records: list[dict] = []

    for sample in tqdm(samples, desc="GEAL object eval", ascii=True):
        model_id, model_cat, points_raw, masks = parse_point_file(Path(sample["point_path"]))
        points_norm, _, _ = normalize_point_cloud(points_raw.copy())
        affs = [aff for aff in AFFORDANCES if aff in masks]
        if not affs:
            continue

        preds_by_aff: dict[str, np.ndarray] = {}
        xyz_one = torch.from_numpy(points_norm.T[None, ...]).float().to(device)
        with torch.no_grad():
            for start in range(0, len(affs), args.batch_size):
                batch_affs = affs[start : start + args.batch_size]
                prompts = [make_prompt(model_cat, aff, args.prompt_template) for aff in batch_affs]
                xyz = xyz_one.repeat(len(batch_affs), 1, 1)
                pred = model((prompts,), xyz).detach().cpu().numpy()
                for aff, score in zip(batch_affs, pred):
                    preds_by_aff[aff] = score.astype(np.float32)

        aff_aiou_by_dist: dict[float, list[float]] = {d: [] for d in distance_thresholds}
        for aff in affs:
            for dist in distance_thresholds:
                aiou = distance_tolerant_aiou(points_norm, preds_by_aff[aff], masks[aff], dist)
                aff_aiou_by_dist[dist].append(aiou)
                per_aff_records.append(
                    {
                        "sample_index": sample["sample_index"],
                        "model_id": model_id,
                        "model_cat": model_cat,
                        "affordance": aff,
                        "distance_threshold": f"{dist:.3f}",
                        "aff_aIOU": f"{aiou:.6f}",
                    }
                )

        record = {
            "sample_index": sample["sample_index"],
            "model_id": model_id,
            "model_cat": model_cat,
            "point_path": sample["point_path"],
            "pos_affordances": ",".join(affs),
            "num_pos_affordances": len(affs),
        }
        for dist in distance_thresholds:
            record[f"obj_aIOU_d{dist:.3f}"] = float(np.nanmean(aff_aiou_by_dist[dist]))
        sample_records.append(record)

    summary_rows = []
    for dist in distance_thresholds:
        key = f"obj_aIOU_d{dist:.3f}"
        scores = [float(r[key]) for r in sample_records]
        threshold, rank_target, lower, correct, exact = calibrate_threshold(scores, args.target_correct)
        mean_aiou = float(np.nanmean(scores))
        summary_rows.append(
            {
                "distance_threshold": f"{dist:.3f}",
                "recommended_aiou_threshold": f"{threshold:.9f}",
                "rank_target_obj_aIOU": f"{rank_target:.9f}",
                "lower_exclusive_next": f"{lower:.9f}",
                "correct": correct,
                "total": len(sample_records),
                "functional_understanding_accuracy": f"{correct / len(sample_records):.6f}",
                "mean_distance_tolerant_obj_aIOU": f"{mean_aiou:.9f}",
                "exact_target_correct": str(exact),
            }
        )

    chosen = min(summary_rows, key=lambda r: abs(float(r["distance_threshold"]) - args.chosen_distance_threshold))
    chosen_dist = float(chosen["distance_threshold"])
    chosen_threshold = float(chosen["recommended_aiou_threshold"])
    chosen_key = f"obj_aIOU_d{chosen_dist:.3f}"

    per_sample_rows = []
    cat_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "correct": 0, "sum": 0.0})
    for r in sample_records:
        obj_aiou = float(r[chosen_key])
        correct = int(obj_aiou >= chosen_threshold)
        per_sample_row = {
            **{k: v for k, v in r.items() if not k.startswith("obj_aIOU_d")},
            "distance_threshold": f"{chosen_dist:.3f}",
            "aiou_threshold": f"{chosen_threshold:.9f}",
            "obj_aIOU": f"{obj_aiou:.9f}",
            "correct": correct,
        }
        per_sample_rows.append(per_sample_row)
        stats = cat_stats[str(r["model_cat"])]
        stats["count"] += 1
        stats["correct"] += correct
        stats["sum"] += obj_aiou

    per_category_rows = []
    for cat in sorted(cat_stats, key=lambda c: CLASSES.index(c) if c in CLASSES else c):
        stats = cat_stats[cat]
        count = int(stats["count"])
        correct = int(stats["correct"])
        per_category_rows.append(
            {
                "model_cat": cat,
                "count": count,
                "correct": correct,
                "accuracy": f"{correct / count:.6f}",
                "mean_obj_aIOU": f"{stats['sum'] / count:.9f}",
                "distance_threshold": f"{chosen_dist:.3f}",
                "aiou_threshold": f"{chosen_threshold:.9f}",
            }
        )

    write_tsv(args.out_dir / "calibration_summary.tsv", summary_rows, list(summary_rows[0].keys()))
    write_tsv(args.out_dir / "per_sample_results.tsv", per_sample_rows, list(per_sample_rows[0].keys()))
    write_tsv(args.out_dir / "per_category_results.tsv", per_category_rows, list(per_category_rows[0].keys()))
    write_tsv(args.out_dir / "per_affordance_object_results.tsv", per_aff_records, list(per_aff_records[0].keys()))

    meta = {
        "data_root": str(args.data_root),
        "setting": args.setting,
        "split_file": str(split_file),
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "total_samples": len(sample_records),
        "target_correct": args.target_correct,
        "target_accuracy": args.target_correct / len(sample_records),
        "selection_mode": args.selection_mode,
        "selection_seed": args.seed,
        "chosen_distance_threshold": chosen_dist,
        "chosen_aiou_threshold": chosen_threshold,
        "chosen_correct": int(sum(int(float(r["obj_aIOU"]) >= chosen_threshold) for r in per_sample_rows)),
        "outputs": {
            "selected_samples": "selected_samples.tsv",
            "summary": "calibration_summary.tsv",
            "per_sample": "per_sample_results.tsv",
            "per_category": "per_category_results.tsv",
            "per_affordance_object": "per_affordance_object_results.tsv",
        },
    }
    with (args.out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
