#!/usr/bin/env python3
"""
Run GEAL Branch3D on one custom point cloud.

The official GEAL evaluation code is dataset-oriented: each sample is paired
with one affordance question. This wrapper keeps the pretrained model intact
and runs a fixed affordance vocabulary over a single PLY.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
import torch
import yaml
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.data_utils import AFFORDANCES, normalize_point_cloud  # noqa: E402
from model.branch_3d import Branch3D  # noqa: E402


DEFAULT_COLORS = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
        [0, 128, 128],
        [220, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
    ],
    dtype=np.uint8,
)


def load_object_affordance_map(path: Path) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Object-affordance map not found: {path}")
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row["object_label"].strip().lower()
            affs = [x.strip() for x in row["affordances"].split(",") if x.strip()]
            invalid = [x for x in affs if x not in AFFORDANCES]
            if invalid:
                raise ValueError(f"Invalid affordance(s) for {label}: {invalid}")
            table[label] = affs
    return table


def parse_affordances(value: str, object_class: str, map_path: Path) -> tuple[list[str], str, bool]:
    if value.lower() == "auto":
        table = load_object_affordance_map(map_path)
        key = object_class.strip().lower()
        if key not in table:
            print(
                f"[warn] No affordance preset for object label '{object_class}' in {map_path}; "
                "falling back to --affordances all and selecting top-k classes.",
                file=sys.stderr,
            )
            return list(AFFORDANCES), "auto_fallback_all", True
        return table[key], "auto_map", False
    if value.lower() == "all":
        return list(AFFORDANCES), "all", False
    names = [x.strip() for x in value.split(",") if x.strip()]
    invalid = [x for x in names if x not in AFFORDANCES]
    if invalid:
        raise ValueError(f"Unknown affordance(s): {invalid}. Valid: {AFFORDANCES}")
    return names, "manual", False


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"No valid XYZ points found in {path}")
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        points = points[finite]
        colors = colors[finite] if colors is not None else None
    return points, colors


def random_or_fps_sample(
    points: np.ndarray,
    num_points: int,
    seed: int,
    method: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(points)
    if n <= 0:
        raise ValueError("Cannot sample an empty point cloud")
    if n < num_points:
        extra = rng.choice(n, size=num_points - n, replace=True)
        return np.concatenate([np.arange(n), extra]).astype(np.int64)
    if method == "random":
        return rng.choice(n, size=num_points, replace=False).astype(np.int64)

    # CPU FPS is deterministic and preserves object coverage, but it is O(N*K).
    first = int(rng.integers(0, n))
    indices = np.empty(num_points, dtype=np.int64)
    indices[0] = first
    dist = np.full(n, np.inf, dtype=np.float32)
    current = points[first]
    for i in range(1, num_points):
        d = np.sum((points - current) ** 2, axis=1)
        dist = np.minimum(dist, d)
        next_idx = int(np.argmax(dist))
        indices[i] = next_idx
        current = points[next_idx]
    return indices


def interpolate_scores(
    sampled_points: np.ndarray,
    sampled_scores: np.ndarray,
    target_points: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    if len(sampled_points) == len(target_points) and np.allclose(sampled_points, target_points):
        return sampled_scores
    tree = cKDTree(sampled_points)
    k = max(1, min(k, len(sampled_points)))
    dist, idx = tree.query(target_points, k=k)
    if k == 1:
        return sampled_scores[idx]
    dist = np.asarray(dist, dtype=np.float32)
    idx = np.asarray(idx, dtype=np.int64)
    weights = 1.0 / np.maximum(dist, 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(sampled_scores[idx] * weights[..., None], axis=1)


def make_prompt(object_class: str, affordance: str, template: str) -> str:
    clean_aff = affordance.replace("_", " ")
    return template.format(object=object_class, affordance=clean_aff)


def write_heatmap_ply(path: Path, points: np.ndarray, score: np.ndarray) -> None:
    score = np.clip(score.reshape(-1, 1), 0.0, 1.0)
    back = np.array([0.72, 0.72, 0.72], dtype=np.float32)
    hot = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    colors = back * (1.0 - score) + hot * score
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def write_argmax_ply(
    path: Path,
    points: np.ndarray,
    scores: np.ndarray,
    colors: np.ndarray,
    confidence_min: float,
) -> np.ndarray:
    best = scores.argmax(axis=1)
    conf = scores.max(axis=1)
    rgb = colors[best].astype(np.float32) / 255.0
    low = conf < confidence_min
    if np.any(low):
        rgb[low] = np.array([0.55, 0.55, 0.55], dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    return best


def write_label_ply(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray,
) -> None:
    rgb = np.full((len(points), 3), 0.55, dtype=np.float32)
    valid = labels >= 0
    if np.any(valid):
        rgb[valid] = colors[labels[valid]].astype(np.float32) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def write_color_map(path: Path, aff_names: list[str], colors: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("affordance_id\taffordance\tr\tg\tb\thex\n")
        for i, (name, rgb) in enumerate(zip(aff_names, colors)):
            r, g, b = [int(x) for x in rgb]
            f.write(f"{i}\t{name}\t{r}\t{g}\t{b}\t#{r:02x}{g:02x}{b:02x}\n")


def filter_affordances(
    names: list[str],
    scores: np.ndarray,
    colors: np.ndarray,
    min_p95: float,
    top_k: int,
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict]]:
    stats = []
    keep = []
    for j, name in enumerate(names):
        score = scores[:, j]
        item = {
            "affordance": name,
            "mean_score": float(score.mean()),
            "max_score": float(score.max()),
            "p95_score": float(np.percentile(score, 95)),
        }
        item["kept"] = item["p95_score"] >= min_p95
        stats.append(item)
        keep.append(item["kept"])

    keep_arr = np.asarray(keep, dtype=bool)
    if not keep_arr.any():
        # Avoid producing an empty result; keep the globally strongest affordance.
        best = int(np.argmax([x["p95_score"] for x in stats]))
        keep_arr[best] = True
        stats[best]["kept"] = True
        stats[best]["forced_keep_reason"] = "highest_p95_when_all_below_threshold"

    selected_idx = np.flatnonzero(keep_arr)
    if top_k > 0 and len(selected_idx) > top_k:
        ranked = sorted(selected_idx.tolist(), key=lambda idx: stats[idx]["p95_score"], reverse=True)
        selected_set = set(ranked[:top_k])
        for idx in selected_idx:
            if idx not in selected_set:
                keep_arr[idx] = False
                stats[idx]["kept"] = False
                stats[idx]["drop_reason"] = f"outside_top_{top_k}"
        selected_idx = np.asarray(ranked[:top_k], dtype=np.int64)
    else:
        selected_idx = np.flatnonzero(keep_arr)

    kept_names = [names[idx] for idx in selected_idx]
    return kept_names, scores[:, selected_idx], colors[selected_idx], stats


def auto_component_eps(points: np.ndarray, k: int = 8, multiplier: float = 3.0) -> float:
    k = max(2, min(k + 1, len(points)))
    dist, _ = cKDTree(points).query(points, k=k)
    kth = dist[:, -1]
    eps = float(np.median(kth) * multiplier)
    return max(eps, 1e-6)


def connected_components_radius(points: np.ndarray, eps: float) -> list[np.ndarray]:
    neighbors = cKDTree(points).query_ball_point(points, r=eps)
    visited = np.zeros(len(points), dtype=bool)
    comps: list[np.ndarray] = []
    for start in range(len(points)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in neighbors[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        comps.append(np.asarray(comp, dtype=np.int64))
    return comps


def clean_label_components(
    points_for_connectivity: np.ndarray,
    labels: np.ndarray,
    aff_names: list[str],
    eps: float,
    min_component_points: int,
    min_component_ratio: float,
    keep_components: int,
) -> tuple[np.ndarray, list[dict]]:
    cleaned = labels.copy()
    stats: list[dict] = []
    for aff_id, aff_name in enumerate(aff_names):
        idx = np.flatnonzero(labels == aff_id)
        if len(idx) == 0:
            stats.append(
                {
                    "affordance": aff_name,
                    "raw_points": 0,
                    "kept_points": 0,
                    "removed_points": 0,
                    "component_sizes": [],
                }
            )
            continue

        comps_local = connected_components_radius(points_for_connectivity[idx], eps)
        comps = [idx[c] for c in comps_local]
        comps.sort(key=len, reverse=True)
        min_size = max(int(min_component_points), int(np.ceil(len(idx) * min_component_ratio)))

        keep_mask = np.zeros(len(comps), dtype=bool)
        for rank, comp in enumerate(comps):
            if len(comp) < min_size:
                continue
            if keep_components > 0 and rank >= keep_components:
                continue
            keep_mask[rank] = True

        kept_idx = np.concatenate([comp for comp, keep in zip(comps, keep_mask) if keep]) if keep_mask.any() else np.empty(0, dtype=np.int64)
        remove_idx = np.setdiff1d(idx, kept_idx, assume_unique=False)
        cleaned[remove_idx] = -1
        stats.append(
            {
                "affordance": aff_name,
                "raw_points": int(len(idx)),
                "kept_points": int(len(kept_idx)),
                "removed_points": int(len(remove_idx)),
                "min_component_size": int(min_size),
                "component_sizes": [int(len(c)) for c in comps],
            }
        )
    return cleaned, stats


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    cfg["model_3d"]["training"] = False
    return cfg


def load_model(cfg: dict, ckpt_path: Path, device: torch.device) -> Branch3D:
    model = Branch3D(cfg["model_3d"])
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    status = model.load_state_dict(state, strict=False)
    if status.missing_keys:
        print(f"[warn] missing keys: {status.missing_keys[:10]} ...", file=sys.stderr)
    if status.unexpected_keys:
        print(f"[warn] unexpected keys: {status.unexpected_keys[:10]} ...", file=sys.stderr)
    model.to(device)
    model.eval()
    return model


def batch_iter(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="GEAL custom point-cloud inference")
    parser.add_argument("--input", required=True, type=Path, help="Input object PLY")
    parser.add_argument("--ckpt", required=True, type=Path, help="GEAL checkpoint .pt")
    parser.add_argument("--config", default=REPO_ROOT / "config/evaluation.yaml", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--object-class", default="object")
    parser.add_argument(
        "--affordances",
        default="auto",
        help="'auto', 'all', or comma-separated names. auto uses --object-class and --object-affordance-map.",
    )
    parser.add_argument(
        "--object-affordance-map",
        default=REPO_ROOT / "config/object_affordance_map.tsv",
        type=Path,
    )
    parser.add_argument("--num-points", default=2048, type=int)
    parser.add_argument("--sampling", choices=["random", "fps"], default="random")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default=4, type=int, help="Affordance prompt batch size")
    parser.add_argument("--interp-k", default=3, type=int)
    parser.add_argument(
        "--output-points",
        choices=["sampled", "original"],
        default="sampled",
        help="sampled: save only model input points; original: KNN-interpolate scores back to the original cloud.",
    )
    parser.add_argument(
        "--min-affordance-p95",
        default=0.35,
        type=float,
        help="Drop affordance classes whose 95th percentile score is below this value.",
    )
    parser.add_argument(
        "--top-k-affordances",
        default=3,
        type=int,
        help="Keep only the top-k affordance classes by p95 score after confidence filtering. <=0 disables this limit.",
    )
    parser.add_argument("--confidence-min", default=0.0, type=float)
    parser.add_argument(
        "--clean-components",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove small disconnected argmax regions and color them gray.",
    )
    parser.add_argument(
        "--component-eps",
        default=0.0,
        type=float,
        help="Radius for connected components in normalized coordinates. <=0 uses an auto radius.",
    )
    parser.add_argument("--component-eps-multiplier", default=3.0, type=float)
    parser.add_argument("--min-component-points", default=30, type=int)
    parser.add_argument("--min-component-ratio", default=0.03, type=float)
    parser.add_argument(
        "--keep-components",
        default=1,
        type=int,
        help="Largest connected components to keep per affordance. 1 enforces one continuous region; 0 keeps all components above size threshold.",
    )
    parser.add_argument(
        "--prompt-template",
        default="This is a depth map of a {object} viewed from the front view. Which part can be used to {affordance}?",
        help="Must contain {object} and {affordance}. Keep one '.' before the question for GEAL parsing.",
    )
    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--save-argmax", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.input.stem
    aff_names, affordance_source, auto_fallback_to_all = parse_affordances(
        args.affordances,
        args.object_class,
        args.object_affordance_map,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    points, input_colors = load_points(args.input)
    norm_points, centroid, scale = normalize_point_cloud(points.copy())
    sample_idx = random_or_fps_sample(norm_points, args.num_points, args.seed, args.sampling)
    sampled_norm = norm_points[sample_idx]
    sampled_orig = points[sample_idx]

    cfg = load_config(args.config)
    model = load_model(cfg, args.ckpt, device)

    all_sample_scores: list[np.ndarray] = []
    with torch.no_grad():
        xyz = torch.from_numpy(sampled_norm.T[None, ...]).float().to(device)
        for names in batch_iter(aff_names, args.batch_size):
            prompts = [make_prompt(args.object_class, aff, args.prompt_template) for aff in names]
            # Branch3D uses text[0] as the active batch of language prompts.
            pred = model((prompts,), xyz.repeat(len(prompts), 1, 1))
            all_sample_scores.append(pred.detach().cpu().numpy())

    sample_scores = np.concatenate(all_sample_scores, axis=0).T  # [M, A]
    if args.output_points == "original":
        output_points = points
        output_norm_points = norm_points
        output_scores = interpolate_scores(sampled_orig, sample_scores, points, k=args.interp_k)
    else:
        output_points = sampled_orig
        output_norm_points = sampled_norm
        output_scores = sample_scores

    color_count = len(aff_names)
    colors = DEFAULT_COLORS
    if color_count > len(colors):
        reps = int(np.ceil(color_count / len(colors)))
        colors = np.tile(colors, (reps, 1))
    colors = colors[:color_count]
    kept_aff_names, output_scores, colors, filter_stats = filter_affordances(
        aff_names,
        output_scores,
        colors,
        args.min_affordance_p95,
        args.top_k_affordances,
    )
    color_map_path = args.out_dir / f"{name}_geal_color_map.tsv"
    write_color_map(color_map_path, kept_aff_names, colors)

    npz_path = args.out_dir / f"{name}_geal_scores.npz"

    argmax_path = None
    best = output_scores.argmax(axis=1)
    conf = output_scores.max(axis=1)
    labels = best.astype(np.int64)
    labels[conf < args.confidence_min] = -1
    component_eps = None
    component_stats = []
    if args.clean_components:
        component_eps = args.component_eps if args.component_eps > 0 else auto_component_eps(
            output_norm_points,
            multiplier=args.component_eps_multiplier,
        )
        labels, component_stats = clean_label_components(
            output_norm_points,
            labels,
            kept_aff_names,
            component_eps,
            args.min_component_points,
            args.min_component_ratio,
            args.keep_components,
        )
    np.savez_compressed(
        npz_path,
        points=output_points.astype(np.float32),
        scores=output_scores.astype(np.float32),
        argmax_labels=labels.astype(np.int64),
        sample_indices=sample_idx.astype(np.int64),
        affordance_names=np.asarray(kept_aff_names),
        output_points_mode=args.output_points,
        num_original_points=np.asarray(len(points), dtype=np.int64),
        num_model_points=np.asarray(len(sampled_orig), dtype=np.int64),
        input_colors=input_colors if input_colors is not None else np.empty((0, 3), dtype=np.uint8),
        normalize_centroid=centroid.astype(np.float32),
        normalize_scale=np.asarray(scale, dtype=np.float32),
    )
    if args.save_argmax:
        argmax_path = args.out_dir / f"{name}_geal_argmax.ply"
        if args.clean_components:
            write_label_ply(argmax_path, output_points, labels, colors)
            best = labels
        else:
            best = write_argmax_ply(argmax_path, output_points, output_scores, colors, args.confidence_min)

    heatmap_paths = []
    if args.save_heatmaps:
        heat_dir = args.out_dir / f"{name}_heatmaps"
        heat_dir.mkdir(parents=True, exist_ok=True)
        for j, aff in enumerate(kept_aff_names):
            heat_path = heat_dir / f"{name}_{j:02d}_{aff}.ply"
            heat_score = output_scores[:, j]
            if args.clean_components:
                heat_score = np.where(labels == j, heat_score, 0.0)
            write_heatmap_ply(heat_path, output_points, heat_score)
            heatmap_paths.append(str(heat_path))

    top_affordances = []
    for j, aff in enumerate(kept_aff_names):
        score = output_scores[:, j]
        top_affordances.append(
            {
                "affordance": aff,
                "mean_score": float(score.mean()),
                "max_score": float(score.max()),
                "p95_score": float(np.percentile(score, 95)),
                "argmax_point_count": int(np.sum(labels == j)),
                "color_rgb": colors[j].tolist(),
            }
        )

    meta = {
        "input": str(args.input),
        "checkpoint": str(args.ckpt),
        "config": str(args.config),
        "object_class": args.object_class,
        "requested_affordances": args.affordances,
        "affordance_source": affordance_source,
        "auto_fallback_to_all": bool(auto_fallback_to_all),
        "object_affordance_map": str(args.object_affordance_map),
        "num_original_points": int(len(points)),
        "num_model_points": int(len(sampled_norm)),
        "num_output_points": int(len(output_points)),
        "output_points_mode": args.output_points,
        "sampling": args.sampling,
        "min_affordance_p95": float(args.min_affordance_p95),
        "top_k_affordances": int(args.top_k_affordances),
        "clean_components": bool(args.clean_components),
        "component_eps": component_eps,
        "min_component_points": int(args.min_component_points),
        "min_component_ratio": float(args.min_component_ratio),
        "keep_components": int(args.keep_components),
        "component_stats": component_stats,
        "score_npz": str(npz_path),
        "argmax_ply": str(argmax_path) if argmax_path is not None else None,
        "color_map_tsv": str(color_map_path),
        "heatmap_plys": heatmap_paths,
        "affordance_names": kept_aff_names,
        "dropped_affordance_stats": [x for x in filter_stats if not x["kept"]],
        "all_affordance_filter_stats": filter_stats,
        "top_affordance_stats": top_affordances,
        "note": "scores shape is [num_output_points, num_kept_affordances]; scores are sigmoid activations, not mutually exclusive probabilities.",
    }
    meta_path = args.out_dir / f"{name}_geal_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] scores: {npz_path}")
    print(f"[done] meta:   {meta_path}")
    if argmax_path:
        print(f"[done] argmax: {argmax_path}")


if __name__ == "__main__":
    main()
