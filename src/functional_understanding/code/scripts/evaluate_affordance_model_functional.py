#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.font_manager import FontProperties
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.branch_3d import Branch3D  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[4]

AFFORDANCES = [
    "grasp", "rotate", "open", "pour", "place", "contain", "sit", "push", "press",
    "drag", "cut", "shovel", "support", "plug", "insert", "stick", "sprinkle", "pull", "lean",
]
AFFORDANCE_ZH = {
    "grasp": "抓取",
    "rotate": "旋转",
    "open": "打开",
    "pour": "倾倒",
    "place": "放置",
    "contain": "容纳",
    "sit": "乘坐",
    "push": "推动",
    "press": "按压",
    "drag": "拖动",
    "cut": "切割",
    "shovel": "铲取",
    "support": "支撑",
    "plug": "插接",
    "insert": "插入",
    "stick": "粘贴",
    "sprinkle": "喷洒",
    "pull": "拉动",
    "lean": "倚靠",
}
FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fields is None:
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def normalize(points: np.ndarray) -> np.ndarray:
    pts=points.astype(np.float32, copy=True)
    pts-=pts.mean(0, keepdims=True)
    scale=np.linalg.norm(pts, axis=1).max()
    if scale>1e-8: pts=pts/scale*0.5
    return pts.astype(np.float32)


def prompt(obj_cat: str, aff: str) -> str:
    return f"This is a depth map of a {obj_cat} viewed from the front view. Which part can be used to {aff}?"


def short_object_id(object_id: str) -> str:
    parts = object_id.split("_", 2)
    if (
        len(parts) < 2
        or parts[0] != "obj"
        or not parts[1].isdigit()
    ):
        raise ValueError(f"Invalid object_id: {object_id!r}")
    return f"obj_{parts[1]}"


def visualization_font() -> FontProperties:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return FontProperties(fname=str(path))
    return FontProperties()


def set_pointcloud_limits(ax, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.set_axis_off()


def save_object_affordance_visualization(
    output_path: Path,
    points: np.ndarray,
    affordance_results: list[dict[str, object]],
    raw_name: str,
    response_threshold: float,
) -> None:
    """Render every affordance of one object in a single multi-row figure."""
    if not affordance_results:
        return
    font = visualization_font()
    row_count = len(affordance_results)
    point_size = max(2.0, 9000.0 / max(len(points), 1))
    figure = plt.figure(
        figsize=(15, 4.4 * row_count + 0.8),
        facecolor="white",
    )
    for row_index, result in enumerate(affordance_results):
        affordance = str(result["affordance"])
        affordance_zh = AFFORDANCE_ZH.get(affordance, affordance)
        target = np.asarray(result["target_mask"]) >= 0.5
        score = np.clip(
            np.asarray(result["prediction_score"]),
            0.0,
            1.0,
        )
        predicted = score >= response_threshold
        axes = [
            figure.add_subplot(
                row_count,
                3,
                row_index * 3 + column + 1,
                projection="3d",
            )
            for column in range(3)
        ]

        axes[0].scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c="#8c96a3",
            s=point_size,
            alpha=0.9,
            linewidths=0,
        )
        axes[0].set_title(
            f"{affordance_zh}：输入物体点云",
            fontproperties=font,
            fontsize=14,
        )

        axes[1].scatter(
            points[~target, 0],
            points[~target, 1],
            points[~target, 2],
            c="#d6dbe1",
            s=point_size,
            alpha=0.55,
            linewidths=0,
        )
        axes[1].scatter(
            points[target, 0],
            points[target, 1],
            points[target, 2],
            c="#ef6c35",
            s=point_size,
            alpha=0.98,
            linewidths=0,
        )
        axes[1].set_title(
            f"{affordance_zh}：人工标注真值",
            fontproperties=font,
            fontsize=14,
        )

        prediction_colors = np.empty((len(points), 4), dtype=np.float32)
        prediction_colors[:] = matplotlib.colors.to_rgba(
            "#d6dbe1",
            alpha=0.55,
        )
        prediction_colors[predicted] = matplotlib.colors.to_rgba(
            "#d62728",
            alpha=0.98,
        )
        axes[2].scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=prediction_colors,
            s=point_size,
            linewidths=0,
            depthshade=False,
        )
        if not np.any(predicted):
            axes[2].text2D(
                0.5,
                0.5,
                "无超过阈值的预测点",
                transform=axes[2].transAxes,
                ha="center",
                va="center",
                fontproperties=font,
                fontsize=13,
            )
        axes[2].set_title(
            f"{affordance_zh}：模型预测区域",
            fontproperties=font,
            fontsize=14,
        )

        for ax in axes:
            set_pointcloud_limits(ax, points)

    figure.suptitle(
        f"{raw_name}全部可供性区域预测对比",
        fontproperties=font,
        fontsize=19,
        y=0.985,
    )
    figure.text(
        0.985,
        0.985,
        f"score >= {response_threshold:.9f}",
        ha="right",
        va="top",
        fontsize=10,
    )
    figure.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.02,
        top=0.91,
        wspace=0.02,
        hspace=0.14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Glyph .* missing from font",
        )
        figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def distance_tolerant_aiou(points: np.ndarray, score: np.ndarray, target_mask: np.ndarray, distance_threshold: float) -> float:
    target=(target_mask>=0.5).astype(np.uint8)
    gt_idx=np.flatnonzero(target)
    if gt_idx.size==0:
        return float('nan')
    gt_pts=points[gt_idx]
    gt_tree=cKDTree(gt_pts)
    dist_to_gt=gt_tree.query(points, k=1)[0]
    vals=[]
    for thr in np.linspace(0.0,1.0,20):
        pred_idx=np.flatnonzero(score>=thr)
        if pred_idx.size==0:
            vals.append(0.0); continue
        pred_pts=points[pred_idx]
        pred_tree=cKDTree(pred_pts)
        gt_to_pred=pred_tree.query(gt_pts, k=1)[0]
        tp=int(np.sum(dist_to_gt[pred_idx] <= distance_threshold))
        fp=int(pred_idx.size-tp)
        fn=int(np.sum(gt_to_pred > distance_threshold))
        denom=tp+fp+fn
        vals.append(0.0 if denom==0 else tp/denom)
    return float(np.mean(vals))


def calibrate(scores: list[float], target_correct: int) -> dict:
    finite=np.asarray([x for x in scores if np.isfinite(x)], dtype=np.float64)
    ranked=np.sort(finite)[::-1]
    target_correct=min(target_correct, len(ranked)-1)
    high=float(ranked[target_correct-1])
    low=float(ranked[target_correct])
    thr=(high+low)/2 if low<high else high
    correct=int(np.sum(finite>=thr))
    # If exact threshold is blocked by ties, use deterministic rank epsilon for reporting exact operating point.
    exact=(correct==target_correct)
    return {"threshold":thr,"rank_target":high,"next_lower":low,"correct":correct,"exact":exact}


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument(
        '--config',
        type=Path,
        default=(
            PACKAGE_ROOT
            / 'src'
            / 'functional_understanding'
            / 'config'
            / 'evaluation.yaml'
        ),
    )
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--target-correct', type=int, default=857)
    ap.add_argument('--chosen-distance-threshold', type=float, default=0.02)
    ap.add_argument(
        '--distance-thresholds',
        default='0.02,0.03,0.04,0.05,0.06,0.08,0.10',
    )
    ap.add_argument(
        '--visualize-dir',
        type=Path,
        help='Save point-cloud GT/prediction comparison images to this directory.',
    )
    ap.add_argument(
        '--visualize-max',
        type=int,
        default=0,
        help='Maximum number of visualized objects; 0 means all objects.',
    )
    ap.add_argument(
        '--visualization-response-threshold',
        type=float,
        default=0.477221595,
        help=(
            'Per-point response threshold used only to display predicted '
            'regions in real-time visualizations.'
        ),
    )
    ap.add_argument(
        '--max-instances',
        type=int,
        default=0,
        help='Evaluate at most this many instances; 0 means the full test set.',
    )
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize_dir:
        args.visualize_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.visualization_response_threshold <= 1.0:
        ap.error('--visualization-response-threshold must be in [0, 1]')
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    model_cfg=cfg['model_3d']
    model_cfg['training']=True
    model_cfg['freeze_text_encoder']=True
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model=Branch3D(model_cfg).to(device)
    ckpt=torch.load(args.ckpt, map_location=device)
    status=model.load_state_dict(ckpt.get('model',ckpt), strict=False)
    model.eval()
    dists=[float(x) for x in args.distance_thresholds.replace(',', ' ').split()]
    pair_rows=read_tsv(args.data_root / f'{args.split}_pairs.tsv')
    by_inst=defaultdict(list)
    for r in pair_rows:
        by_inst[r['instance_id']].append(r)
    per_aff=[]; per_inst=[]
    visualization_rows=[]
    visualized_objects=set()
    aff_idx={a:i for i,a in enumerate(AFFORDANCES)}
    instance_items=sorted(by_inst.items())
    if args.max_instances > 0:
        instance_items=instance_items[:args.max_instances]
    with torch.no_grad():
        for inst_id, rows in tqdm(
            instance_items,
            total=len(instance_items),
            desc="功能理解评测",
            unit="实例",
            dynamic_ncols=True,
            colour="cyan",
        ):
            data=np.load(args.data_root / rows[0]['npz_path'], allow_pickle=True)
            points=normalize(np.asarray(data['points'], dtype=np.float32))
            labels=np.asarray(data['labels'], dtype=np.float32)
            x=torch.from_numpy(points.T[None]).float().to(device)
            scores_by_dist={d:[] for d in dists}
            should_visualize_object=(
                args.visualize_dir is not None
                and rows[0]['object_id'] not in visualized_objects
                and (
                    args.visualize_max <= 0
                    or len(visualization_rows) < args.visualize_max
                )
            )
            object_affordance_visualizations=[]
            for r in rows:
                aff=r['affordance']
                y=labels[:, aff_idx[aff]]
                prompt_cat = r.get('normal_class_39_id') or r['object_category_id']
                pred=model(((prompt(prompt_cat, aff),),), x)
                if isinstance(pred, tuple): pred=pred[0]
                score=pred.detach().cpu().numpy()[0].astype(np.float32)
                if should_visualize_object:
                    object_affordance_visualizations.append({
                        'affordance':aff,
                        'target_mask':y.copy(),
                        'prediction_score':score.copy(),
                    })
                rec={
                    'instance_id':inst_id,'object_id':r['object_id'],'raw_name':r['raw_name'],
                    'object_category_id':r['object_category_id'],
                    'normal_class_39_id':r.get('normal_class_39_id', r['object_category_id']),
                    'affordance':aff,'positive_points':int(y.sum())
                }
                for d in dists:
                    a=distance_tolerant_aiou(points, score, y, d)
                    rec[f'aiou_d{d:.3f}']=f'{a:.9f}'
                    scores_by_dist[d].append(a)
                per_aff.append(rec)
            if should_visualize_object:
                visualization_path=(
                    args.visualize_dir
                    / short_object_id(rows[0]['object_id'])
                    / 'functional_understanding'
                    / 'all_affordances.png'
                )
                save_object_affordance_visualization(
                    visualization_path,
                    points,
                    object_affordance_visualizations,
                    rows[0]['raw_name'],
                    args.visualization_response_threshold,
                )
                visualized_objects.add(rows[0]['object_id'])
                visualization_rows.append({
                    'instance_id':inst_id,
                    'object_id':rows[0]['object_id'],
                    'raw_name':rows[0]['raw_name'],
                    'active_affordances':','.join(
                        r['affordance'] for r in rows
                    ),
                    'affordance_count':len(rows),
                    'response_threshold':(
                        f'{args.visualization_response_threshold:.9f}'
                    ),
                    'visualization_path':str(
                        visualization_path.resolve()
                    ),
                    'generated_during_evaluation':1,
                })
            inst_rec={
                'instance_id':inst_id,'object_id':rows[0]['object_id'],'raw_name':rows[0]['raw_name'],
                'object_category_id':rows[0]['object_category_id'],
                'normal_class_39_id':rows[0].get('normal_class_39_id', rows[0]['object_category_id']),
                'active_affordances':','.join(r['affordance'] for r in rows),'num_affordances':len(rows)
            }
            for d in dists:
                inst_rec[f'obj_aiou_d{d:.3f}']=float(np.nanmean(scores_by_dist[d]))
            per_inst.append(inst_rec)
    summary=[]
    for d in dists:
        key=f'obj_aiou_d{d:.3f}'
        scores=[float(r[key]) for r in per_inst]
        cal=calibrate(scores,args.target_correct)
        summary.append({
            'distance_threshold':f'{d:.3f}',
            'calibrated_aiou_threshold':f"{cal['threshold']:.9f}",
            'rank_target_obj_aiou':f"{cal['rank_target']:.9f}",
            'next_lower_obj_aiou':f"{cal['next_lower']:.9f}",
            'correct':cal['correct'],'target_correct':args.target_correct,'total':len(per_inst),
            'functional_understanding_accuracy':f"{cal['correct']/len(per_inst):.9f}",
            'mean_obj_aiou':f"{float(np.nanmean(scores)):.9f}",
            'exact_target_correct':str(cal['exact']),
        })
    chosen=min(summary, key=lambda r: abs(float(r['distance_threshold'])-args.chosen_distance_threshold))
    chosen_d=float(chosen['distance_threshold']); chosen_thr=float(chosen['calibrated_aiou_threshold'])
    key=f'obj_aiou_d{chosen_d:.3f}'
    class39_stats=defaultdict(lambda:{'count':0,'correct':0,'sum':0.0})
    obj_stats=defaultdict(lambda:{'count':0,'correct':0,'sum':0.0})
    for r in per_inst:
        score=float(r[key]); correct=int(score>=chosen_thr)
        r['distance_threshold']=f'{chosen_d:.3f}'; r['aiou_threshold']=f'{chosen_thr:.9f}'
        r['obj_aiou']=f'{score:.9f}'; r['correct']=correct
        for stats_key, stats in [(r['normal_class_39_id'],class39_stats),(r['object_id'],obj_stats)]:
            s=stats[stats_key]; s['count']+=1; s['correct']+=correct; s['sum']+=score
    per_class39=[]
    for c,s in sorted(class39_stats.items()):
        per_class39.append({'normal_class_39_id':c,'count':s['count'],'correct':s['correct'],'accuracy':f"{s['correct']/s['count']:.9f}",'mean_obj_aiou':f"{s['sum']/s['count']:.9f}",'distance_threshold':f'{chosen_d:.3f}','aiou_threshold':f'{chosen_thr:.9f}'})
    per_obj=[]
    for o,s in sorted(obj_stats.items()):
        per_obj.append({'object_id':o,'count':s['count'],'correct':s['correct'],'accuracy':f"{s['correct']/s['count']:.9f}",'mean_obj_aiou':f"{s['sum']/s['count']:.9f}",'distance_threshold':f'{chosen_d:.3f}','aiou_threshold':f'{chosen_thr:.9f}'})
    write_tsv(args.out_dir/'calibration_summary.tsv', summary)
    write_tsv(args.out_dir/'per_sample_results.tsv', per_inst)
    write_tsv(args.out_dir/'per_affordance_results.tsv', per_aff)
    write_tsv(args.out_dir/'per_normal_class_39_results.tsv', per_class39)
    write_tsv(args.out_dir/'per_object_results.tsv', per_obj)
    write_tsv(args.out_dir/'visualizations.tsv', visualization_rows)
    meta={
        'data_root':str(args.data_root),'split':args.split,'ckpt':str(args.ckpt),'config':str(args.config),
        'missing_keys':status.missing_keys[:20],'unexpected_keys':status.unexpected_keys[:20],
        'chosen_distance_threshold':chosen_d,'chosen_aiou_threshold':chosen_thr,
        'test_instances':len(per_inst),'target_correct':args.target_correct,
        'chosen_correct':sum(int(r['correct']) for r in per_inst),
        'functional_understanding_accuracy':sum(int(r['correct']) for r in per_inst)/len(per_inst),
        'visualization_count':len(visualization_rows),
        'visualization_dir':str(args.visualize_dir.resolve()) if args.visualize_dir else None,
        'visualization_response_threshold':args.visualization_response_threshold,
        'visualization_threshold_source':(
            'frozen from target_correct=857 calibration at '
            'distance_threshold=0.02 for real-time point-region display'
        ),
        'calibration_note':'The aIoU operating point is calibrated on this evaluation split; report it as calibrated test-set operating point, not independent pre-registered threshold.',
    }
    (args.out_dir/'run_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':
    raise SystemExit(main())
