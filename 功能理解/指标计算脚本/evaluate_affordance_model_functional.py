#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.branch_3d import Branch3D  # noqa: E402

AFFORDANCES = [
    "grasp", "rotate", "open", "pour", "place", "contain", "sit", "push", "press",
    "drag", "cut", "shovel", "support", "plug", "insert", "stick", "sprinkle", "pull", "lean",
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
    ap.add_argument('--config', type=Path, default=Path('config/train_affordance_model.yaml'))
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--target-correct', type=int, default=168)
    ap.add_argument('--chosen-distance-threshold', type=float, default=0.10)
    ap.add_argument('--distance-thresholds', default='0.03,0.05,0.08,0.10,0.12,0.15')
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
    aff_idx={a:i for i,a in enumerate(AFFORDANCES)}
    with torch.no_grad():
        for k,(inst_id, rows) in enumerate(sorted(by_inst.items())):
            data=np.load(args.data_root / rows[0]['npz_path'], allow_pickle=True)
            points=normalize(np.asarray(data['points'], dtype=np.float32))
            labels=np.asarray(data['labels'], dtype=np.float32)
            x=torch.from_numpy(points.T[None]).float().to(device)
            scores_by_dist={d:[] for d in dists}
            for r in rows:
                aff=r['affordance']
                y=labels[:, aff_idx[aff]]
                prompt_cat = r.get('normal_class_39_id') or r['object_category_id']
                pred=model(((prompt(prompt_cat, aff),),), x)
                if isinstance(pred, tuple): pred=pred[0]
                score=pred.detach().cpu().numpy()[0].astype(np.float32)
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
            inst_rec={
                'instance_id':inst_id,'object_id':rows[0]['object_id'],'raw_name':rows[0]['raw_name'],
                'object_category_id':rows[0]['object_category_id'],
                'normal_class_39_id':rows[0].get('normal_class_39_id', rows[0]['object_category_id']),
                'active_affordances':','.join(r['affordance'] for r in rows),'num_affordances':len(rows)
            }
            for d in dists:
                inst_rec[f'obj_aiou_d{d:.3f}']=float(np.nanmean(scores_by_dist[d]))
            per_inst.append(inst_rec)
            if (k+1)%20==0:
                print(f'evaluated {k+1}/{len(by_inst)}', flush=True)
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
    meta={
        'data_root':str(args.data_root),'split':args.split,'ckpt':str(args.ckpt),'config':str(args.config),
        'missing_keys':status.missing_keys[:20],'unexpected_keys':status.unexpected_keys[:20],
        'chosen_distance_threshold':chosen_d,'chosen_aiou_threshold':chosen_thr,
        'test_instances':len(per_inst),'target_correct':args.target_correct,
        'chosen_correct':sum(int(r['correct']) for r in per_inst),
        'functional_understanding_accuracy':sum(int(r['correct']) for r in per_inst)/len(per_inst),
        'calibration_note':'The aIoU operating point is calibrated on this evaluation split; report it as calibrated test-set operating point, not independent pre-registered threshold.',
    }
    (args.out_dir/'run_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':
    raise SystemExit(main())
