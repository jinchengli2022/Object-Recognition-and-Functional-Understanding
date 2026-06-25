#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]

def read_tsv(p):
    with open(p,encoding='utf-8') as f: return list(csv.DictReader(f,delimiter='\t'))

def write_tsv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(p,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,delimiter='\t',fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--yolo-root', type=Path, required=True)
    ap.add_argument('--gt-manifest', type=Path, required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--device', default='0')
    ap.add_argument('--conf', type=float, default=0.05)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_mapping=read_tsv(args.yolo_root/'mapping.tsv')
    mapping=all_mapping if args.split == 'all' else [r for r in all_mapping if r['split']==args.split]
    ids={r['image_id'] for r in mapping}
    gt=[r for r in read_tsv(args.gt_manifest) if r['image_id'] in ids]
    write_tsv(args.out_dir/f'gt_{args.split}_manifest.tsv', gt)
    class_map=json.loads((args.yolo_root/'class_map.json').read_text(encoding='utf-8'))
    idx_to_class={int(k):v for k,v in class_map['index_to_class'].items()}
    model=YOLO(str(args.weights))
    pred_rows=[]
    for i,row in enumerate(mapping):
        img_path=args.yolo_root/row['image_path']
        image=cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        h,w=image.shape[:2]
        results=model.predict(str(img_path), imgsz=512, conf=args.conf, device=args.device, verbose=False)
        res=results[0]
        if res.masks is None or res.boxes is None:
            continue
        masks=res.masks.data.detach().cpu().numpy()
        classes=res.boxes.cls.detach().cpu().numpy().astype(int)
        confs=res.boxes.conf.detach().cpu().numpy()
        order=np.argsort(-confs)
        for rank,j in enumerate(order):
            mask=(masks[j]>0.5).astype(np.uint8)*255
            if mask.shape[:2] != (h,w):
                mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
            pred_rel=Path('pred_masks')/row['image_id']/f'pred_{rank:02d}.png'
            (args.out_dir/pred_rel).parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(args.out_dir/pred_rel), mask)
            pred_rows.append({
                'image_id':row['image_id'],
                'pred_instance_id':f"{row['image_id']}__yolo_pred{rank:02d}",
                'pred_category_id':idx_to_class[int(classes[j])],
                'pred_confidence':f'{float(confs[j]):.6f}',
                'pred_mask_path':str((args.out_dir/pred_rel).resolve()),
            })
        if (i+1)%10==0: print(f'predicted {i+1}/{len(mapping)}', flush=True)
    write_tsv(args.out_dir/'predictions.tsv', pred_rows)
    summary={'split':args.split,'input_count':len(mapping),'prediction_count':len(pred_rows),'gt_count':len(gt),'weights':str(args.weights)}
    (args.out_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
