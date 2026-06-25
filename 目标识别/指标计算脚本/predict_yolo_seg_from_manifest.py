#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, delimiter='\t', fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> int:
    ap=argparse.ArgumentParser(description='Run YOLO-seg on a target-recognition manifest.')
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--gt-manifest', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--device', default='0')
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--imgsz', type=int, default=768)
    args=ap.parse_args()
    root=args.gt_manifest.resolve().parent
    rows=read_tsv(args.gt_manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model=YOLO(str(args.weights))
    names=model.names
    if isinstance(names, list):
        idx_to_class={i:n for i,n in enumerate(names)}
    else:
        idx_to_class={int(k):v for k,v in names.items()}
    pred_rows=[]
    for i,row in enumerate(rows):
        img_path=Path(row['image_path'])
        if not img_path.is_absolute():
            img_path=root/img_path
        image=cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(img_path)
        h,w=image.shape[:2]
        res=model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        if res.masks is None or res.boxes is None:
            continue
        masks=res.masks.data.detach().cpu().numpy()
        classes=res.boxes.cls.detach().cpu().numpy().astype(int)
        confs=res.boxes.conf.detach().cpu().numpy()
        order=np.argsort(-confs)
        for rank,j in enumerate(order):
            mask=(masks[j]>0.5).astype(np.uint8)*255
            if mask.shape[:2] != (h,w):
                mask=cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
            pred_rel=Path('pred_masks')/row['image_id']/f'pred_{rank:02d}.png'
            (args.out_dir/pred_rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out_dir/pred_rel), mask)
            pred_rows.append({
                'image_id': row['image_id'],
                'pred_instance_id': f"{row['image_id']}__yolo_pred{rank:02d}",
                'pred_category_id': idx_to_class[int(classes[j])],
                'pred_confidence': f'{float(confs[j]):.6f}',
                'pred_mask_path': str((args.out_dir/pred_rel).resolve()),
            })
        if (i+1)%20==0:
            print(f'predicted {i+1}/{len(rows)}', flush=True)
    write_tsv(args.out_dir/'predictions.tsv', pred_rows)
    summary={'input_count':len(rows),'prediction_count':len(pred_rows),'weights':str(args.weights),'gt_manifest':str(args.gt_manifest)}
    (args.out_dir/'prediction_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
