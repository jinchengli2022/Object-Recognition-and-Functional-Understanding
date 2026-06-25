#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main() -> int:
    ap=argparse.ArgumentParser(description='Compute class-correct + IoU threshold accuracy.')
    ap.add_argument('--per-instance', type=Path, required=True)
    ap.add_argument('--out-json', type=Path, required=True)
    ap.add_argument('--thresholds', default='0.25,0.30,0.50,0.75,0.90')
    args=ap.parse_args()
    rows=list(csv.DictReader(args.per_instance.open(encoding='utf-8'), delimiter='\t'))
    summary={}
    for t in [float(x) for x in args.thresholds.replace(',', ' ').split()]:
        correct=sum(1 for r in rows if int(r['class_correct'])==1 and float(r['mask_iou_class_matched'])>=t)
        summary[f'{t:g}']={'correct':correct,'total':len(rows),'accuracy':correct/len(rows) if rows else 0.0}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
