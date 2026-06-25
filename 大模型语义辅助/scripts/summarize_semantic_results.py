#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description='Summarize semantic baseline TSV results by category.')
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--out-dir', type=Path, default=None)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(encoding='utf-8'), delimiter='\t'))
    out_dir = args.out_dir or args.predictions.parent
    bucket = defaultdict(lambda: [0, 0])
    for r in rows:
        b = bucket[r['gt_category']]
        b[0] += 1
        b[1] += int(r.get('category_correct', 0))
    per_cat = [
        {'category': k, 'total': v[0], 'correct': v[1], 'accuracy': v[1] / v[0] if v[0] else 0.0}
        for k, v in sorted(bucket.items())
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'per_category_semantic_accuracy.tsv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, delimiter='\t', fieldnames=['category', 'total', 'correct', 'accuracy'])
        w.writeheader(); w.writerows(per_cat)
    total = len(rows); correct = sum(int(r.get('category_correct', 0)) for r in rows)
    summary = {'total_images': total, 'category_correct': correct, 'category_accuracy': correct / total if total else 0.0}
    (out_dir / 'summary_semantic_accuracy.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
