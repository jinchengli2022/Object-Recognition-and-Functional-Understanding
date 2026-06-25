#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def load_manifest(path: Path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # Model outputs may include the chat template; parse the last valid JSON object.
    starts = [m.start() for m in re.finditer(r'\{', text)]
    for start in reversed(starts):
        tail = text[start:]
        ends = [m.end() for m in re.finditer(r'\}', tail)]
        for end in reversed(ends):
            try:
                return json.loads(tail[:end])
            except Exception:
                continue
    return {"raw_output": text, "object_category": "", "affordances": [], "description": "", "confidence": 0.0}


def build_category_vocab(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, str]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {'zh': set(), 'raw': set()})
    alias: dict[str, str] = {}
    for row in rows:
        category_id = row['category_id']
        buckets[category_id]['zh'].add(row.get('category_zh', ''))
        buckets[category_id]['raw'].add(row.get('raw_name', ''))
        alias[category_id] = category_id
        if row.get('category_zh'):
            alias[row['category_zh']] = category_id
    lines = []
    for category_id in sorted(buckets):
        zh = '/'.join(x for x in sorted(buckets[category_id]['zh']) if x)
        raw = '、'.join(x for x in sorted(buckets[category_id]['raw'])[:8] if x)
        lines.append(f'- {category_id}: {zh}; 样例：{raw}')
    return lines, alias


def normalize_category(pred: str, valid_categories: set[str], alias: dict[str, str]) -> str:
    pred = str(pred).strip()
    if pred in valid_categories:
        return pred
    if pred in alias:
        return alias[pred]
    low = pred.lower().strip()
    for category_id in valid_categories:
        if low == category_id.lower():
            return category_id
    # Last-resort normalization for outputs like "category: charger".
    for category_id in valid_categories:
        if re.search(rf'(?<![a-zA-Z0-9_]){re.escape(category_id)}(?![a-zA-Z0-9_])', low):
            return category_id
    return pred


def build_prompt(vocab: list[str], category_lines: list[str]) -> str:
    template = (Path(__file__).resolve().parents[1] / 'prompts/qwen_semantic_prompt_zh.txt').read_text(encoding='utf-8')
    return (
        template
        .replace('{affordance_vocab}', ', '.join(vocab))
        .replace('{category_vocab}', '\n'.join(category_lines))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate local Qwen semantic recognition/affordance-prior baseline on RGB images.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/qwen_semantic_config.yaml')
    parser.add_argument('--model-dir', type=Path, default=None)
    parser.add_argument('--manifest', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    root = args.config.resolve().parents[1]
    model_dir = args.model_dir or Path(cfg['model_dir'])
    manifest = args.manifest or (root / cfg['manifest']).resolve()
    out_dir = args.output_dir or (root / cfg['output_dir'])
    vocab = list(cfg['affordance_vocab'])

    # 延迟导入，避免只是查看文档时要求安装完整大模型依赖。
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if args.device.startswith('cuda') and torch.cuda.is_available() else None,
    )
    if not (args.device.startswith('cuda') and torch.cuda.is_available()):
        model = model.to('cpu')

    rows = load_manifest(manifest)
    if args.limit > 0:
        rows = rows[:args.limit]

    category_lines, alias = build_category_vocab(rows)
    categories = set(alias.values())
    prompt = build_prompt(vocab, category_lines)
    results = []
    base = manifest.parent
    for i, row in enumerate(rows, start=1):
        image_path = base / row['image_path']
        image = Image.open(image_path).convert('RGB')
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors='pt')
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        if 'input_ids' in inputs:
            generated = generated[:, inputs['input_ids'].shape[-1]:]
        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
        pred = parse_json(decoded)
        pred_cat_raw = str(pred.get('object_category', '')).strip()
        pred_cat = normalize_category(pred_cat_raw, categories, alias)
        pred_aff = pred.get('affordances', []) or []
        if isinstance(pred_aff, str):
            pred_aff = [x.strip() for x in pred_aff.split(',') if x.strip()]
        results.append({
            'image_id': row['image_id'],
            'gt_category': row['category_id'],
            'pred_category': pred_cat,
            'pred_category_raw': pred_cat_raw,
            'category_correct': int(pred_cat == row['category_id']),
            'pred_affordances': ','.join(pred_aff),
            'description': pred.get('description', ''),
            'confidence': pred.get('confidence', ''),
            'raw_output': decoded.replace('\n', '\\n'),
        })
        if i % 20 == 0:
            print(f'processed {i}/{len(rows)}', flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(results[0]) if results else []
    with (out_dir / 'semantic_predictions.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    total = len(results)
    correct = sum(int(r['category_correct']) for r in results)
    summary = {
        'model_dir': str(model_dir),
        'manifest': str(manifest),
        'total_images': total,
        'category_correct': correct,
        'category_accuracy': correct / total if total else 0.0,
        'note': 'This evaluates semantic object category and text-level affordance prior only; it does not evaluate 2D mask IoU or 3D point-level aIoU.',
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
