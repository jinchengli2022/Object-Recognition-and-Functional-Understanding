#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少结果文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    target = load_json(
        RESULTS
        / "target_recognition"
        / "evaluation"
        / "threshold_accuracy_summary.json"
    )["0.5"]
    functional = load_json(
        RESULTS / "functional_understanding" / "run_meta.json"
    )

    target_correct = int(target["correct"])
    target_total = int(target["total"])
    target_accuracy = target_correct / target_total if target_total else 0.0

    functional_correct = int(functional["chosen_correct"])
    functional_total = int(functional["test_instances"])
    functional_accuracy = (
        functional_correct / functional_total if functional_total else 0.0
    )

    combined_accuracy = target_accuracy * functional_accuracy
    summary = {
        "metric_name": "end_to_end_functional_success_accuracy",
        "definition": (
            "target recognition accuracy at IoU >= 0.50 and class correct "
            "multiplied by functional understanding accuracy"
        ),
        "target_recognition": {
            "criterion": "IoU >= 0.50 and category correct",
            "correct": target_correct,
            "total": target_total,
            "accuracy": target_accuracy,
        },
        "functional_understanding": {
            "criterion": (
                "distance-tolerant object-level aIoU at the calibrated "
                "operating point"
            ),
            "distance_threshold": functional["chosen_distance_threshold"],
            "aiou_threshold": functional["chosen_aiou_threshold"],
            "correct": functional_correct,
            "total": functional_total,
            "accuracy": functional_accuracy,
        },
        "combined_accuracy": combined_accuracy,
        "combined_accuracy_percent": combined_accuracy * 100.0,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "overall_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (RESULTS / "overall_summary.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["target_recognition_accuracy", f"{target_accuracy:.9f}"])
        writer.writerow(["functional_understanding_accuracy", f"{functional_accuracy:.9f}"])
        writer.writerow(["end_to_end_functional_success_accuracy", f"{combined_accuracy:.9f}"])
        writer.writerow(["end_to_end_functional_success_percent", f"{combined_accuracy * 100.0:.6f}"])

    print()
    print("==================== 最终测试结果 ====================")
    print(
        f"目标识别准确率：{target_correct} / {target_total} "
        f"= {target_accuracy:.6f}"
    )
    print(
        f"功能理解准确率：{functional_correct} / {functional_total} "
        f"= {functional_accuracy:.6f}"
    )
    print(
        f"总体功能成功率：{combined_accuracy:.6f}，"
        f"即 {combined_accuracy * 100.0:.6f}%"
    )
    print("======================================================")
    print(f"汇总文件：{RESULTS / 'overall_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
