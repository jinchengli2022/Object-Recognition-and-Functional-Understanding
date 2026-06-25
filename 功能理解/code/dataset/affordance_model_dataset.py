from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

VIEWPOINTS = [
    "from the rear view", "from the left-rear view", "from the left side view",
    "from the left-front view", "from the front view", "from the right-front view",
    "from the right side view", "from the right-rear view", "from the bottom view",
    "from the lower diagonal view", "from the upper diagonal view", "from the top view",
]

AFFORDANCES = [
    "grasp", "rotate", "open", "pour", "place", "contain", "sit", "push", "press",
    "drag", "cut", "shovel", "support", "plug", "insert", "stick", "sprinkle", "pull", "lean",
]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


class AffordanceModelPairDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, num_points: int = 2048, seed: int = 42):
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = int(num_points)
        self.seed = int(seed)
        self.rows = _read_tsv(self.data_root / f"{split}_pairs.tsv")
        self.aff_to_idx = {a: i for i, a in enumerate(AFFORDANCES)}
        if not self.rows:
            raise RuntimeError(f"empty split: {self.data_root / f'{split}_pairs.tsv'}")

    def __len__(self) -> int:
        return len(self.rows)

    def _sample(self, n: int, idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + idx * 9973)
        if n >= self.num_points:
            return rng.choice(n, size=self.num_points, replace=False)
        return rng.choice(n, size=self.num_points, replace=True)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        data = np.load(self.data_root / row["npz_path"], allow_pickle=True)
        points = np.asarray(data["points"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        sel = self._sample(len(points), idx)
        points = points[sel]
        aff = row["affordance"]
        label = labels[sel, self.aff_to_idx[aff]].astype(np.float32)
        obj_cat = row["object_category_id"]
        normal_cat = row.get("normal_class_39_id") or obj_cat
        question0 = f"This is a depth map of a {normal_cat} viewed from the front view. Which part can be used to {aff}?"
        questions = tuple(
            f"This is a depth map of a {normal_cat} viewed {vp}. Which part can be used to {aff}?"
            for vp in VIEWPOINTS
        )
        return {
            "points": torch.from_numpy(points.T.copy()).float(),
            "label": torch.from_numpy(label.copy()).float(),
            "questions": questions,
            "question0": question0,
            "instance_id": row["instance_id"],
            "object_id": row["object_id"],
            "raw_name": row["raw_name"],
            "object_category_id": obj_cat,
            "normal_class_39_id": normal_cat,
            "affordance": aff,
        }


def collate_pair_batch(batch: list[dict]):
    return {
        "points": torch.stack([b["points"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "questions": tuple(b["question0"] for b in batch),
        "instance_id": [b["instance_id"] for b in batch],
        "object_id": [b["object_id"] for b in batch],
        "raw_name": [b["raw_name"] for b in batch],
        "object_category_id": [b["object_category_id"] for b in batch],
        "normal_class_39_id": [b["normal_class_39_id"] for b in batch],
        "affordance": [b["affordance"] for b in batch],
    }
