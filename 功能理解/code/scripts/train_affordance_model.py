#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.affordance_model_dataset import AffordanceModelPairDataset, collate_pair_batch  # noqa: E402
from model.branch_3d import Branch3D  # noqa: E402
from utils.loss import HM_Loss  # noqa: E402


def load_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["model_3d"]["training"] = True
    cfg["model_3d"]["freeze_text_encoder"] = bool(cfg["train"].get("freeze_text_encoder", True))
    return cfg


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def evaluate(model: Branch3D, loader: DataLoader, device: torch.device, criterion: HM_Loss) -> dict[str, float]:
    model.eval()
    losses, ious = [], []
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            out = model((batch["questions"],), points)
            pred = out[0] if isinstance(out, tuple) else out
            loss = criterion(pred, label)
            losses.append(float(loss.detach().cpu()))
            for p, y in zip(pred.detach().cpu().numpy(), label.detach().cpu().numpy()):
                pb = p >= 0.5
                yb = y >= 0.5
                union = np.logical_or(pb, yb).sum()
                ious.append(float(np.logical_and(pb, yb).sum() / union) if union else 1.0)
    model.train()
    return {"loss": float(np.mean(losses)), "iou@0.5": float(np.mean(ious))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config/train_affordance_model.yaml"))
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    train_cfg = cfg["train"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(train_cfg.get("gpu", 0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(train_cfg.get("seed", 42)))
    np.random.seed(int(train_cfg.get("seed", 42)))

    out_dir = Path(train_cfg["save_dir"]) / train_cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "used_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    train_ds = AffordanceModelPairDataset(cfg["dataset"]["data_root"], "train", cfg["dataset"].get("num_points", 2048), seed=train_cfg["seed"])
    val_ds = AffordanceModelPairDataset(cfg["dataset"]["data_root"], "val", cfg["dataset"].get("num_points", 2048), seed=train_cfg["seed"] + 11)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=cfg["dataset"].get("num_workers", 4), drop_last=True, collate_fn=collate_pair_batch, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=cfg["dataset"].get("num_workers", 4), collate_fn=collate_pair_batch, pin_memory=True)

    model = Branch3D(cfg["model_3d"]).to(device)
    if train_cfg.get("init_ckpt"):
        ckpt = torch.load(train_cfg["init_ckpt"], map_location=device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        status = model.load_state_dict(state, strict=False)
        print(json.dumps({"missing_keys": status.missing_keys[:20], "unexpected_keys": status.unexpected_keys[:20]}, ensure_ascii=False), flush=True)
    if cfg["model_3d"].get("freeze_text_encoder", True):
        for p in model.text_encoder.parameters():
            p.requires_grad = False
    criterion = HM_Loss().to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and train_cfg.get("amp", True))

    rows = []
    best = -1.0
    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        run_loss = []
        for step, batch in enumerate(train_loader):
            points = batch["points"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available() and train_cfg.get("amp", True)):
                out = model((batch["questions"],), points)
                pred = out[0] if isinstance(out, tuple) else out
                loss = criterion(pred, label)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss.append(float(loss.detach().cpu()))
            if step % int(train_cfg.get("log_interval", 25)) == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={run_loss[-1]:.5f}", flush=True)
        metrics = evaluate(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": float(np.mean(run_loss)), "val_loss": metrics["loss"], "val_iou@0.5": metrics["iou@0.5"]}
        rows.append(row)
        write_tsv(out_dir / "train_log.tsv", rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if metrics["iou@0.5"] > best:
            best = metrics["iou@0.5"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "config": cfg}, out_dir / "best_affordance_model.pt")
    torch.save({"model": model.state_dict(), "epoch": int(train_cfg["epochs"]) - 1, "config": cfg}, out_dir / "last_affordance_model.pt")
    print(json.dumps({"out_dir": str(out_dir.resolve()), "best_val_iou@0.5": best}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
