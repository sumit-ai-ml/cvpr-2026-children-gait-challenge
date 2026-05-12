"""Train the PoseTransformer (DSTformer-style) at clip level and ensemble with the
existing Track 1 ensemble. Gated on OOF S₁ improvement."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.posetransformer import PoseTransformer
from src.stgcn_train import ClipPoseDataset, _build_clip_pool, _pat_fold_indices, T_TARGET
from src.track1_model import EVGS_ITEMS, compute_s1, tune_thresholds_for_s1


def _train_one_fold(train_clips, val_clips, labels, pos_weight, device, epochs=25, cache=None):
    train_ds = ClipPoseDataset(train_clips, labels, cache=cache)
    val_ds = ClipPoseDataset(val_clips, labels, cache=cache)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = PoseTransformer(in_channels=2, num_classes=34, dim=96, depth=3, heads=4, t_max=T_TARGET + 8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    for ep in range(epochs):
        model.train()
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    val_probs, val_pids = [], []
    with torch.no_grad():
        for x, y, pid in val_dl:
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            val_probs.append(p)
            val_pids.extend(pid.tolist())
    return model, (np.concatenate(val_probs, axis=0) if val_probs else np.zeros((0, 34))), val_pids


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PoseTransformer training on {device}")
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_clips, labels = _build_clip_pool()
    train_clips = [c for c in all_clips if c.patient_id in labels]
    print(f"train_clips={len(train_clips)} total_clips={len(all_clips)}")

    pos_count = np.zeros(34, dtype=np.float32)
    neg_count = np.zeros(34, dtype=np.float32)
    for c in train_clips:
        lab = labels[c.patient_id]
        for i, it in enumerate(range(1, 18)):
            v = int(lab["left"][str(it)]); pos_count[i] += v; neg_count[i] += 1 - v
            v = int(lab["right"][str(it)]); pos_count[17 + i] += v; neg_count[17 + i] += 1 - v
    pos_weight = torch.from_numpy(np.clip(neg_count / np.clip(pos_count, 1, None), 1.0, 20.0).astype(np.float32))

    pids_per_clip = np.array([c.patient_id for c in train_clips])
    unique_pids = np.unique(pids_per_clip)
    cache: dict = {}

    oof = pd.DataFrame({"patient_id": pids_per_clip, "clip_path": [c.path.name for c in train_clips]})
    for it in range(1, 18):
        oof[f"oof_L_{it}"] = np.nan
        oof[f"oof_R_{it}"] = np.nan

    t0 = time.time()
    for fi, (tr_pids_idx, va_pids_idx) in enumerate(_pat_fold_indices(unique_pids, n_splits=5)):
        tr_pids = set(unique_pids[tr_pids_idx].tolist())
        va_pids = set(unique_pids[va_pids_idx].tolist())
        tr = [c for c in train_clips if c.patient_id in tr_pids]
        va = [c for c in train_clips if c.patient_id in va_pids]
        print(f"\n[fold {fi+1}/5] train_clips={len(tr)} val_clips={len(va)}")
        _, va_probs, _ = _train_one_fold(tr, va, labels, pos_weight, device, epochs=25, cache=cache)
        idxs = np.where(np.isin(pids_per_clip, list(va_pids)))[0]
        for k, idx in enumerate(idxs):
            for it in range(17):
                oof.at[idx, f"oof_L_{it+1}"] = float(va_probs[k, it])
                oof.at[idx, f"oof_R_{it+1}"] = float(va_probs[k, 17 + it])
        print(f"  elapsed: {time.time()-t0:.1f}s")

    print("\nAggregating clip-level OOF to (patient, side) ...")
    agg_rows = []
    for pid in sorted(set(pids_per_clip.tolist())):
        sub = oof[oof.patient_id == pid]
        for side in ("L", "R"):
            row = {"patient_id": pid, "side": side}
            for it in range(1, 18):
                col = f"oof_{side}_{it}"
                row[f"oof_{it}"] = float(sub[col].mean()) if not sub[col].isna().all() else 0.5
            agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_parquet(cfg.CACHE_DIR / "track1_posetransformer_oof.parquet", index=False)

    print("\nRefitting on full train set ...")
    full_model, _, _ = _train_one_fold(train_clips, train_clips[:64], labels, pos_weight, device, epochs=25, cache=cache)
    all_ds = ClipPoseDataset(all_clips, labels, cache=cache)
    all_dl = DataLoader(all_ds, batch_size=64, shuffle=False)
    full_model.eval()
    all_probs, all_pids = [], []
    with torch.no_grad():
        for x, y, pid in all_dl:
            p = torch.sigmoid(full_model(x.to(device))).cpu().numpy()
            all_probs.append(p)
            all_pids.extend(pid.tolist())
    pcat = np.concatenate(all_probs, axis=0)
    pred_rows = []
    pid_arr = np.array(all_pids)
    for pid in sorted(set(all_pids)):
        mask = pid_arr == pid
        avg = pcat[mask].mean(axis=0)
        pred_rows.append({"patient_id": int(pid), "side": "L", **{f"prob_{i}": float(avg[i - 1]) for i in range(1, 18)}})
        pred_rows.append({"patient_id": int(pid), "side": "R", **{f"prob_{i}": float(avg[17 + i - 1]) for i in range(1, 18)}})
    pd.DataFrame(pred_rows).to_parquet(cfg.CACHE_DIR / "track1_posetransformer_full.parquet", index=False)

    # Ensemble eval
    print("\n=== ENSEMBLE EVALUATION ===")
    tree_oof = pd.read_parquet(cfg.CACHE_DIR / "track1_oof_train.parquet")
    pt_oof = pd.read_parquet(cfg.CACHE_DIR / "track1_posetransformer_oof.parquet")
    tree_oof = tree_oof.set_index(["patient_id", "side"])
    pt_oof = pt_oof.set_index(["patient_id", "side"])
    common = tree_oof.index.intersection(pt_oof.index)
    tree_oof = tree_oof.loc[common].sort_index()
    pt_oof = pt_oof.loc[common].sort_index()
    y_true = {it: tree_oof[f"y_{it}"].values.astype(int) for it in EVGS_ITEMS}
    y_total = tree_oof["y_total"].values.astype(int)
    pids = np.array([idx[0] for idx in tree_oof.index])

    def eval_w(w_pt: float):
        probs = {it: (1 - w_pt) * tree_oof[f"oof_{it}"].values + w_pt * pt_oof[f"oof_{it}"].values for it in EVGS_ITEMS}
        thrs, _ = tune_thresholds_for_s1(probs, y_true, y_total, pids, n_iters=4)
        return compute_s1(probs, y_true, y_total, pids, thrs), thrs

    (acc0, n0, s0), _ = eval_w(0.0)
    print(f"Tree only:                  Acc={acc0:.4f}  NRMSE={n0:.4f}  S_1={s0:.4f}")
    best = (0.0, s0)
    for w in (0.10, 0.20, 0.30, 0.40, 0.50):
        (acc, n, s), _ = eval_w(w)
        print(f"Tree {1-w:.2f} + PoseTr {w:.2f}: Acc={acc:.4f}  NRMSE={n:.4f}  S_1={s:.4f}")
        if s > best[1]:
            best = (w, s)
    w, s = best
    print()
    if w == 0.0:
        print(f"PoseTransformer did NOT improve S_1. Keeping tree-only.")
    else:
        print(f"PoseTransformer ADDED: weight={w:.2f}  S_1={s:.4f}  Δ={s - s0:+.4f}")
        # Persist the blended preds
        (_, _, _), thrs = eval_w(w)
        tree_full = pd.read_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet")
        pt_full = pd.read_parquet(cfg.CACHE_DIR / "track1_posetransformer_full.parquet")
        merged = tree_full.merge(pt_full, on=["patient_id", "side"], suffixes=("_tree", "_pt"))
        for it in EVGS_ITEMS:
            merged[f"prob_{it}"] = (1 - w) * merged[f"prob_{it}_tree"] + w * merged[f"prob_{it}_pt"]
            merged[f"pred_{it}"] = (merged[f"prob_{it}"] >= thrs[it]).astype(int)
        merged["pred_total_sum"] = merged[[f"pred_{it}" for it in EVGS_ITEMS]].sum(axis=1)
        merged["pred_total"] = merged["pred_total_sum"]
        out = merged[["patient_id", "side"] + [f"prob_{it}" for it in EVGS_ITEMS]
                     + [f"pred_{it}" for it in EVGS_ITEMS] + ["pred_total_sum", "pred_total"]]
        out.to_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet", index=False)
    (cfg.CACHE_DIR / "posetransformer_summary.json").write_text(
        json.dumps({"weight": w, "s1": s, "baseline_s1": s0, "delta": s - s0}, indent=2))


if __name__ == "__main__":
    main()
