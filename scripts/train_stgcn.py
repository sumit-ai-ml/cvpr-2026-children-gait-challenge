"""Train the ST-GCN clip-level model and dump OOF + full-train predictions.

Then ensemble with the existing 3-tree ensemble; report S₁ delta. If positive,
update the Track 1 cached predictions; otherwise discard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.stgcn_train import train_stgcn
from src.track1_model import EVGS_ITEMS, compute_s1, tune_thresholds_for_s1


def evaluate_ensemble(weight_stgcn: float) -> tuple[float, float, float, dict[str, float]]:
    """Blend tree OOF probs with ST-GCN OOF probs at given weight. Return (acc, nrmse, s1, thresholds)."""
    tree_oof = pd.read_parquet(cfg.CACHE_DIR / "track1_oof_train.parquet")
    stgcn_oof = pd.read_parquet(cfg.CACHE_DIR / "track1_stgcn_oof.parquet")

    # Join on (patient_id, side)
    tree_oof = tree_oof.set_index(["patient_id", "side"])
    stgcn_oof = stgcn_oof.set_index(["patient_id", "side"])
    # Tree OOF rows = 188; ST-GCN OOF rows = 188 also (94 patients × 2)
    # Filter to common index
    common = tree_oof.index.intersection(stgcn_oof.index)
    tree_oof = tree_oof.loc[common].sort_index()
    stgcn_oof = stgcn_oof.loc[common].sort_index()

    oof_probs = {}
    y_true = {}
    for it in EVGS_ITEMS:
        oof_probs[it] = (1 - weight_stgcn) * tree_oof[f"oof_{it}"].values + weight_stgcn * stgcn_oof[f"oof_{it}"].values
        y_true[it] = tree_oof[f"y_{it}"].values.astype(int)
    y_total = tree_oof["y_total"].values.astype(int)
    pids = np.array([idx[0] for idx in tree_oof.index])

    thrs, _ = tune_thresholds_for_s1(oof_probs, y_true, y_total, pids, n_iters=4)
    acc, nrmse, s1 = compute_s1(oof_probs, y_true, y_total, pids, thrs)
    return acc, nrmse, s1, thrs


def main() -> None:
    print("=== TRAINING ST-GCN (clip level, 5-fold patient CV) ===")
    train_stgcn()
    print()
    print("=== ENSEMBLE EVALUATION ===")
    # Baseline (tree only)
    acc0, n0, s0, _ = evaluate_ensemble(weight_stgcn=0.0)
    print(f"Tree only:                Acc={acc0:.4f}  NRMSE={n0:.4f}  S_1={s0:.4f}")
    best = (0.0, s0, acc0, n0)
    for w in (0.10, 0.20, 0.30, 0.40, 0.50):
        acc, n, s, _ = evaluate_ensemble(weight_stgcn=w)
        print(f"Tree {1-w:.2f} + ST-GCN {w:.2f}: Acc={acc:.4f}  NRMSE={n:.4f}  S_1={s:.4f}")
        if s > best[1]:
            best = (w, s, acc, n)
    w, s, acc, n = best
    print()
    if w == 0.0:
        print(f"ST-GCN did NOT improve S_1 (best baseline 0.0). Keeping tree-only.")
    else:
        print(f"ST-GCN ADDED to ensemble: weight={w:.2f}  S_1={s:.4f}  Δ={s - s0:+.4f}")
        # Persist the chosen weight + a combined predictions file.
        tree_full = pd.read_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet")
        stgcn_full = pd.read_parquet(cfg.CACHE_DIR / "track1_stgcn_full.parquet")
        merged = tree_full.merge(stgcn_full, on=["patient_id", "side"], suffixes=("_tree", "_stgcn"))
        # Recompute blended probs + new thresholds
        _, _, _, thrs = evaluate_ensemble(weight_stgcn=w)
        for it in EVGS_ITEMS:
            merged[f"prob_{it}"] = (1 - w) * merged[f"prob_{it}_tree"] + w * merged[f"prob_{it}_stgcn"]
            merged[f"pred_{it}"] = (merged[f"prob_{it}"] >= thrs[it]).astype(int)
        merged["pred_total_sum"] = merged[[f"pred_{it}" for it in EVGS_ITEMS]].sum(axis=1)
        merged["pred_total"] = merged["pred_total_sum"]
        out = merged[["patient_id", "side"] + [f"prob_{it}" for it in EVGS_ITEMS]
                     + [f"pred_{it}" for it in EVGS_ITEMS] + ["pred_total_sum", "pred_total"]]
        out.to_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet", index=False)
        print(f"Updated cache/track1_full_preds.parquet with blended predictions.")
        (cfg.CACHE_DIR / "stgcn_summary.json").write_text(json.dumps({
            "stgcn_weight": w, "oof_acc": acc, "oof_nrmse": n, "oof_s1": s, "delta": s - s0,
        }, indent=2))


if __name__ == "__main__":
    main()
