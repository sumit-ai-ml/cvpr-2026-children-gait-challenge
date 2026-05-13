"""Step 8: Augment tree input with physics signals.

Adds 17 physics signals (one per EVGS item) to the existing 1858-dim pooled
features and retrains the 3-tree ensemble. Compare OOF S_1 to baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.track1_model import (
    EVGS_ITEMS, _run_item, tune_thresholds_for_s1, compute_s1, build_track1_dataset,
)


def main():
    # Load pooled features and physics signals
    train_df, all_df, feature_cols = build_track1_dataset()
    phys_train = pd.read_parquet(cfg.CACHE_DIR / "physics_signals_train.parquet")
    phys_all = pd.read_parquet(cfg.CACHE_DIR / "physics_signals_all.parquet")

    # Merge physics signals into train_df and all_df
    # Match on (patient_id, side)
    phys_cols = [f"sig_{i}" for i in range(1, 18)]

    train_aug = train_df.merge(phys_train[["patient_id", "side"] + phys_cols],
                                on=["patient_id", "side"], how="left")
    all_aug = all_df.merge(phys_all[["patient_id", "side"] + phys_cols],
                            on=["patient_id", "side"], how="left")

    # NaN handling
    for c in phys_cols:
        train_aug[c] = train_aug[c].fillna(0.0)
        all_aug[c] = all_aug[c].fillna(0.0)

    new_feature_cols = feature_cols + phys_cols
    print(f"Original features: {len(feature_cols)}, augmented: {len(new_feature_cols)}")

    X_tr = train_aug[new_feature_cols].to_numpy(dtype=np.float32)
    pids = train_aug["patient_id"].to_numpy()
    y_total = train_aug["y_total"].to_numpy(dtype=np.float32)

    oof_probs_dict: dict[str, np.ndarray] = {}
    y_true_dict: dict[str, np.ndarray] = {}

    for it in EVGS_ITEMS:
        y = train_aug[f"y_{it}"].to_numpy(dtype=np.int32)
        oof, spw = _run_item(X_tr, y, pids)
        oof_probs_dict[it] = oof
        y_true_dict[it] = y
        per_acc = ((oof >= 0.5).astype(int) == y).mean()
        print(f"  item {it}: OOF@0.5 acc={per_acc:.3f}")

    # Tune thresholds
    print()
    print("Tuning thresholds jointly for S_1 ...")
    thrs, best_s1 = tune_thresholds_for_s1(oof_probs_dict, y_true_dict, y_total, pids, n_iters=5)
    acc, nrmse, s1 = compute_s1(oof_probs_dict, y_true_dict, y_total, pids, thrs)
    print(f"  OOF Acc:   {acc:.4f}")
    print(f"  OOF NRMSE: {nrmse:.4f}")
    print(f"  OOF S_1:   {s1:.4f}")
    print(f"  (Baseline OOF S_1: 0.8267)")

    # Save augmented OOF for comparison
    oof_df = pd.DataFrame({"patient_id": pids, "side": train_aug["side"].values})
    for it in EVGS_ITEMS:
        oof_df[f"oof_{it}"] = oof_probs_dict[it]
        oof_df[f"y_{it}"] = y_true_dict[it]
    oof_df["y_total"] = y_total
    oof_df.to_parquet(cfg.CACHE_DIR / "track1_oof_train_aug.parquet", index=False)
    print(f"Saved cache/track1_oof_train_aug.parquet")


if __name__ == "__main__":
    main()
