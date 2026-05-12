"""Train Track 1 ensemble on augmented (original + mirrored) features.

For each (patient_id, side, mirrored) row:
  - mirrored=0: use original (left/right) labels.
  - mirrored=1, side='L': features represent the original RIGHT limb's kinematics → use right labels.
  - mirrored=1, side='R': features represent the original LEFT limb's kinematics → use left labels.

OOF eval is computed ONLY on original (mirrored=0) rows. Mirrored rows are train-only.
Gate: accept only if OOF S1 > baseline 0.8263.
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
from src import cv as cvmod
from src.data_io import load_track1_labels
from src.track1_model import (
    EVGS_ITEMS,
    _fit_lgb,
    _fit_xgb,
    _fit_cb,
    _refit_full,
    _lgb_params,
    _xgb_params,
    _cb_params,
    compute_s1,
    tune_thresholds_for_s1,
)

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def build_augmented_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    pooled = pd.read_parquet(cfg.CACHE_DIR / "features_patient_limb_mirrored.parquet")
    labels = load_track1_labels()
    side_key_orig = {"L": "left", "R": "right"}
    side_key_mirror = {"L": "right", "R": "left"}  # mirrored: L features <- original R

    train_rows: list[dict] = []
    for _, row in pooled.iterrows():
        pid = int(row["patient_id"])
        if pid not in labels:
            continue
        if row["mirrored"] == 0:
            lab = labels[pid][side_key_orig[row["side"]]]
        else:
            lab = labels[pid][side_key_mirror[row["side"]]]
        rec = row.to_dict()
        for it in EVGS_ITEMS:
            rec[f"y_{it}"] = int(lab[it])
        rec["y_total"] = int(lab["Total"])
        train_rows.append(rec)
    train_df = pd.DataFrame(train_rows)
    # All rows (not just labeled) for inference
    all_df = pooled[pooled.mirrored == 0].copy()
    feature_cols = [c for c in pooled.columns if c not in ("patient_id", "side", "mirrored")]
    return train_df, all_df, feature_cols


def _run_item(X, y, pids, mirrored_mask):
    """5-fold patient-grouped CV. Mirrored rows always in train, never in val.
    Returns OOF probs (computed only on mirrored=0 rows; mirrored=1 rows get NaN)."""
    pos = max(int(y[~mirrored_mask].sum()), 1)
    neg = max(int((1 - y[~mirrored_mask]).sum()), 1)
    spw = neg / pos
    oof_lgb = np.full(len(y), np.nan)
    oof_xgb = np.full(len(y), np.nan)
    oof_cb = np.full(len(y), np.nan)

    # Patient-grouped 5-fold on ORIGINAL patients only (we don't want to leak any patient).
    orig_pids = pids[~mirrored_mask]
    for tr_idx_orig, va_idx_orig in cvmod.patient_kfold(orig_pids, n_splits=5, seed=cfg.CFG.seed):
        # Resolve to indices in the full augmented array
        va_pids_set = set(orig_pids[va_idx_orig].tolist())
        # Val mask: original AND patient in val
        va_mask_full = (~mirrored_mask) & np.isin(pids, list(va_pids_set))
        # Train mask: everything else (incl. mirrored rows of train patients; exclude mirrored of val patients to prevent leakage)
        tr_mask_full = ~np.isin(pids, list(va_pids_set))
        Xtr, ytr = X[tr_mask_full], y[tr_mask_full]
        Xva, yva = X[va_mask_full], y[va_mask_full]
        m_lgb = _fit_lgb(Xtr, ytr, Xva, yva, spw)
        oof_lgb[va_mask_full] = m_lgb.predict(Xva, num_iteration=m_lgb.best_iteration)
        m_xgb = _fit_xgb(Xtr, ytr, Xva, yva, spw)
        oof_xgb[va_mask_full] = m_xgb.predict_proba(Xva)[:, 1]
        m_cb = _fit_cb(Xtr, ytr, Xva, yva, spw)
        oof_cb[va_mask_full] = m_cb.predict_proba(Xva)[:, 1]
    oof_avg = (oof_lgb + oof_xgb + oof_cb) / 3.0
    return oof_avg, spw


def _refit_full_aug(X, y, spw):
    rng = np.random.default_rng(cfg.CFG.seed)
    n = len(y)
    perm = rng.permutation(n)
    split = max(int(n * 0.1), 1)
    va_idx = perm[:split]; tr_idx = perm[split:]
    m_lgb = _fit_lgb(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], spw)
    m_xgb = _fit_xgb(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], spw)
    m_cb = _fit_cb(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], spw)
    return m_lgb, m_xgb, m_cb


def _predict_ensemble(models, X):
    m_lgb, m_xgb, m_cb = models
    p_lgb = m_lgb.predict(X)
    p_xgb = m_xgb.predict_proba(X)[:, 1]
    p_cb = m_cb.predict_proba(X)[:, 1]
    return (p_lgb + p_xgb + p_cb) / 3.0


def main() -> None:
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_df, all_df, feature_cols = build_augmented_dataset()
    n_orig = (train_df.mirrored == 0).sum()
    n_mirror = (train_df.mirrored == 1).sum()
    print(f"Augmented train: total={len(train_df)}, original={n_orig}, mirrored={n_mirror}")

    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    pids = train_df["patient_id"].to_numpy()
    y_total = train_df["y_total"].to_numpy(dtype=np.float32)
    mirrored_mask = (train_df["mirrored"].to_numpy() == 1)

    oof_probs_dict: dict[str, np.ndarray] = {}
    y_true_dict: dict[str, np.ndarray] = {}
    item_models: dict[str, tuple] = {}

    for it in EVGS_ITEMS:
        y = train_df[f"y_{it}"].to_numpy(dtype=np.int32)
        oof, spw = _run_item(X, y, pids, mirrored_mask)
        # Filter to ORIGINAL rows for the OOF table
        orig_rows = ~mirrored_mask
        oof_probs_dict[it] = oof[orig_rows]
        y_true_dict[it] = y[orig_rows]
        item_models[it] = _refit_full_aug(X, y, spw)
        per_item_acc_orig = float(((oof[orig_rows] >= 0.5).astype(int) == y[orig_rows]).mean())
        print(f"  item {it}: base_rate={y[orig_rows].mean():.2f}  OOF@0.5 acc(orig)={per_item_acc_orig:.3f}")

    # Joint threshold tuning on the OOF (original rows only).
    y_total_orig = y_total[~mirrored_mask]
    pids_orig = pids[~mirrored_mask]
    thrs, best_s1 = tune_thresholds_for_s1(oof_probs_dict, y_true_dict, y_total_orig, pids_orig, n_iters=5)
    acc, nrmse, s1 = compute_s1(oof_probs_dict, y_true_dict, y_total_orig, pids_orig, thrs)
    print()
    print(f"== TRACK 1 (mirror-augmented) OOF SCORE ==")
    print(f"  Acc:   {acc:.4f}")
    print(f"  NRMSE: {nrmse:.4f}")
    print(f"  S_1:   {s1:.4f}")

    baseline_s1 = 0.8263
    if s1 <= baseline_s1:
        print(f"\nMirror-aug did NOT improve S_1 (baseline {baseline_s1}, got {s1:.4f}). Discarding.")
        return

    print(f"\nMIRROR-AUG IMPROVED S_1: {baseline_s1:.4f} -> {s1:.4f}  (Δ=+{s1-baseline_s1:.4f})")

    # Predict on all 110 patients × 2 sides (original only).
    X_all = all_df[feature_cols].to_numpy(dtype=np.float32)
    full = pd.DataFrame({"patient_id": all_df["patient_id"].values, "side": all_df["side"].values})
    for it in EVGS_ITEMS:
        prob = _predict_ensemble(item_models[it], X_all)
        full[f"prob_{it}"] = prob
        full[f"pred_{it}"] = (prob >= thrs[it]).astype(int)
    full["pred_total_sum"] = full[[f"pred_{it}" for it in EVGS_ITEMS]].sum(axis=1)
    full["pred_total"] = full["pred_total_sum"]
    full.to_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet", index=False)

    # OOF table
    oof_df = pd.DataFrame({"patient_id": pids_orig, "side": train_df[~mirrored_mask]["side"].values})
    for it in EVGS_ITEMS:
        oof_df[f"oof_{it}"] = oof_probs_dict[it]
        oof_df[f"y_{it}"] = y_true_dict[it]
    oof_df["y_total"] = y_total_orig
    oof_df.to_parquet(cfg.CACHE_DIR / "track1_oof_train.parquet", index=False)

    print(f"Updated cache/track1_full_preds.parquet and track1_oof_train.parquet")
    (cfg.CACHE_DIR / "track1_mirror_summary.json").write_text(json.dumps({
        "baseline_s1": baseline_s1, "new_s1": float(s1), "delta": float(s1 - baseline_s1),
        "acc": float(acc), "nrmse": float(nrmse),
    }, indent=2))


if __name__ == "__main__":
    main()
