"""Step 1: Replace 17-dim EVGS-only LR input with 35-dim EVGS bridge.

35-dim = 17 raw probs + 17 binaries (at tuned thresholds) + 1 predicted Total.

Tune LR C on LOPO (0.01, 0.1, 1, 10). Re-run Track 2 ensemble search with the
new EVGS-only branch. Save predictions to cache/track2_preds.parquet if OOF S_2 lifts.

Gate: only accept if OOF S_2 > current baseline.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import cv as cvmod
from src.data_io import load_track2_labels
from src.evgs_bridge import bridge_feature_cols, build_bridge_table
from src.track2_model import CLASSES, _fit_knn, _fit_lgb_multi, _fit_lr, build_track2_dataset


def _fit_evgs_bridge(Xtr, ytr, C: float):
    return LogisticRegression(
        penalty="l2", C=C, multi_class="multinomial",
        solver="lbfgs", max_iter=2000, class_weight="balanced",
        random_state=cfg.CFG.seed,
    ).fit(Xtr, ytr)


def main() -> None:
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    train_df, all_df, _ = build_track2_dataset()
    bridge_df = build_bridge_table(use_oof_for_train=True)
    bridge_cols = bridge_feature_cols()

    # Attach bridge cols to train_df
    bridge_indexed = bridge_df.set_index(["patient_id", "side"])
    train_df = train_df.merge(
        bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left"
    )
    all_df_b = all_df.merge(
        bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left"
    )

    feature_cols = [c for c in train_df.columns
                    if c not in ("patient_id", "side", "y") and not c.startswith("evgs_") and c not in bridge_cols]
    # Use full kinematic features (unchanged) for the LGB/kNN/LR-on-features models.
    # The EVGS-bridge LR uses ONLY the bridge cols.

    X_full = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_bridge = train_df[bridge_cols].to_numpy(dtype=np.float32)
    y = np.array([label_to_idx[s] for s in train_df["y"].values], dtype=np.int64)
    pids = train_df["patient_id"].to_numpy()

    print(f"Track 2 train: {len(train_df)} limbs. Full features: {len(feature_cols)}. Bridge dim: {X_bridge.shape[1]}")

    # LOPO over original 22 patients
    folds = list(cvmod.leave_one_patient_out(pids))

    # Tune C for EVGS-bridge LR on LOPO
    print()
    print("Tuning LR C on EVGS-bridge LR (LOPO):")
    best_C, best_C_s2 = 1.0, -1.0
    bridge_oof_by_C: dict[float, np.ndarray] = {}
    for C in (0.01, 0.1, 1.0, 10.0):
        oof = np.zeros((len(y), 5), dtype=np.float64)
        for tr_idx, va_idx in folds:
            m = _fit_evgs_bridge(X_bridge[tr_idx], y[tr_idx], C=C)
            probs = np.zeros((len(va_idx), 5))
            for j, cls in enumerate(m.classes_):
                probs[:, int(cls)] = m.predict_proba(X_bridge[va_idx])[:, j]
            oof[va_idx] = probs
        preds = oof.argmax(axis=1)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, labels=list(range(5)), average="macro", zero_division=0)
        s2 = (acc + f1) / 2
        bridge_oof_by_C[C] = oof
        print(f"  C={C:>5}: Acc={acc:.4f}  F1={f1:.4f}  S_2={s2:.4f}")
        if s2 > best_C_s2:
            best_C_s2 = s2
            best_C = C

    print(f"\nBest C: {best_C} with S_2={best_C_s2:.4f}")
    oof_bridge = bridge_oof_by_C[best_C]

    # Now run the OTHER ensemble members (LGB-MC + kNN + LR-on-features) for blending.
    oof_lgb = np.zeros((len(y), 5))
    oof_knn = np.zeros((len(y), 5))
    oof_lr = np.zeros((len(y), 5))
    for tr_idx, va_idx in folds:
        sc = StandardScaler().fit(X_full[tr_idx])
        m_lgb = _fit_lgb_multi(X_full[tr_idx], y[tr_idx], n_class=5)
        oof_lgb[va_idx] = m_lgb.predict(X_full[va_idx])
        m_knn = _fit_knn(sc.transform(X_full[tr_idx]), y[tr_idx], k=3)
        kp = np.zeros((len(va_idx), 5))
        for j, cls in enumerate(m_knn.classes_):
            kp[:, int(cls)] = m_knn.predict_proba(sc.transform(X_full[va_idx]))[:, j]
        oof_knn[va_idx] = kp
        m_lr = _fit_lr(sc.transform(X_full[tr_idx]), y[tr_idx])
        lp = np.zeros((len(va_idx), 5))
        for j, cls in enumerate(m_lr.classes_):
            lp[:, int(cls)] = m_lr.predict_proba(sc.transform(X_full[va_idx]))[:, j]
        oof_lr[va_idx] = lp

    def score(probs):
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, labels=list(range(5)), average="macro", zero_division=0)
        return acc, f1, (acc + f1) / 2

    print()
    print("Individual model OOF (with 35-dim bridge):")
    for name, probs in (("lgb", oof_lgb), ("knn", oof_knn), ("lr", oof_lr), ("bridge35", oof_bridge)):
        acc, f1, s2 = score(probs)
        print(f"  {name:<10}  acc={acc:.3f}  f1={f1:.3f}  S2={s2:.3f}")

    # Grid search ensemble weights over 4 models summing to 1.
    grid = np.arange(0.0, 1.01, 0.1)
    best = (-1.0, None)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1: continue
            for w3 in grid:
                if w1 + w2 + w3 > 1: continue
                w4 = 1 - (w1 + w2 + w3)
                if w4 < 0 or w4 > 1: continue
                blend = w1*oof_lgb + w2*oof_knn + w3*oof_lr + w4*oof_bridge
                acc, f1, s2 = score(blend)
                if s2 > best[0]:
                    best = (s2, (w1, w2, w3, w4, acc, f1))
    s2, (w1, w2, w3, w4, acc, f1) = best
    print(f"\nBEST (lgb,knn,lr,bridge35)=({w1:.1f},{w2:.1f},{w3:.1f},{w4:.1f})  Acc={acc:.4f}  F1={f1:.4f}  S2={s2:.4f}")

    baseline = 0.4570  # from current track2_finalize_with_pseudo run
    print(f"Baseline (17-dim bridge): {baseline:.4f}")
    if s2 > baseline:
        delta = s2 - baseline
        print(f"35-DIM BRIDGE IMPROVED Track 2: +{delta:.4f}")
    else:
        print(f"Did NOT improve over 17-dim baseline; bridge expansion is wash or noise on this CV.")

    # Predict on full 220 limbs and save (independent of OOF lift decision; user controls submission).
    X_full_all = all_df_b[feature_cols].to_numpy(dtype=np.float32)
    X_bridge_all = all_df_b[bridge_cols].to_numpy(dtype=np.float32)
    sc_full = StandardScaler().fit(X_full)
    m_lgb_full = _fit_lgb_multi(X_full, y, n_class=5)
    m_knn_full = _fit_knn(sc_full.transform(X_full), y, k=3)
    m_lr_full = _fit_lr(sc_full.transform(X_full), y)
    m_bridge_full = _fit_evgs_bridge(X_bridge, y, C=best_C)

    def predict_probs(model, X_in, n_cls=5):
        out = np.zeros((X_in.shape[0], n_cls))
        for j, cls in enumerate(model.classes_):
            out[:, int(cls)] = model.predict_proba(X_in)[:, j]
        return out

    p_lgb = m_lgb_full.predict(X_full_all)
    p_knn = predict_probs(m_knn_full, sc_full.transform(X_full_all))
    p_lr = predict_probs(m_lr_full, sc_full.transform(X_full_all))
    p_bridge = predict_probs(m_bridge_full, X_bridge_all)
    blend = w1 * p_lgb + w2 * p_knn + w3 * p_lr + w4 * p_bridge
    preds_idx = blend.argmax(axis=1)

    out_df = pd.DataFrame({
        "patient_id": all_df_b["patient_id"].values,
        "side": all_df_b["side"].values,
        "subtype": [idx_to_label[int(i)] for i in preds_idx],
    })
    for ci, cn in enumerate(CLASSES):
        out_df[f"prob_{cn}"] = blend[:, ci]
    out_df.to_parquet(cfg.CACHE_DIR / "track2_preds_step1.parquet", index=False)

    (cfg.CACHE_DIR / "step1_summary.json").write_text(json.dumps({
        "best_C": best_C, "best_bridge_only_s2": float(best_C_s2),
        "final_s2": float(s2), "baseline_s2": baseline,
        "delta": float(s2 - baseline),
        "weights": {"lgb": float(w1), "knn": float(w2), "lr": float(w3), "bridge35": float(w4)},
    }, indent=2))
    print(f"\nSaved cache/track2_preds_step1.parquet + step1_summary.json")


if __name__ == "__main__":
    main()
