"""Step 1, properly integrated: swap the 17-dim EVGS-only LR input for the 35-dim
EVGS bridge in the FULL pseudo-label pipeline. Compare to current Track 2 baseline."""
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
from src.track2_model import CLASSES, EVGS_ITEMS, _fit_knn, _fit_lgb_multi, _fit_lr, build_track2_dataset


def _fit_bridge35_lr(Xtr, ytr, C):
    return LogisticRegression(
        penalty="l2", C=C, multi_class="multinomial",
        solver="lbfgs", max_iter=2000, class_weight="balanced",
        random_state=cfg.CFG.seed,
    ).fit(Xtr, ytr)


def main(conf_thr: float = 0.5, lr_C: float = 1.0) -> None:
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    wnl_idx = label_to_idx["WNL"]

    train_df, all_df, feature_cols_full = build_track2_dataset()
    bridge_df = build_bridge_table(use_oof_for_train=True)
    bridge_cols = bridge_feature_cols()

    # Attach bridge cols
    train_df = train_df.merge(
        bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left"
    )
    all_df_b = all_df.merge(
        bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left"
    )

    feature_cols = [c for c in train_df.columns
                    if c not in ("patient_id", "side", "y") and c not in bridge_cols]
    train_pids_set = set(train_df.patient_id.unique().tolist())

    # First: train initial ensemble with bridge35 on ORIGINAL 44 limbs only (no pseudo yet)
    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_bridge = train_df[bridge_cols].to_numpy(dtype=np.float32)
    y = np.array([label_to_idx[s] for s in train_df["y"].values], dtype=np.int64)
    pids = train_df["patient_id"].to_numpy()

    print(f"Initial training: {len(train_df)} limbs. Full features: {len(feature_cols)}. Bridge35: {len(bridge_cols)}")

    folds = list(cvmod.leave_one_patient_out(pids))
    oof_lgb = np.zeros((len(y), 5))
    oof_knn = np.zeros((len(y), 5))
    oof_lr = np.zeros((len(y), 5))
    oof_bridge35 = np.zeros((len(y), 5))

    for tr_idx, va_idx in folds:
        sc = StandardScaler().fit(X[tr_idx])
        m_lgb = _fit_lgb_multi(X[tr_idx], y[tr_idx], n_class=5)
        oof_lgb[va_idx] = m_lgb.predict(X[va_idx])
        m_knn = _fit_knn(sc.transform(X[tr_idx]), y[tr_idx], k=3)
        kp = np.zeros((len(va_idx), 5))
        for j, cls in enumerate(m_knn.classes_):
            kp[:, int(cls)] = m_knn.predict_proba(sc.transform(X[va_idx]))[:, j]
        oof_knn[va_idx] = kp
        m_lr = _fit_lr(sc.transform(X[tr_idx]), y[tr_idx])
        lp = np.zeros((len(va_idx), 5))
        for j, cls in enumerate(m_lr.classes_):
            lp[:, int(cls)] = m_lr.predict_proba(sc.transform(X[va_idx]))[:, j]
        oof_lr[va_idx] = lp
        m_b = _fit_bridge35_lr(X_bridge[tr_idx], y[tr_idx], C=lr_C)
        bp = np.zeros((len(va_idx), 5))
        for j, cls in enumerate(m_b.classes_):
            bp[:, int(cls)] = m_b.predict_proba(X_bridge[va_idx])[:, j]
        oof_bridge35[va_idx] = bp

    def score(probs):
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, labels=list(range(5)), average="macro", zero_division=0)
        return acc, f1, (acc + f1) / 2

    # Search ensemble weights on initial (44-limb) data
    grid = np.arange(0.0, 1.01, 0.1)
    best = (-1.0, None)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1: continue
            for w3 in grid:
                if w1 + w2 + w3 > 1: continue
                w4 = 1 - (w1 + w2 + w3)
                if w4 < 0 or w4 > 1: continue
                blend = w1*oof_lgb + w2*oof_knn + w3*oof_lr + w4*oof_bridge35
                acc, f1, s2 = score(blend)
                if s2 > best[0]:
                    best = (s2, (w1, w2, w3, w4, acc, f1))
    s2_init, (w1, w2, w3, w4, acc_i, f1_i) = best
    print(f"INITIAL (no pseudo) bridge35 ensemble: weights=({w1:.1f},{w2:.1f},{w3:.1f},{w4:.1f})")
    print(f"  Acc={acc_i:.4f}  F1={f1_i:.4f}  S2={s2_init:.4f}")

    # Refit on full 44 limbs and predict on all 220 to source pseudo-labels
    sc_full = StandardScaler().fit(X)
    m_lgb_full = _fit_lgb_multi(X, y, n_class=5)
    m_knn_full = _fit_knn(sc_full.transform(X), y, k=3)
    m_lr_full = _fit_lr(sc_full.transform(X), y)
    m_bridge_full = _fit_bridge35_lr(X_bridge, y, C=lr_C)
    X_full_all = all_df_b[feature_cols].to_numpy(dtype=np.float32)
    X_bridge_all = all_df_b[bridge_cols].to_numpy(dtype=np.float32)

    def predict_probs(model, X_in, n_cls=5):
        out = np.zeros((X_in.shape[0], n_cls))
        for j, cls in enumerate(model.classes_):
            out[:, int(cls)] = model.predict_proba(X_in)[:, j]
        return out

    p_lgb = m_lgb_full.predict(X_full_all)
    p_knn = predict_probs(m_knn_full, sc_full.transform(X_full_all))
    p_lr = predict_probs(m_lr_full, sc_full.transform(X_full_all))
    p_bridge = predict_probs(m_bridge_full, X_bridge_all)
    blend_all = w1*p_lgb + w2*p_knn + w3*p_lr + w4*p_bridge

    # Pseudo-label: NOT in train, NOT in test, max prob >= conf_thr
    test_pids = set(cfg.TRACK2_TEST_IDS)
    all_pids = all_df_b["patient_id"].values
    all_sides = all_df_b["side"].values
    eligible = []
    for i in range(len(all_pids)):
        if all_pids[i] in train_pids_set or all_pids[i] in test_pids:
            continue
        mp = blend_all[i].max()
        if mp >= conf_thr:
            eligible.append((int(all_pids[i]), all_sides[i], int(blend_all[i].argmax()), mp))

    print(f"Pseudo-labels at conf>={conf_thr}: {len(eligible)} limbs")
    if eligible:
        from collections import Counter as _C
        print(f"  distribution: {dict(_C(idx_to_label[lab] for _, _, lab, _ in eligible))}")

    # Build augmented training set
    aug_rows = []
    all_b_indexed = all_df_b.set_index(["patient_id", "side"])
    for pid, side, lab_idx, _ in eligible:
        key = (int(pid), side)
        if key not in all_b_indexed.index:
            continue
        base = all_b_indexed.loc[key].to_dict()
        base["patient_id"] = int(pid); base["side"] = side
        base["y"] = idx_to_label[lab_idx]
        aug_rows.append(base)
    aug_df = pd.DataFrame(aug_rows)
    augmented = pd.concat([train_df, aug_df], ignore_index=True)
    print(f"Augmented training: {len(augmented)} limbs (originals + pseudo)")

    # Retrain with LOPO on ORIGINAL 22 patients only.
    X_a = augmented[feature_cols].to_numpy(dtype=np.float32)
    X_a_b = augmented[bridge_cols].to_numpy(dtype=np.float32)
    y_a = np.array([label_to_idx[s] for s in augmented["y"].values], dtype=np.int64)
    pids_a = augmented["patient_id"].to_numpy()
    original_mask = np.array([int(p) in train_pids_set for p in pids_a])

    oof_lgb = np.zeros((len(y_a), 5))
    oof_knn = np.zeros((len(y_a), 5))
    oof_lr = np.zeros((len(y_a), 5))
    oof_bridge35 = np.zeros((len(y_a), 5))

    for held_pid in sorted(train_pids_set):
        tr_idx = np.where(pids_a != held_pid)[0]
        va_idx = np.where((pids_a == held_pid) & original_mask)[0]
        if len(va_idx) == 0:
            continue
        sc = StandardScaler().fit(X_a[tr_idx])
        m_lgb = _fit_lgb_multi(X_a[tr_idx], y_a[tr_idx], n_class=5)
        oof_lgb[va_idx] = m_lgb.predict(X_a[va_idx])
        m_knn = _fit_knn(sc.transform(X_a[tr_idx]), y_a[tr_idx], k=3)
        kp = np.zeros((len(va_idx), 5))
        for j, cls in enumerate(m_knn.classes_):
            kp[:, int(cls)] = m_knn.predict_proba(sc.transform(X_a[va_idx]))[:, j]
        oof_knn[va_idx] = kp
        m_lr = _fit_lr(sc.transform(X_a[tr_idx]), y_a[tr_idx])
        lp = np.zeros((len(va_idx), 5))
        for j, cls in enumerate(m_lr.classes_):
            lp[:, int(cls)] = m_lr.predict_proba(sc.transform(X_a[va_idx]))[:, j]
        oof_lr[va_idx] = lp
        m_b = _fit_bridge35_lr(X_a_b[tr_idx], y_a[tr_idx], C=lr_C)
        bp = np.zeros((len(va_idx), 5))
        for j, cls in enumerate(m_b.classes_):
            bp[:, int(cls)] = m_b.predict_proba(X_a_b[va_idx])[:, j]
        oof_bridge35[va_idx] = bp

    def score_aug(weights):
        probs = weights[0]*oof_lgb + weights[1]*oof_knn + weights[2]*oof_lr + weights[3]*oof_bridge35
        preds = probs.argmax(axis=1)
        preds_e = preds[original_mask]
        y_e = y_a[original_mask]
        acc = accuracy_score(y_e, preds_e)
        f1 = f1_score(y_e, preds_e, labels=list(range(5)), average="macro", zero_division=0)
        return acc, f1, (acc + f1) / 2

    best2 = (-1.0, None)
    for ww1 in grid:
        for ww2 in grid:
            if ww1 + ww2 > 1: continue
            for ww3 in grid:
                if ww1 + ww2 + ww3 > 1: continue
                ww4 = 1 - (ww1 + ww2 + ww3)
                if ww4 < 0 or ww4 > 1: continue
                acc, f1, s2 = score_aug((ww1, ww2, ww3, ww4))
                if s2 > best2[0]:
                    best2 = (s2, (ww1, ww2, ww3, ww4, acc, f1))
    s2, (W1, W2, W3, W4, acc_f, f1_f) = best2
    print(f"\nFINAL (with pseudo) bridge35 ensemble: weights=({W1:.1f},{W2:.1f},{W3:.1f},{W4:.1f})")
    print(f"  Acc={acc_f:.4f}  F1={f1_f:.4f}  S2={s2:.4f}")

    # Baseline (current cache)
    baseline_s2 = 0.4570
    delta = s2 - baseline_s2
    print(f"\nBaseline (17-dim, current cache): {baseline_s2:.4f}")
    print(f"Delta: {delta:+.4f}")

    # Refit on full augmented + predict full + save
    sc_a = StandardScaler().fit(X_a)
    m_lgb_a = _fit_lgb_multi(X_a, y_a, n_class=5)
    m_knn_a = _fit_knn(sc_a.transform(X_a), y_a, k=3)
    m_lr_a = _fit_lr(sc_a.transform(X_a), y_a)
    m_b_a = _fit_bridge35_lr(X_a_b, y_a, C=lr_C)

    P_lgb = m_lgb_a.predict(X_full_all)
    P_knn = predict_probs(m_knn_a, sc_a.transform(X_full_all))
    P_lr = predict_probs(m_lr_a, sc_a.transform(X_full_all))
    P_b = predict_probs(m_b_a, X_bridge_all)
    P_blend = W1*P_lgb + W2*P_knn + W3*P_lr + W4*P_b
    preds_idx = P_blend.argmax(axis=1)

    out_df = pd.DataFrame({
        "patient_id": all_df_b["patient_id"].values,
        "side": all_df_b["side"].values,
        "subtype": [idx_to_label[int(i)] for i in preds_idx],
    })
    for ci, cn in enumerate(CLASSES):
        out_df[f"prob_{cn}"] = P_blend[:, ci]
    out_df.to_parquet(cfg.CACHE_DIR / "track2_preds_step1.parquet", index=False)

    if s2 > baseline_s2:
        out_df.to_parquet(cfg.CACHE_DIR / "track2_preds.parquet", index=False)
        print(f"\nSTEP 1 IMPROVED: {baseline_s2:.4f} -> {s2:.4f}. Updated cache/track2_preds.parquet")
    else:
        print(f"\nStep 1 did NOT improve over baseline. cache/track2_preds.parquet unchanged.")

    (cfg.CACHE_DIR / "step1_summary.json").write_text(json.dumps({
        "lr_C": lr_C, "conf_thr": conf_thr,
        "initial_s2": float(s2_init), "augmented_s2": float(s2),
        "baseline_s2": baseline_s2, "delta": float(delta),
        "weights": {"lgb": float(W1), "knn": float(W2), "lr": float(W3), "bridge35": float(W4)},
        "n_pseudo": len(eligible),
    }, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--conf-thr", type=float, default=0.5)
    args = p.parse_args()
    main(conf_thr=args.conf_thr, lr_C=args.C)
