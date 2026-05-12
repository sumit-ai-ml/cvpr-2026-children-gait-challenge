"""Step 2: Hierarchical Track 2 head.

3 classifiers, each LightGBM + EVGS-bridge LR (the 35-dim bridge from Step 1):
  A: WNL vs not-WNL (binary, on all 22 patients)
  B: type4 vs not-type4 (binary, on the 20 not-WNL patients)
  C: type1 vs type2 vs type3 (3-way, on 18 patients excluding WNL and type4)

Inference order:
  Run A. If P(WNL) >= thr_wnl OR predicted_total <= thr_total -> WNL.
  Else run B. If P(type4) >= thr_t4 -> type4.
  Else run C. Output argmax.

Tune (thr_wnl, thr_total, thr_t4) on LOPO.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import cv as cvmod
from src.evgs_bridge import bridge_feature_cols, build_bridge_table
from src.track2_model import CLASSES, build_track2_dataset


def _fit_binary_ensemble(X_full, X_bridge, y_bin):
    """Fit LightGBM (on full features) + LR (on 35-dim bridge), return both. Equal blend."""
    # LightGBM binary
    spw = max((y_bin == 0).sum() / max((y_bin == 1).sum(), 1), 1.0)
    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.05, num_leaves=15, max_depth=4, min_child_samples=2,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=3, lambda_l2=1.0,
        scale_pos_weight=spw, verbose=-1, random_state=cfg.CFG.seed,
    )
    dtr = lgb.Dataset(X_full, label=y_bin)
    m_lgb = lgb.train(params, dtr, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
    # LR binary on bridge
    m_lr = LogisticRegression(
        penalty="l2", C=10.0, solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=cfg.CFG.seed,
    ).fit(X_bridge, y_bin)
    return m_lgb, m_lr


def _pred_binary_ensemble(m_lgb, m_lr, X_full, X_bridge) -> np.ndarray:
    p_lgb = m_lgb.predict(X_full)
    p_lr = m_lr.predict_proba(X_bridge)[:, 1]
    return 0.5 * p_lgb + 0.5 * p_lr


def _fit_3way(X_full, X_bridge, y3):
    """LightGBM 3-class + LR 3-class on bridge. Equal blend."""
    classes = sorted(np.unique(y3).tolist())
    n_cls = len(classes)
    counts = np.bincount(y3, minlength=max(classes) + 1)
    total = y3.shape[0]
    weights = np.array([total / (n_cls * max(counts[c], 1)) for c in y3])
    params = dict(
        objective="multiclass", num_class=max(classes) + 1, metric="multi_logloss",
        learning_rate=0.05, num_leaves=15, max_depth=4, min_child_samples=2,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=3, lambda_l2=1.0,
        verbose=-1, random_state=cfg.CFG.seed,
    )
    dtr = lgb.Dataset(X_full, label=y3, weight=weights)
    m_lgb = lgb.train(params, dtr, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
    m_lr = LogisticRegression(
        penalty="l2", C=10.0, solver="lbfgs", max_iter=2000,
        multi_class="multinomial", class_weight="balanced", random_state=cfg.CFG.seed,
    ).fit(X_bridge, y3)
    return m_lgb, m_lr


def _pred_3way(m_lgb, m_lr, X_full, X_bridge, n_total_cls: int = 5) -> np.ndarray:
    p_lgb = m_lgb.predict(X_full)
    # m_lr.classes_ has only the 3 classes used. Reshape to full 5-dim.
    p_lr_raw = m_lr.predict_proba(X_bridge)
    p_lr = np.zeros((X_bridge.shape[0], n_total_cls))
    for j, c in enumerate(m_lr.classes_):
        p_lr[:, int(c)] = p_lr_raw[:, j]
    # p_lgb is already (N, n_total_cls)
    if p_lgb.shape[1] != n_total_cls:
        # If lgb didn't include all classes, pad
        padded = np.zeros((X_bridge.shape[0], n_total_cls))
        padded[:, : p_lgb.shape[1]] = p_lgb
        p_lgb = padded
    return 0.5 * p_lgb + 0.5 * p_lr


def main() -> None:
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    WNL_IDX = label_to_idx["WNL"]
    T4_IDX = label_to_idx["type4"]
    T1_IDX, T2_IDX, T3_IDX = label_to_idx["type1"], label_to_idx["type2"], label_to_idx["type3"]

    train_df, all_df, feature_cols_full = build_track2_dataset()
    bridge_df = build_bridge_table(use_oof_for_train=True)
    bridge_cols = bridge_feature_cols()

    train_df = train_df.merge(
        bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left"
    )
    all_df_b = all_df.merge(
        bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left"
    )
    feature_cols = [c for c in train_df.columns
                    if c not in ("patient_id", "side", "y") and c not in bridge_cols]

    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    Xb = train_df[bridge_cols].to_numpy(dtype=np.float32)
    y = np.array([label_to_idx[s] for s in train_df["y"].values], dtype=np.int64)
    pids = train_df["patient_id"].to_numpy()
    # Predicted total = bridge_total column
    totals = train_df["bridge_total"].to_numpy()

    print(f"Track 2 train: {len(train_df)} limbs")
    print(f"Class dist: { {idx_to_label[c]: int((y==c).sum()) for c in range(5)} }")

    folds = list(cvmod.leave_one_patient_out(pids))
    # OOF P(WNL), P(type4 | not-WNL), P(class | not-WNL & not-type4)
    oof_pWNL = np.zeros(len(y))
    oof_pT4 = np.zeros(len(y))
    oof_p3way = np.zeros((len(y), 5))  # full 5-dim, only type1/2/3 filled

    for tr_idx, va_idx in folds:
        # A: WNL vs not
        y_A = (y[tr_idx] == WNL_IDX).astype(int)
        if y_A.sum() == 0 or y_A.sum() == len(y_A):
            # Held-out is the only WNL patient; cannot train A as binary, fall back to 0.
            oof_pWNL[va_idx] = 0.0
        else:
            m_lgb_A, m_lr_A = _fit_binary_ensemble(X[tr_idx], Xb[tr_idx], y_A)
            oof_pWNL[va_idx] = _pred_binary_ensemble(m_lgb_A, m_lr_A, X[va_idx], Xb[va_idx])
        # B: type4 vs not, trained on NOT-WNL training patients
        mask_not_WNL = (y[tr_idx] != WNL_IDX)
        Xb_tr_NW = Xb[tr_idx][mask_not_WNL]
        X_tr_NW = X[tr_idx][mask_not_WNL]
        y_NW = y[tr_idx][mask_not_WNL]
        y_B = (y_NW == T4_IDX).astype(int)
        if y_B.sum() == 0 or y_B.sum() == len(y_B):
            oof_pT4[va_idx] = 0.0
        else:
            m_lgb_B, m_lr_B = _fit_binary_ensemble(X_tr_NW, Xb_tr_NW, y_B)
            oof_pT4[va_idx] = _pred_binary_ensemble(m_lgb_B, m_lr_B, X[va_idx], Xb[va_idx])
        # C: 3-way over type1/2/3, on patients excluding WNL and type4
        mask_3 = mask_not_WNL & (y[tr_idx] != T4_IDX)
        X_tr_3 = X[tr_idx][mask_3]
        Xb_tr_3 = Xb[tr_idx][mask_3]
        y_3 = y[tr_idx][mask_3]
        if len(np.unique(y_3)) < 2:
            oof_p3way[va_idx] = 0.0
        else:
            m_lgb_C, m_lr_C = _fit_3way(X_tr_3, Xb_tr_3, y_3)
            oof_p3way[va_idx] = _pred_3way(m_lgb_C, m_lr_C, X[va_idx], Xb[va_idx])

    # Tune hierarchical thresholds (thr_wnl, thr_total, thr_t4) on LOPO
    def decode(thr_wnl, thr_total, thr_t4):
        preds = np.zeros(len(y), dtype=np.int64)
        for i in range(len(y)):
            if oof_pWNL[i] >= thr_wnl or totals[i] <= thr_total:
                preds[i] = WNL_IDX
            elif oof_pT4[i] >= thr_t4:
                preds[i] = T4_IDX
            else:
                p3 = oof_p3way[i].copy()
                # restrict argmax to type1/2/3
                mask3 = np.zeros(5)
                mask3[T1_IDX] = mask3[T2_IDX] = mask3[T3_IDX] = 1
                p3 = p3 * mask3
                preds[i] = int(np.argmax(p3))
        return preds

    def score_preds(preds):
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, labels=list(range(5)), average="macro", zero_division=0)
        return acc, f1, (acc + f1) / 2

    best = (-1.0, None)
    print()
    print("Tuning thresholds (thr_wnl, thr_total, thr_t4)...")
    for thr_wnl in np.arange(0.30, 0.71, 0.05):
        for thr_total in (4, 5, 6, 7, 8):
            for thr_t4 in np.arange(0.30, 0.71, 0.05):
                preds = decode(thr_wnl, thr_total, thr_t4)
                acc, f1, s2 = score_preds(preds)
                if s2 > best[0]:
                    best = (s2, (float(thr_wnl), int(thr_total), float(thr_t4), acc, f1))
    s2, (thr_wnl, thr_total, thr_t4, acc, f1) = best
    print(f"BEST thresholds: thr_wnl={thr_wnl:.2f}, thr_total={thr_total}, thr_t4={thr_t4:.2f}")
    print(f"  OOF Acc={acc:.4f}  F1={f1:.4f}  S2={s2:.4f}")

    baseline = 0.5405  # from step1
    print(f"Step 1 baseline: {baseline:.4f}")
    print(f"Delta: {s2 - baseline:+.4f}")

    # Now refit on full train and produce test predictions
    print("\nRefitting on full training set and predicting on all 220 limbs...")
    # A: WNL vs not on full
    y_A_full = (y == WNL_IDX).astype(int)
    m_lgb_A, m_lr_A = _fit_binary_ensemble(X, Xb, y_A_full)
    # B: type4 vs not, on not-WNL
    mask_NW = (y != WNL_IDX)
    y_B_full = (y[mask_NW] == T4_IDX).astype(int)
    m_lgb_B, m_lr_B = _fit_binary_ensemble(X[mask_NW], Xb[mask_NW], y_B_full)
    # C: 3-way over 1/2/3
    mask_3 = mask_NW & (y != T4_IDX)
    m_lgb_C, m_lr_C = _fit_3way(X[mask_3], Xb[mask_3], y[mask_3])

    X_all = all_df_b[feature_cols].to_numpy(dtype=np.float32)
    Xb_all = all_df_b[bridge_cols].to_numpy(dtype=np.float32)
    totals_all = all_df_b["bridge_total"].to_numpy()

    pWNL_all = _pred_binary_ensemble(m_lgb_A, m_lr_A, X_all, Xb_all)
    pT4_all = _pred_binary_ensemble(m_lgb_B, m_lr_B, X_all, Xb_all)
    p3way_all = _pred_3way(m_lgb_C, m_lr_C, X_all, Xb_all)

    rows = []
    for i in range(len(X_all)):
        if pWNL_all[i] >= thr_wnl or totals_all[i] <= thr_total:
            pred = WNL_IDX
        elif pT4_all[i] >= thr_t4:
            pred = T4_IDX
        else:
            p3 = p3way_all[i].copy()
            mask3 = np.zeros(5)
            mask3[T1_IDX] = mask3[T2_IDX] = mask3[T3_IDX] = 1
            p3 = p3 * mask3
            pred = int(np.argmax(p3))
        rec = {
            "patient_id": int(all_df_b.iloc[i]["patient_id"]),
            "side": all_df_b.iloc[i]["side"],
            "subtype": idx_to_label[int(pred)],
            "pWNL": float(pWNL_all[i]),
            "pT4": float(pT4_all[i]),
            "pred_total": int(totals_all[i]),
        }
        # full prob dist (combine the hierarchical signals into a 5-vector)
        probs5 = np.zeros(5)
        probs5[WNL_IDX] = pWNL_all[i]
        probs5[T4_IDX] = (1 - pWNL_all[i]) * pT4_all[i]
        remain = (1 - pWNL_all[i]) * (1 - pT4_all[i])
        for c in (T1_IDX, T2_IDX, T3_IDX):
            probs5[c] = remain * p3way_all[i, c] / max(p3way_all[i, [T1_IDX, T2_IDX, T3_IDX]].sum(), 1e-8)
        for ci, cn in enumerate(CLASSES):
            rec[f"prob_{cn}"] = float(probs5[ci])
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_path = cfg.CACHE_DIR / "track2_preds_step2.parquet"
    out_df.to_parquet(out_path, index=False)
    if s2 > baseline:
        out_df.to_parquet(cfg.CACHE_DIR / "track2_preds.parquet", index=False)
        print(f"STEP 2 IMPROVED: {baseline:.4f} -> {s2:.4f}. Updated cache/track2_preds.parquet")
    else:
        print(f"Step 2 did NOT improve over Step 1. cache/track2_preds.parquet unchanged.")

    (cfg.CACHE_DIR / "step2_summary.json").write_text(json.dumps({
        "thr_wnl": thr_wnl, "thr_total": thr_total, "thr_t4": thr_t4,
        "oof_acc": float(acc), "oof_f1": float(f1), "oof_s2": float(s2),
        "baseline_s2": baseline, "delta": float(s2 - baseline),
    }, indent=2))


if __name__ == "__main__":
    main()
