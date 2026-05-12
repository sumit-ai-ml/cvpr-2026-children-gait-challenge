"""Step 2 redone: hierarchical head trained on the augmented (pseudo-labeled) pool
from Step 1. Fairer comparison."""
from __future__ import annotations

import json
import sys
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
    spw = max((y_bin == 0).sum() / max((y_bin == 1).sum(), 1), 1.0)
    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.05, num_leaves=15, max_depth=4, min_child_samples=2,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=3, lambda_l2=1.0,
        scale_pos_weight=spw, verbose=-1, random_state=cfg.CFG.seed,
    )
    dtr = lgb.Dataset(X_full, label=y_bin)
    m_lgb = lgb.train(params, dtr, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
    m_lr = LogisticRegression(
        penalty="l2", C=10.0, solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=cfg.CFG.seed,
    ).fit(X_bridge, y_bin)
    return m_lgb, m_lr


def _pred_binary(m_lgb, m_lr, X_full, X_bridge):
    return 0.5 * m_lgb.predict(X_full) + 0.5 * m_lr.predict_proba(X_bridge)[:, 1]


def _fit_3way(X_full, X_bridge, y3):
    classes = sorted(np.unique(y3).tolist())
    counts = np.bincount(y3, minlength=max(classes) + 1)
    total = y3.shape[0]
    weights = np.array([total / (len(classes) * max(counts[c], 1)) for c in y3])
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


def _pred_3way(m_lgb, m_lr, X_full, X_bridge, n=5):
    p_lgb = m_lgb.predict(X_full)
    if p_lgb.shape[1] != n:
        padded = np.zeros((X_full.shape[0], n))
        padded[:, : p_lgb.shape[1]] = p_lgb
        p_lgb = padded
    p_lr_raw = m_lr.predict_proba(X_bridge)
    p_lr = np.zeros((X_bridge.shape[0], n))
    for j, c in enumerate(m_lr.classes_):
        p_lr[:, int(c)] = p_lr_raw[:, j]
    return 0.5 * p_lgb + 0.5 * p_lr


def main() -> None:
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    WNL, T4 = label_to_idx["WNL"], label_to_idx["type4"]
    T1, T2, T3 = label_to_idx["type1"], label_to_idx["type2"], label_to_idx["type3"]

    train_df, all_df, _ = build_track2_dataset()
    bridge_df = build_bridge_table(use_oof_for_train=True)
    bridge_cols = bridge_feature_cols()
    train_df = train_df.merge(bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left")
    all_df_b = all_df.merge(bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left")
    feature_cols = [c for c in train_df.columns
                    if c not in ("patient_id", "side", "y") and c not in bridge_cols]

    train_pids_set = set(train_df.patient_id.unique().tolist())
    test_pids_set = set(cfg.TRACK2_TEST_IDS)

    # === Pseudo-labeling step (replicate Step 1's pseudo-label generation) ===
    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    Xb = train_df[bridge_cols].to_numpy(dtype=np.float32)
    y = np.array([label_to_idx[s] for s in train_df["y"].values], dtype=np.int64)

    # Quick initial 4-model fit to source pseudo-labels (same as step1_full_pipeline)
    from src.track2_model import _fit_knn, _fit_lgb_multi, _fit_lr
    from sklearn.preprocessing import StandardScaler as SS
    sc_init = SS().fit(X)
    m_lgb_init = _fit_lgb_multi(X, y, n_class=5)
    m_knn_init = _fit_knn(sc_init.transform(X), y, k=3)
    m_lr_init = _fit_lr(sc_init.transform(X), y)
    m_b_init = LogisticRegression(
        penalty="l2", C=10.0, solver="lbfgs", max_iter=2000,
        multi_class="multinomial", class_weight="balanced", random_state=cfg.CFG.seed,
    ).fit(Xb, y)

    X_all = all_df_b[feature_cols].to_numpy(dtype=np.float32)
    Xb_all = all_df_b[bridge_cols].to_numpy(dtype=np.float32)

    def predict_probs(m, X_in):
        out = np.zeros((X_in.shape[0], 5))
        for j, c in enumerate(m.classes_):
            out[:, int(c)] = m.predict_proba(X_in)[:, j]
        return out

    p_lgb_a = m_lgb_init.predict(X_all)
    p_knn_a = predict_probs(m_knn_init, sc_init.transform(X_all))
    p_lr_a = predict_probs(m_lr_init, sc_init.transform(X_all))
    p_b_a = predict_probs(m_b_init, Xb_all)
    blend_a = 0.0 * p_lgb_a + 0.2 * p_knn_a + 0.0 * p_lr_a + 0.8 * p_b_a  # step1's best weights

    aug_rows = []
    all_b_indexed = all_df_b.set_index(["patient_id", "side"])
    for i in range(len(all_df_b)):
        pid = int(all_df_b.iloc[i]["patient_id"])
        side = all_df_b.iloc[i]["side"]
        if pid in train_pids_set or pid in test_pids_set:
            continue
        mp = blend_a[i].max()
        if mp >= 0.5:
            base = all_b_indexed.loc[(pid, side)].to_dict()
            base["patient_id"] = pid; base["side"] = side
            base["y"] = idx_to_label[int(blend_a[i].argmax())]
            aug_rows.append(base)
    aug_df = pd.DataFrame(aug_rows)
    augmented = pd.concat([train_df, aug_df], ignore_index=True)
    print(f"Augmented training: {len(augmented)} limbs")

    Xa = augmented[feature_cols].to_numpy(dtype=np.float32)
    Xab = augmented[bridge_cols].to_numpy(dtype=np.float32)
    ya = np.array([label_to_idx[s] for s in augmented["y"].values], dtype=np.int64)
    pids_a = augmented["patient_id"].to_numpy()
    totals_a = augmented["bridge_total"].to_numpy()
    original_mask = np.array([int(p) in train_pids_set for p in pids_a])

    # === Hierarchical OOF on ORIGINAL 22 patients with augmented training pool ===
    print(f"Class dist (incl pseudo): { {idx_to_label[c]: int((ya==c).sum()) for c in range(5)} }")
    oof_pWNL = np.zeros(len(ya))
    oof_pT4 = np.zeros(len(ya))
    oof_p3way = np.zeros((len(ya), 5))

    for held_pid in sorted(train_pids_set):
        tr_idx = np.where(pids_a != held_pid)[0]
        va_idx = np.where((pids_a == held_pid) & original_mask)[0]
        if len(va_idx) == 0:
            continue
        # A
        y_A = (ya[tr_idx] == WNL).astype(int)
        if y_A.sum() in (0, len(y_A)):
            oof_pWNL[va_idx] = 0.0
        else:
            m_lgb_A, m_lr_A = _fit_binary_ensemble(Xa[tr_idx], Xab[tr_idx], y_A)
            oof_pWNL[va_idx] = _pred_binary(m_lgb_A, m_lr_A, Xa[va_idx], Xab[va_idx])
        # B
        mask_NW = (ya[tr_idx] != WNL)
        y_B = (ya[tr_idx][mask_NW] == T4).astype(int)
        if y_B.sum() in (0, len(y_B)):
            oof_pT4[va_idx] = 0.0
        else:
            m_lgb_B, m_lr_B = _fit_binary_ensemble(Xa[tr_idx][mask_NW], Xab[tr_idx][mask_NW], y_B)
            oof_pT4[va_idx] = _pred_binary(m_lgb_B, m_lr_B, Xa[va_idx], Xab[va_idx])
        # C
        mask_3 = mask_NW & (ya[tr_idx] != T4)
        y_3 = ya[tr_idx][mask_3]
        if len(np.unique(y_3)) < 2:
            oof_p3way[va_idx] = 0.0
        else:
            m_lgb_C, m_lr_C = _fit_3way(Xa[tr_idx][mask_3], Xab[tr_idx][mask_3], y_3)
            oof_p3way[va_idx] = _pred_3way(m_lgb_C, m_lr_C, Xa[va_idx], Xab[va_idx])

    # Score with tuned thresholds
    y_orig = ya[original_mask]
    def decode(thr_wnl, thr_total, thr_t4):
        preds = np.zeros(len(ya), dtype=np.int64)
        for i in range(len(ya)):
            if oof_pWNL[i] >= thr_wnl or totals_a[i] <= thr_total:
                preds[i] = WNL
            elif oof_pT4[i] >= thr_t4:
                preds[i] = T4
            else:
                p3 = oof_p3way[i].copy()
                mask3 = np.zeros(5); mask3[T1] = mask3[T2] = mask3[T3] = 1
                preds[i] = int(np.argmax(p3 * mask3))
        return preds[original_mask]

    def score_p(preds):
        acc = accuracy_score(y_orig, preds)
        f1 = f1_score(y_orig, preds, labels=list(range(5)), average="macro", zero_division=0)
        return acc, f1, (acc + f1) / 2

    best = (-1.0, None)
    print()
    print("Tuning thresholds (thr_wnl, thr_total, thr_t4) on LOPO over original 22...")
    for thr_wnl in np.arange(0.30, 0.71, 0.05):
        for thr_total in (4, 5, 6, 7, 8):
            for thr_t4 in np.arange(0.30, 0.71, 0.05):
                p = decode(thr_wnl, thr_total, thr_t4)
                acc, f1, s2 = score_p(p)
                if s2 > best[0]:
                    best = (s2, (float(thr_wnl), int(thr_total), float(thr_t4), acc, f1))
    s2, (thr_wnl, thr_total, thr_t4, acc, f1) = best
    print(f"BEST thresholds: thr_wnl={thr_wnl:.2f}, thr_total={thr_total}, thr_t4={thr_t4:.2f}")
    print(f"  OOF Acc={acc:.4f}  F1={f1:.4f}  S2={s2:.4f}")

    baseline = 0.5405
    delta = s2 - baseline
    print(f"\nStep 1 baseline: {baseline:.4f}")
    print(f"Delta: {delta:+.4f}")

    # Refit + predict on all 220
    y_A_full = (ya == WNL).astype(int)
    m_lgb_A, m_lr_A = _fit_binary_ensemble(Xa, Xab, y_A_full)
    mask_NW = (ya != WNL)
    y_B_full = (ya[mask_NW] == T4).astype(int)
    m_lgb_B, m_lr_B = _fit_binary_ensemble(Xa[mask_NW], Xab[mask_NW], y_B_full)
    mask_3 = mask_NW & (ya != T4)
    m_lgb_C, m_lr_C = _fit_3way(Xa[mask_3], Xab[mask_3], ya[mask_3])

    pWNL_all = _pred_binary(m_lgb_A, m_lr_A, X_all, Xb_all)
    pT4_all = _pred_binary(m_lgb_B, m_lr_B, X_all, Xb_all)
    p3way_all = _pred_3way(m_lgb_C, m_lr_C, X_all, Xb_all)
    totals_all = all_df_b["bridge_total"].to_numpy()

    rows = []
    for i in range(len(X_all)):
        if pWNL_all[i] >= thr_wnl or totals_all[i] <= thr_total:
            pred = WNL
        elif pT4_all[i] >= thr_t4:
            pred = T4
        else:
            p3 = p3way_all[i].copy()
            mask3 = np.zeros(5); mask3[T1] = mask3[T2] = mask3[T3] = 1
            pred = int(np.argmax(p3 * mask3))
        # build 5-vector probs
        probs5 = np.zeros(5)
        probs5[WNL] = pWNL_all[i]
        probs5[T4] = (1 - pWNL_all[i]) * pT4_all[i]
        remain = (1 - pWNL_all[i]) * (1 - pT4_all[i])
        if p3way_all[i, [T1, T2, T3]].sum() > 0:
            for c in (T1, T2, T3):
                probs5[c] = remain * p3way_all[i, c] / p3way_all[i, [T1, T2, T3]].sum()
        rec = {
            "patient_id": int(all_df_b.iloc[i]["patient_id"]),
            "side": all_df_b.iloc[i]["side"],
            "subtype": idx_to_label[int(pred)],
            "pWNL": float(pWNL_all[i]),
            "pT4": float(pT4_all[i]),
            "pred_total": int(totals_all[i]),
        }
        for ci, cn in enumerate(CLASSES):
            rec[f"prob_{cn}"] = float(probs5[ci])
        rows.append(rec)
    out_df = pd.DataFrame(rows)
    out_path = cfg.CACHE_DIR / "track2_preds_step2.parquet"
    out_df.to_parquet(out_path, index=False)

    if s2 > baseline:
        out_df.to_parquet(cfg.CACHE_DIR / "track2_preds.parquet", index=False)
        print(f"\nSTEP 2 IMPROVED: {baseline:.4f} -> {s2:.4f}. Updated cache/track2_preds.parquet")
    else:
        print(f"\nStep 2 did NOT improve over Step 1. cache/track2_preds.parquet unchanged.")

    (cfg.CACHE_DIR / "step2_summary.json").write_text(json.dumps({
        "thr_wnl": thr_wnl, "thr_total": thr_total, "thr_t4": thr_t4,
        "oof_acc": float(acc), "oof_f1": float(f1), "oof_s2": float(s2),
        "baseline_s2": baseline, "delta": float(delta),
    }, indent=2))


if __name__ == "__main__":
    main()
