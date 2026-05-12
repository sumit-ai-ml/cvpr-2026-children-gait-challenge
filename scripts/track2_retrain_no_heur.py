"""Retrain Track 2 without the broken WNL heuristic (which collapsed probabilities to near-uniform).
The heuristic should never get >0 weight in the ensemble search."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import cv as cvmod
from src.track2_model import CLASSES, EVGS_ITEMS, _fit_evgs_only, _fit_knn, _fit_lgb_multi, _fit_lr, build_track2_dataset


def main() -> None:
    train_df, all_df, feature_cols = build_track2_dataset()
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    evgs_cols = [f"evgs_{it}" for it in EVGS_ITEMS]
    X_evgs = train_df[evgs_cols].to_numpy(dtype=np.float32)
    y = np.array([label_to_idx[s] for s in train_df["y"].values], dtype=np.int64)
    pids = train_df["patient_id"].to_numpy()

    folds = list(cvmod.leave_one_patient_out(pids))
    oof_lgb = np.zeros((len(y), 5))
    oof_knn = np.zeros((len(y), 5))
    oof_lr = np.zeros((len(y), 5))
    oof_evgs = np.zeros((len(y), 5))

    for fi, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr = y[tr_idx]
        sc = StandardScaler().fit(Xtr)
        m_lgb = _fit_lgb_multi(Xtr, ytr, n_class=5)
        oof_lgb[va_idx] = m_lgb.predict(Xva)
        m_knn = _fit_knn(sc.transform(Xtr), ytr, k=3)
        kp = np.zeros((Xva.shape[0], 5))
        for i, cls in enumerate(m_knn.classes_):
            kp[:, int(cls)] = m_knn.predict_proba(sc.transform(Xva))[:, i]
        oof_knn[va_idx] = kp
        m_lr = _fit_lr(sc.transform(Xtr), ytr)
        lp = np.zeros((Xva.shape[0], 5))
        for i, cls in enumerate(m_lr.classes_):
            lp[:, int(cls)] = m_lr.predict_proba(sc.transform(Xva))[:, i]
        oof_lr[va_idx] = lp
        m_evgs = _fit_evgs_only(X_evgs[tr_idx], ytr)
        ep = np.zeros((Xva.shape[0], 5))
        for i, cls in enumerate(m_evgs.classes_):
            ep[:, int(cls)] = m_evgs.predict_proba(X_evgs[va_idx])[:, i]
        oof_evgs[va_idx] = ep

    def score(probs):
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, labels=list(range(5)), average="macro", zero_division=0)
        return acc, f1, (acc + f1) / 2

    print("Individual models (no heuristic):")
    for name, probs in (("lgb", oof_lgb), ("knn", oof_knn), ("lr", oof_lr), ("evgs_only", oof_evgs)):
        acc, f1, s2 = score(probs)
        print(f"  {name:<10} Acc={acc:.3f}  F1={f1:.3f}  S2={s2:.3f}")

    # Grid over 4 weights summing to 1 (heuristic dropped).
    grid = np.arange(0.0, 1.01, 0.1)
    best = (-1.0, None)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1: continue
            for w3 in grid:
                if w1 + w2 + w3 > 1: continue
                w4 = 1 - (w1 + w2 + w3)
                if w4 < 0 or w4 > 1: continue
                blend = w1*oof_lgb + w2*oof_knn + w3*oof_lr + w4*oof_evgs
                acc, f1, s2 = score(blend)
                if s2 > best[0]:
                    best = (s2, (w1, w2, w3, w4, acc, f1))
    s2, (w1, w2, w3, w4, acc, f1) = best
    print(f"\nBEST (no heur): (lgb,knn,lr,evgs)=({w1:.1f},{w2:.1f},{w3:.1f},{w4:.1f})  Acc={acc:.4f}  F1={f1:.4f}  S2={s2:.4f}")

    # Refit on all 44 + predict all 220 limbs.
    sc_full = StandardScaler().fit(X)
    m_lgb_full = _fit_lgb_multi(X, y, n_class=5)
    m_knn_full = _fit_knn(sc_full.transform(X), y, k=3)
    m_lr_full = _fit_lr(sc_full.transform(X), y)
    m_evgs_full = _fit_evgs_only(X_evgs, y)

    X_all = all_df[feature_cols].to_numpy(dtype=np.float32)
    X_all_s = sc_full.transform(X_all)
    X_all_evgs = all_df[evgs_cols].to_numpy(dtype=np.float32)

    def predict_probs(model, X_in):
        out = np.zeros((X_in.shape[0], 5))
        if hasattr(model, "predict_proba"):
            for i, cls in enumerate(model.classes_):
                out[:, int(cls)] = model.predict_proba(X_in)[:, i]
        else:
            out = model.predict(X_in)
        return out

    p_lgb = predict_probs(m_lgb_full, X_all)
    p_knn = predict_probs(m_knn_full, X_all_s)
    p_lr = predict_probs(m_lr_full, X_all_s)
    p_evgs = predict_probs(m_evgs_full, X_all_evgs)
    blend = w1*p_lgb + w2*p_knn + w3*p_lr + w4*p_evgs
    preds_idx = blend.argmax(axis=1)
    preds_label = [idx_to_label[int(i)] for i in preds_idx]

    out_df = pd.DataFrame({
        "patient_id": all_df["patient_id"].values, "side": all_df["side"].values, "subtype": preds_label,
    })
    for ci, cn in enumerate(CLASSES):
        out_df[f"prob_{cn}"] = blend[:, ci]
    out_df.to_parquet(cfg.CACHE_DIR / "track2_preds.parquet", index=False)
    print(f"Updated cache/track2_preds.parquet")
    # Max prob distribution
    mp = blend.max(axis=1)
    print(f"max prob distribution: min={mp.min():.3f} mean={mp.mean():.3f} max={mp.max():.3f}")

    (cfg.CACHE_DIR / "track2_summary.json").write_text(json.dumps({
        "oof_acc": float(acc), "oof_macro_f1": float(f1), "oof_s2": float(s2),
        "weights_no_heur": {"lgb": float(w1), "knn": float(w2), "lr": float(w3), "evgs_only": float(w4)},
    }, indent=2))


if __name__ == "__main__":
    main()
