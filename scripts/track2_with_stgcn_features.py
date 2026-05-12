"""Experiment: add ST-GCN OOF predictions as additional Track 2 features.

ST-GCN didn't improve Track 1, but its 17-dim probability vector per limb might
separate subtype classes differently from kinematic features. Cheap to try since
the ST-GCN OOF artifact already exists.

Gated: accept only if OOF S₂ improves over current 0.309.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import cv as cvmod
from src.data_io import load_track2_labels
from src.track2_model import CLASSES, EVGS_ITEMS, _fit_evgs_only, _fit_knn, _fit_lgb_multi, _fit_lr, _heuristic_wnl


def main() -> None:
    pooled = pd.read_parquet(cfg.CACHE_DIR / "features_patient_limb.parquet")
    tree_oof = pd.read_parquet(cfg.CACHE_DIR / "track1_oof_train.parquet")
    tree_full = pd.read_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet")
    stgcn_oof = pd.read_parquet(cfg.CACHE_DIR / "track1_stgcn_oof.parquet")
    stgcn_full = pd.read_parquet(cfg.CACHE_DIR / "track1_stgcn_full.parquet")

    side_key = {"L": "left", "R": "right"}
    t2_labels = load_track2_labels()

    # Build (patient, side) → tree+stgcn evgs vectors
    tree_oof_lookup = {(int(r["patient_id"]), r["side"]): {f"evgs_{it}": float(r[f"oof_{it}"]) for it in EVGS_ITEMS}
                        for _, r in tree_oof.iterrows()}
    tree_full_lookup = {(int(r["patient_id"]), r["side"]): {f"evgs_{it}": float(r[f"prob_{it}"]) for it in EVGS_ITEMS}
                        for _, r in tree_full.iterrows()}
    stgcn_oof_lookup = {(int(r["patient_id"]), r["side"]): {f"stgcn_{it}": float(r[f"oof_{it}"]) for it in EVGS_ITEMS}
                        for _, r in stgcn_oof.iterrows()}
    stgcn_full_lookup = {(int(r["patient_id"]), r["side"]): {f"stgcn_{it}": float(r[f"prob_{it}"]) for it in EVGS_ITEMS}
                         for _, r in stgcn_full.iterrows()}

    def evgs_for(pid: int, side: str, in_train: bool):
        if in_train and (pid, side) in tree_oof_lookup:
            tree = tree_oof_lookup[(pid, side)]
        else:
            tree = tree_full_lookup[(pid, side)]
        if in_train and (pid, side) in stgcn_oof_lookup:
            sg = stgcn_oof_lookup[(pid, side)]
        else:
            sg = stgcn_full_lookup[(pid, side)]
        merged = dict(tree)
        merged.update(sg)
        return merged

    pooled_indexed = pooled.set_index(["patient_id", "side"])
    train_rows: list[dict] = []
    for (pid, side), pooled_row in pooled_indexed.iterrows():
        if int(pid) not in t2_labels:
            continue
        rec = pooled_row.to_dict()
        rec["patient_id"] = int(pid); rec["side"] = side
        rec.update(evgs_for(int(pid), side, in_train=True))
        rec["y"] = t2_labels[int(pid)][side_key[side]]["gait_subtype"]
        train_rows.append(rec)
    train_df = pd.DataFrame(train_rows)

    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    wnl_idx = label_to_idx["WNL"]

    feature_cols = [c for c in train_df.columns if c not in ("patient_id", "side", "y")]
    evgs_cols = [f"evgs_{it}" for it in EVGS_ITEMS]
    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_evgs = train_df[evgs_cols].to_numpy(dtype=np.float32)
    y = np.array([label_to_idx[s] for s in train_df["y"].values], dtype=np.int64)
    pids = train_df["patient_id"].to_numpy()

    print(f"Track 2 augmented: {len(train_df)} limbs, feature_dim={len(feature_cols)} (including 17 stgcn cols)")
    print(f"Class dist: {dict(Counter([idx_to_label[int(yy)] for yy in y]))}")

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
        knn_probs = np.zeros((Xva.shape[0], 5))
        for i, cls in enumerate(m_knn.classes_):
            knn_probs[:, int(cls)] = m_knn.predict_proba(sc.transform(Xva))[:, i]
        oof_knn[va_idx] = knn_probs
        m_lr = _fit_lr(sc.transform(Xtr), ytr)
        lr_probs = np.zeros((Xva.shape[0], 5))
        for i, cls in enumerate(m_lr.classes_):
            lr_probs[:, int(cls)] = m_lr.predict_proba(sc.transform(Xva))[:, i]
        oof_lr[va_idx] = lr_probs
        m_evgs = _fit_evgs_only(X_evgs[tr_idx], ytr)
        evgs_probs = np.zeros((Xva.shape[0], 5))
        for i, cls in enumerate(m_evgs.classes_):
            evgs_probs[:, int(cls)] = m_evgs.predict_proba(X_evgs[va_idx])[:, i]
        oof_evgs[va_idx] = evgs_probs

    oof_heur = _heuristic_wnl(X_evgs, wnl_idx)

    def score(probs):
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y, preds)
        labels_idx = list(range(5))
        f1 = f1_score(y, preds, labels=labels_idx, average="macro", zero_division=0)
        return acc, f1, (acc + f1) / 2

    print()
    print("Individual model OOF (with ST-GCN features added):")
    for name, probs in (("lgb+stgcn", oof_lgb), ("knn+stgcn", oof_knn), ("lr+stgcn", oof_lr), ("evgs_only", oof_evgs), ("heuristic", oof_heur)):
        acc, f1, s2 = score(probs)
        print(f"  {name:<12} acc={acc:.3f}  f1_macro={f1:.3f}  S_2={s2:.3f}")

    grid = np.arange(0.0, 1.01, 0.1)
    best = (-1.0, None)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1: continue
            for w3 in grid:
                if w1 + w2 + w3 > 1: continue
                for w4 in grid:
                    if w1 + w2 + w3 + w4 > 1: continue
                    w5 = 1 - (w1 + w2 + w3 + w4)
                    if w5 < 0 or w5 > 1: continue
                    blend = (w1 * oof_lgb + w2 * oof_knn + w3 * oof_lr + w4 * oof_evgs + w5 * oof_heur)
                    acc, f1, s2 = score(blend)
                    if s2 > best[0]:
                        best = (s2, (w1, w2, w3, w4, w5, acc, f1))
    s2, (w1, w2, w3, w4, w5, acc, f1) = best
    print()
    print(f"BEST weights (lgb, knn, lr, evgs, heur) = ({w1:.1f}, {w2:.1f}, {w3:.1f}, {w4:.1f}, {w5:.1f})")
    print(f"  OOF acc:       {acc:.4f}")
    print(f"  OOF macro-F1:  {f1:.4f}")
    print(f"  OOF S_2:       {s2:.4f}")

    baseline = 0.3089
    delta = s2 - baseline
    print()
    if s2 > baseline:
        print(f"ADDED ST-GCN as Track 2 feature: S_2 {baseline:.4f} -> {s2:.4f}  (Δ={delta:+.4f})")
        # Persist updated track2_preds for inference
        # Re-fit on all train + predict all limbs with the augmented features.
        sc_full = StandardScaler().fit(X)
        m_lgb_full = _fit_lgb_multi(X, y, n_class=5)
        m_knn_full = _fit_knn(sc_full.transform(X), y, k=3)
        m_lr_full = _fit_lr(sc_full.transform(X), y)
        m_evgs_full = _fit_evgs_only(X_evgs, y)

        # Build all-limb feature table.
        all_rows = []
        for (pid, side), pooled_row in pooled_indexed.iterrows():
            rec = pooled_row.to_dict()
            rec["patient_id"] = int(pid); rec["side"] = side
            rec.update(evgs_for(int(pid), side, in_train=False))
            all_rows.append(rec)
        all_df = pd.DataFrame(all_rows)
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
        p_heur = _heuristic_wnl(X_all_evgs, wnl_idx)
        blend = (w1 * p_lgb + w2 * p_knn + w3 * p_lr + w4 * p_evgs + w5 * p_heur)
        preds_idx = blend.argmax(axis=1)
        preds_label = [idx_to_label[int(i)] for i in preds_idx]
        out_df = pd.DataFrame({
            "patient_id": all_df["patient_id"].values,
            "side": all_df["side"].values,
            "subtype": preds_label,
        })
        for ci, cn in enumerate(CLASSES):
            out_df[f"prob_{cn}"] = blend[:, ci]
        out_df.to_parquet(cfg.CACHE_DIR / "track2_preds.parquet", index=False)
        print(f"Updated cache/track2_preds.parquet")

        (cfg.CACHE_DIR / "track2_stgcn_aug_summary.json").write_text(json.dumps({
            "baseline_s2": baseline, "new_s2": s2, "delta": delta,
            "weights": {"lgb": w1, "knn": w2, "lr": w3, "evgs": w4, "heuristic": w5},
        }, indent=2))
    else:
        print(f"Adding ST-GCN features did NOT improve Track 2 (baseline {baseline:.4f}, got {s2:.4f}). Discarding.")


if __name__ == "__main__":
    main()
