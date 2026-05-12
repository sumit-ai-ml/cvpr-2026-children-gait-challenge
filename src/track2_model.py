"""Track 2 — Bilateral spastic CP gait pattern classification.

Per-limb 5-class classification: {type1, type2, type3, type4, WNL}.
N = 22 train patients × 2 limbs = 44 examples. Severe few-shot.

Ensemble:
  - LightGBM (class_weight=balanced)
  - kNN (k=3, cosine)
  - L2 multinomial logistic regression
  - Empirical clinical-rule classifier: multinomial LR on EVGS-vec only,
    plus a hard rule "if sum(EVGS) <= 2 -> WNL".

Features per limb = pooled kinematic + 17-dim EVGS vector.
  - For patients in Track1∩Track2 train (17 patients): EVGS = leak-free OOF.
  - For Track2-only patients (5 patients): EVGS = full-trained Track 1 preds.
  - For Track 2 test patients (9 patients): EVGS = full-trained Track 1 preds.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from . import cv as cvmod
from .data_io import load_track1_labels, load_track2_labels


CLASSES = ["type1", "type2", "type3", "type4", "WNL"]
EVGS_ITEMS = [str(i) for i in range(1, 18)]


def build_track2_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (train_df, all_df, feature_cols).

    train_df rows: 22 patients × 2 limbs = 44 (with `y` label column).
    all_df rows: all 110 × 2 = 220 limbs (subset to test by caller).
    feature_cols: pooled features + 17 evgs_* columns.
    """
    pooled = pd.read_parquet(cfg.CACHE_DIR / "features_patient_limb.parquet")
    oof = pd.read_parquet(cfg.CACHE_DIR / "track1_oof_train.parquet")  # 94 train patients
    full = pd.read_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet")  # all 110 patients

    # Build EVGS feature table per (patient_id, side). Prefer OOF (leak-free) over full-trained.
    oof_lookup = {(int(r["patient_id"]), r["side"]): {f"evgs_{it}": float(r[f"oof_{it}"]) for it in EVGS_ITEMS}
                  for _, r in oof.iterrows()}
    full_lookup = {(int(r["patient_id"]), r["side"]): {f"evgs_{it}": float(r[f"prob_{it}"]) for it in EVGS_ITEMS}
                   for _, r in full.iterrows()}

    def evgs_for(pid: int, side: str, in_track2_train: bool) -> dict:
        """For Track 2 train patients: OOF if available (i.e. patient was in Track 1 train), else full-trained.
        For Track 2 test/inference: always full-trained.
        """
        if in_track2_train and (pid, side) in oof_lookup:
            return oof_lookup[(pid, side)]
        return full_lookup[(pid, side)]

    t2_labels = load_track2_labels()  # {pid: {'left': {'gait_subtype': ...}, 'right': ...}}
    side_key = {"L": "left", "R": "right"}

    pooled_indexed = pooled.set_index(["patient_id", "side"])
    train_rows: list[dict] = []
    all_rows: list[dict] = []

    for (pid, side), pooled_row in pooled_indexed.iterrows():
        rec_base = pooled_row.to_dict()
        rec_base["patient_id"] = int(pid)
        rec_base["side"] = side

        # Track 2 train: attach OOF EVGS + label
        if int(pid) in t2_labels:
            rec = dict(rec_base)
            rec.update(evgs_for(int(pid), side, in_track2_train=True))
            rec["y"] = t2_labels[int(pid)][side_key[side]]["gait_subtype"]
            train_rows.append(rec)

        # For inference: every (patient, side) gets full-trained EVGS.
        rec_all = dict(rec_base)
        rec_all.update(evgs_for(int(pid), side, in_track2_train=False))
        all_rows.append(rec_all)

    train_df = pd.DataFrame(train_rows)
    all_df = pd.DataFrame(all_rows)
    evgs_cols = [f"evgs_{it}" for it in EVGS_ITEMS]
    feature_cols = [c for c in all_df.columns if c not in ("patient_id", "side", "y")]
    # Ensure evgs cols at the end of feature list for readability — order doesn't matter for trees.
    feature_cols = [c for c in feature_cols if c not in evgs_cols] + evgs_cols
    return train_df, all_df, feature_cols


# ---- individual classifiers -----------------------------------------------

def _fit_lgb_multi(Xtr, ytr_ord, n_class: int) -> lgb.Booster:
    """Multiclass LightGBM with balanced class weights."""
    counts = Counter(ytr_ord.tolist())
    total = sum(counts.values())
    weights = {c: total / (n_class * max(counts.get(c, 1), 1)) for c in range(n_class)}
    w = np.array([weights[int(y)] for y in ytr_ord])
    params = dict(
        objective="multiclass",
        num_class=n_class,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=2,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=3,
        lambda_l2=1.0,
        verbose=-1,
        random_state=cfg.CFG.seed,
    )
    dtr = lgb.Dataset(Xtr, label=ytr_ord, weight=w)
    return lgb.train(params, dtr, num_boost_round=400)


def _fit_knn(Xtr, ytr_ord, k: int = 3) -> KNeighborsClassifier:
    # Cosine on scaled features = decent metric for limb-similarity.
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    knn.fit(Xtr, ytr_ord)
    return knn


def _fit_lr(Xtr, ytr_ord) -> LogisticRegression:
    lr = LogisticRegression(
        penalty="l2", C=0.5, multi_class="multinomial",
        solver="lbfgs", max_iter=2000, class_weight="balanced",
        random_state=cfg.CFG.seed,
    )
    lr.fit(Xtr, ytr_ord)
    return lr


def _fit_evgs_only(Xtr_evgs, ytr_ord) -> LogisticRegression:
    """The 'clinical' classifier: multinomial LR on the 17-dim EVGS vector only."""
    lr = LogisticRegression(
        penalty="l2", C=1.0, multi_class="multinomial",
        solver="lbfgs", max_iter=2000, class_weight="balanced",
        random_state=cfg.CFG.seed,
    )
    lr.fit(Xtr_evgs, ytr_ord)
    return lr


def _heuristic_wnl(X_evgs: np.ndarray, wnl_idx: int, threshold: float = 2.0) -> np.ndarray:
    """Return a (N, K) probability matrix that boosts WNL when sum(EVGS) is low.

    Mechanics: a softmax-ish mask. If sum(evgs probs) < threshold, output is WNL with high prob;
    otherwise uniform over the other classes."""
    n = X_evgs.shape[0]
    k = 5
    out = np.full((n, k), 1.0 / k)
    sums = X_evgs.sum(axis=1)
    is_wnl = sums < threshold
    out[is_wnl] = 0.05  # tiny prob for other classes
    out[is_wnl, wnl_idx] = 0.80
    return out


# ---- training loop --------------------------------------------------------

@dataclass
class T2Result:
    cv_acc: float
    cv_f1_macro: float
    cv_s2: float
    per_class_f1: dict[str, float]


def train_and_predict() -> dict:
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_df, all_df, feature_cols = build_track2_dataset()
    print(f"Track 2 train: {len(train_df)} limbs ({train_df['patient_id'].nunique()} patients × 2)")
    print(f"Feature dim:  {len(feature_cols)}")

    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    wnl_idx = label_to_idx["WNL"]

    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    evgs_cols = [f"evgs_{it}" for it in EVGS_ITEMS]
    X_evgs = train_df[evgs_cols].to_numpy(dtype=np.float32)
    y = np.array([label_to_idx[s] for s in train_df["y"].values], dtype=np.int64)
    pids = train_df["patient_id"].to_numpy()

    print(f"Class distribution: {dict(Counter([idx_to_label[int(yy)] for yy in y]))}")

    # Leave-one-patient-out CV (each fold removes 2 limbs from one patient)
    folds = list(cvmod.leave_one_patient_out(pids))
    n_folds = len(folds)
    print(f"LOPO folds: {n_folds}")

    # OOF probabilities per model
    oof_lgb = np.zeros((len(y), 5))
    oof_knn = np.zeros((len(y), 5))
    oof_lr = np.zeros((len(y), 5))
    oof_evgs = np.zeros((len(y), 5))

    scaler = StandardScaler()

    for fi, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr = y[tr_idx]
        # Scale for kNN/LR (trees don't need it)
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr)
        Xva_s = sc.transform(Xva)

        m_lgb = _fit_lgb_multi(Xtr, ytr, n_class=5)
        oof_lgb[va_idx] = m_lgb.predict(Xva)

        m_knn = _fit_knn(Xtr_s, ytr, k=3)
        knn_probs = np.zeros((Xva_s.shape[0], 5))
        for i, cls in enumerate(m_knn.classes_):
            knn_probs[:, int(cls)] = m_knn.predict_proba(Xva_s)[:, i]
        oof_knn[va_idx] = knn_probs

        m_lr = _fit_lr(Xtr_s, ytr)
        lr_probs = np.zeros((Xva_s.shape[0], 5))
        for i, cls in enumerate(m_lr.classes_):
            lr_probs[:, int(cls)] = m_lr.predict_proba(Xva_s)[:, i]
        oof_lr[va_idx] = lr_probs

        m_evgs = _fit_evgs_only(X_evgs[tr_idx], ytr)
        evgs_probs = np.zeros((Xva_s.shape[0], 5))
        for i, cls in enumerate(m_evgs.classes_):
            evgs_probs[:, int(cls)] = m_evgs.predict_proba(X_evgs[va_idx])[:, i]
        oof_evgs[va_idx] = evgs_probs

    # Hard heuristic on the held-out evgs vectors
    oof_heur = _heuristic_wnl(X_evgs, wnl_idx)

    # Try various ensemble weights to maximize S_2 on OOF
    def score(probs: np.ndarray) -> tuple[float, float, float, dict[str, float]]:
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y, preds)
        # macro-F1 across all 5 classes
        labels_idx = list(range(5))
        f1 = f1_score(y, preds, labels=labels_idx, average="macro", zero_division=0)
        per_class = {idx_to_label[i]: f1_score(y, preds, labels=[i], average="macro", zero_division=0)
                     for i in labels_idx}
        s2 = (acc + f1) / 2
        return acc, f1, s2, per_class

    print()
    print("Individual model OOF scores:")
    for name, probs in (("lgb", oof_lgb), ("knn", oof_knn), ("lr", oof_lr), ("evgs_only", oof_evgs), ("heuristic", oof_heur)):
        acc, f1, s2, pc = score(probs)
        print(f"  {name:<12} acc={acc:.3f}  f1_macro={f1:.3f}  S_2={s2:.3f}  per-class={pc}")

    # Coarse grid over 5 ensemble weights (sum=1)
    best_s2 = -1.0
    best_w = (0.25, 0.20, 0.20, 0.20, 0.15)
    best_breakdown = None
    grid = np.arange(0.0, 1.01, 0.1)
    print()
    print("Searching ensemble weights ...")
    for w_lgb in grid:
        for w_knn in grid:
            if w_lgb + w_knn > 1.0:
                continue
            for w_lr in grid:
                if w_lgb + w_knn + w_lr > 1.0:
                    continue
                for w_evgs in grid:
                    if w_lgb + w_knn + w_lr + w_evgs > 1.0:
                        continue
                    w_heur = 1.0 - (w_lgb + w_knn + w_lr + w_evgs)
                    if w_heur < 0 or w_heur > 1.0:
                        continue
                    blend = (w_lgb * oof_lgb + w_knn * oof_knn + w_lr * oof_lr
                             + w_evgs * oof_evgs + w_heur * oof_heur)
                    acc, f1, s2, pc = score(blend)
                    if s2 > best_s2:
                        best_s2 = s2
                        best_w = (float(w_lgb), float(w_knn), float(w_lr), float(w_evgs), float(w_heur))
                        best_breakdown = (acc, f1, s2, pc)
    acc, f1, s2, pc = best_breakdown
    print()
    print(f"BEST ensemble weights (lgb, knn, lr, evgs, heur) = {best_w}")
    print(f"  OOF acc:       {acc:.4f}")
    print(f"  OOF macro-F1:  {f1:.4f}")
    print(f"  OOF S_2:       {s2:.4f}")
    print(f"  per-class F1:  {pc}")

    # Refit each model on all 44 training limbs.
    sc_full = StandardScaler().fit(X)
    X_full_s = sc_full.transform(X)
    m_lgb_full = _fit_lgb_multi(X, y, n_class=5)
    m_knn_full = _fit_knn(X_full_s, y, k=3)
    m_lr_full = _fit_lr(X_full_s, y)
    m_evgs_full = _fit_evgs_only(X_evgs, y)

    # Predict for all 110 × 2 = 220 limbs (we filter to test set at submission time).
    X_all = all_df[feature_cols].to_numpy(dtype=np.float32)
    X_all_s = sc_full.transform(X_all)
    X_all_evgs = all_df[evgs_cols].to_numpy(dtype=np.float32)

    def predict_probs(model, X_in, n_cls=5):
        out = np.zeros((X_in.shape[0], n_cls))
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

    w_lgb, w_knn, w_lr, w_evgs, w_heur = best_w
    blend = (w_lgb * p_lgb + w_knn * p_knn + w_lr * p_lr + w_evgs * p_evgs + w_heur * p_heur)
    preds_idx = blend.argmax(axis=1)
    preds_label = [idx_to_label[int(i)] for i in preds_idx]

    out_df = pd.DataFrame({
        "patient_id": all_df["patient_id"].values,
        "side": all_df["side"].values,
        "subtype": preds_label,
    })
    for ci, cn in enumerate(CLASSES):
        out_df[f"prob_{cn}"] = blend[:, ci]

    out_path = cfg.CACHE_DIR / "track2_preds.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    summary = {
        "oof_acc": float(acc),
        "oof_macro_f1": float(f1),
        "oof_s2": float(s2),
        "per_class_f1": pc,
        "ensemble_weights": dict(zip(["lgb", "knn", "lr", "evgs_only", "heuristic"], best_w)),
    }
    (cfg.CACHE_DIR / "track2_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
