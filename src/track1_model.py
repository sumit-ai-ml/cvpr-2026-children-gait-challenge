"""Track 1 — EVGS scoring with 3-tree ensemble + S₁-joint threshold tuning.

Models per item: LightGBM + XGBoost + CatBoost. OOF probabilities averaged.
Thresholds tuned jointly via coordinate descent to maximize the official
S_1 = (Acc + 1 - NRMSE) / 2 metric.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

from . import config as cfg
from . import cv as cvmod
from .data_io import load_track1_labels


EVGS_ITEMS = [str(i) for i in range(1, 18)]


def build_track1_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    pooled = pd.read_parquet(cfg.CACHE_DIR / "features_patient_limb.parquet")
    labels = load_track1_labels()
    side_key = {"L": "left", "R": "right"}
    train_rows: list[dict] = []
    for _, row in pooled.iterrows():
        pid = int(row["patient_id"])
        if pid not in labels:
            continue
        lab = labels[pid][side_key[row["side"]]]
        rec = row.to_dict()
        for it in EVGS_ITEMS:
            rec[f"y_{it}"] = int(lab[it])
        rec["y_total"] = int(lab["Total"])
        train_rows.append(rec)
    train_df = pd.DataFrame(train_rows)
    feature_cols = [c for c in pooled.columns if c not in ("patient_id", "side")]
    return train_df, pooled.copy(), feature_cols


# ---- params ----------------------------------------------------------------

def _lgb_params(spw: float) -> dict:
    return dict(
        objective="binary", metric="binary_logloss",
        learning_rate=0.05, num_leaves=15, max_depth=4,
        min_child_samples=5, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=3, lambda_l2=1.0,
        scale_pos_weight=spw, verbose=-1, random_state=cfg.CFG.seed,
    )


def _xgb_params(spw: float) -> dict:
    return dict(
        objective="binary:logistic", eval_metric="logloss",
        learning_rate=0.05, max_depth=4, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        scale_pos_weight=spw, verbosity=0, random_state=cfg.CFG.seed,
        tree_method="hist",
    )


def _cb_params(spw: float) -> dict:
    return dict(
        loss_function="Logloss", learning_rate=0.05, depth=4,
        l2_leaf_reg=3.0, subsample=0.8, bootstrap_type="Bernoulli",
        scale_pos_weight=spw, verbose=False, random_seed=cfg.CFG.seed,
        allow_writing_files=False,
    )


# ---- one-fold trainers ------------------------------------------------------

def _fit_lgb(Xtr, ytr, Xva, yva, spw):
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    return lgb.train(
        _lgb_params(spw), dtr, num_boost_round=500,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )


def _fit_xgb(Xtr, ytr, Xva, yva, spw):
    model = xgb.XGBClassifier(n_estimators=500, early_stopping_rounds=30, **_xgb_params(spw))
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model


def _fit_cb(Xtr, ytr, Xva, yva, spw):
    model = cb.CatBoostClassifier(iterations=500, **_cb_params(spw))
    model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=30, verbose=False)
    return model


# ---- one-item end-to-end ---------------------------------------------------

@dataclass
class ItemResult:
    item: str
    oof_acc_thr: float       # per-item accuracy at the tuned threshold
    base_rate: float
    n_pos: int


def _run_item(X: np.ndarray, y: np.ndarray, pids: np.ndarray) -> tuple[np.ndarray, float]:
    """Train 3-tree ensemble with 5-fold patient-grouped CV. Return averaged OOF probs."""
    pos = max(int(y.sum()), 1)
    neg = max(int((1 - y).sum()), 1)
    spw = neg / pos

    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cb = np.zeros(len(y))
    for tr_idx, va_idx in cvmod.patient_kfold(pids, n_splits=5, seed=cfg.CFG.seed):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        m_lgb = _fit_lgb(Xtr, ytr, Xva, yva, spw)
        oof_lgb[va_idx] = m_lgb.predict(Xva, num_iteration=m_lgb.best_iteration)
        m_xgb = _fit_xgb(Xtr, ytr, Xva, yva, spw)
        oof_xgb[va_idx] = m_xgb.predict_proba(Xva)[:, 1]
        m_cb = _fit_cb(Xtr, ytr, Xva, yva, spw)
        oof_cb[va_idx] = m_cb.predict_proba(Xva)[:, 1]
    oof_avg = (oof_lgb + oof_xgb + oof_cb) / 3.0
    return oof_avg, spw


def _refit_full(X: np.ndarray, y: np.ndarray, spw: float):
    """Refit all three models on full train. Returns the three boosters."""
    # Use 90/10 internal split for early stopping.
    rng = np.random.default_rng(cfg.CFG.seed)
    n = len(y)
    perm = rng.permutation(n)
    split = max(int(n * 0.1), 1)
    va_idx = perm[:split]
    tr_idx = perm[split:]
    m_lgb = _fit_lgb(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], spw)
    m_xgb = _fit_xgb(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], spw)
    m_cb = _fit_cb(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], spw)
    return m_lgb, m_xgb, m_cb


def _predict_ensemble(models, X) -> np.ndarray:
    m_lgb, m_xgb, m_cb = models
    p_lgb = m_lgb.predict(X)
    p_xgb = m_xgb.predict_proba(X)[:, 1]
    p_cb = m_cb.predict_proba(X)[:, 1]
    return (p_lgb + p_xgb + p_cb) / 3.0


# ---- S₁ metric & joint threshold tuning -----------------------------------

def compute_s1(
    oof_probs: dict[str, np.ndarray],
    y_true: dict[str, np.ndarray],
    y_total_per_limb: np.ndarray,
    patient_ids: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[float, float, float]:
    """Return (acc, nrmse, s1) for the given thresholds.

    Item accuracy is pooled across all items × all limbs.
    Total RMSE is per-patient (L_total + R_total).
    """
    correct = 0
    total_n = 0
    per_limb_pred_total = np.zeros(len(y_total_per_limb), dtype=np.float64)
    for it in EVGS_ITEMS:
        pred = (oof_probs[it] >= thresholds[it]).astype(int)
        correct += int((pred == y_true[it]).sum())
        total_n += len(pred)
        per_limb_pred_total += pred
    acc = correct / max(total_n, 1)

    df = pd.DataFrame({"pid": patient_ids, "pred": per_limb_pred_total, "y": y_total_per_limb})
    per_pat = df.groupby("pid").agg(pred=("pred", "sum"), y=("y", "sum"))
    rmse = float(np.sqrt(np.mean((per_pat["pred"] - per_pat["y"]) ** 2)))
    nrmse = rmse / 34.0
    s1 = (acc + 1.0 - nrmse) / 2.0
    return acc, nrmse, s1


def tune_thresholds_for_s1(
    oof_probs: dict[str, np.ndarray],
    y_true: dict[str, np.ndarray],
    y_total_per_limb: np.ndarray,
    patient_ids: np.ndarray,
    n_iters: int = 5,
) -> tuple[dict[str, float], float]:
    """Coordinate descent over per-item thresholds to maximize S_1."""
    thrs: dict[str, float] = {it: 0.5 for it in EVGS_ITEMS}
    grid = np.linspace(0.05, 0.95, 19)
    best_acc, best_nrmse, best_s1 = compute_s1(oof_probs, y_true, y_total_per_limb, patient_ids, thrs)
    for it_pass in range(n_iters):
        improved = False
        for it in EVGS_ITEMS:
            current = thrs[it]
            for t in grid:
                thrs[it] = float(t)
                acc, nrmse, s1 = compute_s1(oof_probs, y_true, y_total_per_limb, patient_ids, thrs)
                if s1 > best_s1 + 1e-6:
                    best_s1, best_acc, best_nrmse = s1, acc, nrmse
                    current = float(t)
                    improved = True
            thrs[it] = current
        if not improved:
            break
    return thrs, best_s1


# ---- main pipeline ---------------------------------------------------------

def train_and_predict() -> dict:
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_df, all_df, feature_cols = build_track1_dataset()
    print(f"Track 1 train rows: {len(train_df)} ({train_df['patient_id'].nunique()} patients × 2 sides)")
    print(f"Feature dim: {len(feature_cols)}")

    X_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
    pids = train_df["patient_id"].to_numpy()
    y_total = train_df["y_total"].to_numpy(dtype=np.float32)

    oof_probs_dict: dict[str, np.ndarray] = {}
    y_true_dict: dict[str, np.ndarray] = {}
    item_models: dict[str, tuple] = {}
    summaries: list[ItemResult] = []

    for it in EVGS_ITEMS:
        y = train_df[f"y_{it}"].to_numpy(dtype=np.int32)
        oof, spw = _run_item(X_tr, y, pids)
        oof_probs_dict[it] = oof
        y_true_dict[it] = y
        # Quick per-item accuracy at 0.5 to track imbalance handling.
        per_item_acc = accuracy_score(y, (oof >= 0.5).astype(int))
        summaries.append(ItemResult(item=it, oof_acc_thr=per_item_acc, base_rate=float(y.mean()), n_pos=int(y.sum())))
        # Refit on full train.
        item_models[it] = _refit_full(X_tr, y, spw)
        print(f"  item {it}: base_rate={y.mean():.2f}  OOF@0.5 acc={per_item_acc:.3f}")

    # Joint threshold tuning to max S_1
    print()
    print("Tuning thresholds jointly for S_1 ...")
    thrs, best_s1 = tune_thresholds_for_s1(oof_probs_dict, y_true_dict, y_total, pids, n_iters=5)
    acc, nrmse, s1 = compute_s1(oof_probs_dict, y_true_dict, y_total, pids, thrs)
    print(f"  OOF Acc:   {acc:.4f}")
    print(f"  OOF NRMSE: {nrmse:.4f}  (Total RMSE per patient: {nrmse*34:.3f})")
    print(f"  OOF S_1:   {s1:.4f}")

    # Persist OOF table (probabilities, not thresholded).
    oof_df = pd.DataFrame({"patient_id": pids, "side": train_df["side"].values})
    for it in EVGS_ITEMS:
        oof_df[f"oof_{it}"] = oof_probs_dict[it]
        oof_df[f"y_{it}"] = y_true_dict[it]
    oof_df["y_total"] = y_total
    oof_df.to_parquet(cfg.CACHE_DIR / "track1_oof_train.parquet", index=False)

    # Predict on all 110 patients × 2 sides.
    X_all = all_df[feature_cols].to_numpy(dtype=np.float32)
    full = pd.DataFrame({"patient_id": all_df["patient_id"].values, "side": all_df["side"].values})
    for it in EVGS_ITEMS:
        prob = _predict_ensemble(item_models[it], X_all)
        full[f"prob_{it}"] = prob
        full[f"pred_{it}"] = (prob >= thrs[it]).astype(int)
    full["pred_total_sum"] = full[[f"pred_{it}" for it in EVGS_ITEMS]].sum(axis=1)
    full["pred_total"] = full["pred_total_sum"]
    full.to_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet", index=False)

    with (cfg.CACHE_DIR / "track1_models.pkl").open("wb") as f:
        pickle.dump({"item_models": item_models, "thresholds": thrs, "feature_cols": feature_cols}, f)

    return {
        "oof_acc": acc,
        "oof_nrmse": nrmse,
        "oof_s1": s1,
        "thresholds": thrs,
        "summaries": [s.__dict__ for s in summaries],
    }
