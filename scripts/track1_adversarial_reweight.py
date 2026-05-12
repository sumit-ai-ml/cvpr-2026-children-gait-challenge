"""Adversarial validation + sample reweighting for Track 1.

Step 1: Train a binary classifier (LightGBM) to distinguish train (94) vs test (16) patients
        on the pooled kinematic features. Use 5-fold CV.
Step 2: For each train patient, get OOF p(test). Convert to sample weight = p(test) / (1 - p(test)).
        (Importance-weighted likelihood ratio.) Clip to [0.1, 10].
Step 3: Retrain Track 1 ensemble with these sample weights.
Step 4: Gate on OOF S1 improvement vs current baseline.

If the AUC of the adversarial classifier is near 0.5, train and test ARE from same distribution
and reweighting won't help. If AUC is high (>0.7), there's a real shift to fix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

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
    compute_s1,
    tune_thresholds_for_s1,
)


def main() -> None:
    pooled = pd.read_parquet(cfg.CACHE_DIR / "features_patient_limb.parquet")
    feature_cols = [c for c in pooled.columns if c not in ("patient_id", "side")]
    labels = load_track1_labels()
    train_pids = set(labels.keys())
    test_pids = set(cfg.TRACK1_TEST_IDS)

    # ---- Step 1: Adversarial classifier ----
    # One row per (patient, side). Label = 1 if test, 0 if train.
    pooled["is_test"] = pooled.patient_id.isin(test_pids).astype(int)
    X_adv = pooled[feature_cols].to_numpy(dtype=np.float32)
    y_adv = pooled["is_test"].to_numpy()
    pids_adv = pooled["patient_id"].to_numpy()

    n_test_limbs = int(y_adv.sum())
    print(f"Adversarial classifier: train_limbs={(y_adv == 0).sum()}, test_limbs={n_test_limbs}")
    print(f"Class imbalance: {n_test_limbs / len(y_adv):.3f} test")

    # 5-fold patient-grouped CV
    oof_adv = np.zeros(len(y_adv))
    for tr_idx, va_idx in cvmod.patient_kfold(pids_adv, n_splits=5, seed=cfg.CFG.seed):
        spw = (y_adv[tr_idx] == 0).sum() / max((y_adv[tr_idx] == 1).sum(), 1)
        dtr = lgb.Dataset(X_adv[tr_idx], label=y_adv[tr_idx])
        dva = lgb.Dataset(X_adv[va_idx], label=y_adv[va_idx], reference=dtr)
        params = dict(
            objective="binary", metric="auc",
            learning_rate=0.05, num_leaves=15, max_depth=4,
            min_child_samples=5, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=3,
            lambda_l2=1.0, scale_pos_weight=spw, verbose=-1, random_state=cfg.CFG.seed,
        )
        booster = lgb.train(params, dtr, num_boost_round=400,
                            valid_sets=[dva],
                            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        oof_adv[va_idx] = booster.predict(X_adv[va_idx], num_iteration=booster.best_iteration)

    auc = roc_auc_score(y_adv, oof_adv)
    print(f"Adversarial OOF AUC: {auc:.4f}")
    if auc < 0.6:
        print(f"AUC < 0.6 — train and test are NOT meaningfully separable. Reweighting will not help. Discarding.")
        return
    print(f"AUC >= 0.6 — there IS a train/test distribution shift. Continuing.")

    # ---- Step 2: Compute sample weights for train rows ----
    train_mask = (y_adv == 0)
    p_test = oof_adv[train_mask].clip(1e-3, 1 - 1e-3)
    # Importance weight: p(test) / p(train) = p / (1-p)
    weights = p_test / (1 - p_test)
    weights = np.clip(weights, 0.1, 10.0)
    print(f"Sample weights: min={weights.min():.3f} mean={weights.mean():.3f} max={weights.max():.3f}")

    # ---- Step 3: Build training set, attach labels, retrain Track 1 with weights ----
    train_df = pooled[train_mask].reset_index(drop=True).copy()
    train_df["sample_weight"] = weights
    side_key = {"L": "left", "R": "right"}
    for it in EVGS_ITEMS:
        train_df[f"y_{it}"] = train_df.apply(
            lambda r: int(labels[int(r.patient_id)][side_key[r.side]][it]), axis=1
        )
    train_df["y_total"] = train_df.apply(
        lambda r: int(labels[int(r.patient_id)][side_key[r.side]]["Total"]), axis=1
    )

    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    pids = train_df["patient_id"].to_numpy()
    w = train_df["sample_weight"].to_numpy(dtype=np.float32)

    def _fit_lgb_w(Xtr, ytr, Xva, yva, spw, wtr):
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        dva = lgb.Dataset(Xva, label=yva, reference=dtr)
        params = dict(
            objective="binary", metric="binary_logloss",
            learning_rate=0.05, num_leaves=15, max_depth=4,
            min_child_samples=5, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=3,
            lambda_l2=1.0, scale_pos_weight=spw, verbose=-1, random_state=cfg.CFG.seed,
        )
        return lgb.train(params, dtr, num_boost_round=400,
                          valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])

    def _fit_xgb_w(Xtr, ytr, Xva, yva, spw, wtr):
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=500, early_stopping_rounds=30, objective="binary:logistic",
            eval_metric="logloss", learning_rate=0.05, max_depth=4, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
            scale_pos_weight=spw, verbosity=0, random_state=cfg.CFG.seed, tree_method="hist",
        )
        model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
        return model

    def _fit_cb_w(Xtr, ytr, Xva, yva, spw, wtr):
        import catboost as cb
        model = cb.CatBoostClassifier(
            iterations=500, loss_function="Logloss", learning_rate=0.05, depth=4,
            l2_leaf_reg=3.0, subsample=0.8, bootstrap_type="Bernoulli",
            scale_pos_weight=spw, verbose=False, random_seed=cfg.CFG.seed,
            allow_writing_files=False,
        )
        model.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xva, yva), early_stopping_rounds=30, verbose=False)
        return model

    oof_probs_dict: dict[str, np.ndarray] = {}
    y_true_dict: dict[str, np.ndarray] = {}
    for it in EVGS_ITEMS:
        y_it = train_df[f"y_{it}"].to_numpy(dtype=np.int32)
        pos = max(int(y_it.sum()), 1); neg = max(int((1 - y_it).sum()), 1)
        spw = neg / pos
        oof_lgb = np.zeros(len(y_it)); oof_xgb = np.zeros(len(y_it)); oof_cb = np.zeros(len(y_it))
        for tr_idx, va_idx in cvmod.patient_kfold(pids, n_splits=5, seed=cfg.CFG.seed):
            m1 = _fit_lgb_w(X[tr_idx], y_it[tr_idx], X[va_idx], y_it[va_idx], spw, w[tr_idx])
            oof_lgb[va_idx] = m1.predict(X[va_idx], num_iteration=m1.best_iteration)
            m2 = _fit_xgb_w(X[tr_idx], y_it[tr_idx], X[va_idx], y_it[va_idx], spw, w[tr_idx])
            oof_xgb[va_idx] = m2.predict_proba(X[va_idx])[:, 1]
            m3 = _fit_cb_w(X[tr_idx], y_it[tr_idx], X[va_idx], y_it[va_idx], spw, w[tr_idx])
            oof_cb[va_idx] = m3.predict_proba(X[va_idx])[:, 1]
        oof_probs_dict[it] = (oof_lgb + oof_xgb + oof_cb) / 3.0
        y_true_dict[it] = y_it
        print(f"  item {it}: base_rate={y_it.mean():.2f}  OOF@0.5={float(((oof_probs_dict[it]>=0.5).astype(int)==y_it).mean()):.3f}")

    y_total = train_df["y_total"].to_numpy(dtype=np.float32)
    thrs, best_s1 = tune_thresholds_for_s1(oof_probs_dict, y_true_dict, y_total, pids, n_iters=5)
    acc, nrmse, s1 = compute_s1(oof_probs_dict, y_true_dict, y_total, pids, thrs)
    print(f"\n== TRACK 1 (adversarial-reweighted) OOF SCORE ==")
    print(f"  Acc:   {acc:.4f}")
    print(f"  NRMSE: {nrmse:.4f}")
    print(f"  S_1:   {s1:.4f}")

    baseline_s1 = 0.8263
    if s1 <= baseline_s1:
        print(f"\nAdv-reweight did NOT improve S_1 (baseline {baseline_s1}, got {s1:.4f}). Discarding.")
        return

    print(f"\nADV-REWEIGHT IMPROVED S_1: {baseline_s1:.4f} -> {s1:.4f}  (Δ=+{s1-baseline_s1:.4f})")
    # Refit on full train with weights, predict on all 220 rows.
    rng = np.random.default_rng(cfg.CFG.seed)
    n = len(y_total)
    perm = rng.permutation(n)
    split = max(int(n * 0.1), 1)
    va_i = perm[:split]; tr_i = perm[split:]

    full = pd.DataFrame({"patient_id": pooled.patient_id.values, "side": pooled.side.values})
    X_all = pooled[feature_cols].to_numpy(dtype=np.float32)
    for it in EVGS_ITEMS:
        y_it = train_df[f"y_{it}"].to_numpy(dtype=np.int32)
        pos = max(int(y_it.sum()), 1); neg = max(int((1 - y_it).sum()), 1)
        spw = neg / pos
        m1 = _fit_lgb_w(X[tr_i], y_it[tr_i], X[va_i], y_it[va_i], spw, w[tr_i])
        m2 = _fit_xgb_w(X[tr_i], y_it[tr_i], X[va_i], y_it[va_i], spw, w[tr_i])
        m3 = _fit_cb_w(X[tr_i], y_it[tr_i], X[va_i], y_it[va_i], spw, w[tr_i])
        p1 = m1.predict(X_all)
        p2 = m2.predict_proba(X_all)[:, 1]
        p3 = m3.predict_proba(X_all)[:, 1]
        prob = (p1 + p2 + p3) / 3.0
        full[f"prob_{it}"] = prob
        full[f"pred_{it}"] = (prob >= thrs[it]).astype(int)
    full["pred_total_sum"] = full[[f"pred_{it}" for it in EVGS_ITEMS]].sum(axis=1)
    full["pred_total"] = full["pred_total_sum"]
    full.to_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet", index=False)
    print("Updated cache/track1_full_preds.parquet")

    (cfg.CACHE_DIR / "track1_adv_summary.json").write_text(json.dumps({
        "adversarial_auc": float(auc), "baseline_s1": baseline_s1,
        "new_s1": float(s1), "delta": float(s1 - baseline_s1),
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
