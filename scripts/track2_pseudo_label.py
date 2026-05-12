"""Pseudo-label the 79 patients without Track 2 labels using the current Track 2 ensemble,
filter to high-confidence predictions, retrain. Gated on OOF S₂ improvement.

Approach:
1. Predict subtypes for ALL 110 patients using current Track 2 ensemble (already done).
2. Filter to: NOT in Track 2 train, NOT in Track 2 test, AND max(prob) >= conf_thr.
3. Add those (patient, side) rows with their predicted label to training.
4. Retrain Track 2 with original 44 limbs + pseudo-labeled extras.
5. LOPO over ORIGINAL 22 patients only — pseudo-labels never go into val.
6. Compare to baseline S₂.
"""
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
from src.data_io import load_track2_labels
from src.track2_model import (
    CLASSES,
    EVGS_ITEMS,
    _fit_evgs_only,
    _fit_knn,
    _fit_lgb_multi,
    _fit_lr,
    _heuristic_wnl,
    build_track2_dataset,
)


CONF_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.60]


def main() -> None:
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    wnl_idx = label_to_idx["WNL"]

    # 1. Load current predictions on all 220 limbs.
    cur = pd.read_parquet(cfg.CACHE_DIR / "track2_preds.parquet")
    # 2. Load existing training set.
    train_df, all_df, feature_cols = build_track2_dataset()
    train_pids = set(train_df.patient_id.unique().tolist())
    test_pids = set(cfg.TRACK2_TEST_IDS)

    # 3. Identify unlabeled (eligible for pseudo-labels) patients.
    eligible = [p for p in cur.patient_id.unique() if int(p) not in train_pids and int(p) not in test_pids]
    print(f"Track 2 train patients: {len(train_pids)}")
    print(f"Track 2 test patients:  {len(test_pids)}")
    print(f"Eligible for pseudo-label: {len(eligible)}")

    # 4. Compute per-limb max probability for each eligible (patient, side).
    cur["max_prob"] = cur[[f"prob_{c}" for c in CLASSES]].max(axis=1)

    baseline_s2 = 0.3089
    results = []
    best = (None, baseline_s2)

    for conf_thr in CONF_THRESHOLDS:
        keep = cur[(cur.patient_id.isin(eligible)) & (cur.max_prob >= conf_thr)]
        print(f"\nconf_thr={conf_thr}: {len(keep)} pseudo-labeled limbs ({keep.patient_id.nunique()} patients)")
        if len(keep) == 0:
            continue
        print(f"  pseudo-label distribution: {dict(Counter(keep.subtype))}")

        # Build augmented training pool.
        all_features = all_df.set_index(["patient_id", "side"])
        pseudo_rows = []
        for _, r in keep.iterrows():
            key = (int(r.patient_id), r.side)
            if key not in all_features.index:
                continue
            base = all_features.loc[key].to_dict()
            base["patient_id"] = int(r.patient_id)
            base["side"] = r.side
            base["y"] = r.subtype
            pseudo_rows.append(base)
        pseudo_df = pd.DataFrame(pseudo_rows)
        augmented = pd.concat([train_df, pseudo_df], ignore_index=True)
        print(f"  augmented training size: {len(augmented)} limbs")

        # LOPO over ORIGINAL 22 train patients only.
        X = augmented[feature_cols].to_numpy(dtype=np.float32)
        evgs_cols = [f"evgs_{it}" for it in EVGS_ITEMS]
        X_evgs = augmented[evgs_cols].to_numpy(dtype=np.float32)
        y = np.array([label_to_idx[s] for s in augmented["y"].values], dtype=np.int64)
        pids = augmented["patient_id"].to_numpy()
        # Only patients in train_pids go to val. Pseudo-labeled patients always train.
        original_mask = np.array([int(p) in train_pids for p in pids])

        oof_lgb = np.zeros((len(y), 5))
        oof_knn = np.zeros((len(y), 5))
        oof_lr = np.zeros((len(y), 5))
        oof_evgs = np.zeros((len(y), 5))

        original_pids = sorted(train_pids)
        for held_pid in original_pids:
            # Train: all rows EXCEPT held-out patient's rows (pseudo-labeled stays in train)
            tr_idx = np.where(pids != held_pid)[0]
            va_idx = np.where((pids == held_pid) & original_mask)[0]
            if len(va_idx) == 0:
                continue
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

        # Filter to ORIGINAL patients for scoring (no pseudo-labels in eval).
        eval_mask = original_mask
        y_eval = y[eval_mask]

        def score_blend(weights):
            probs = (weights[0] * oof_lgb + weights[1] * oof_knn + weights[2] * oof_lr +
                     weights[3] * oof_evgs + weights[4] * oof_heur)
            preds = probs.argmax(axis=1)
            preds_e = preds[eval_mask]
            acc = accuracy_score(y_eval, preds_e)
            f1 = f1_score(y_eval, preds_e, labels=list(range(5)), average="macro", zero_division=0)
            return acc, f1, (acc + f1) / 2

        grid = np.arange(0.0, 1.01, 0.1)
        best_local = (-1.0, None)
        for w1 in grid:
            for w2 in grid:
                if w1 + w2 > 1: continue
                for w3 in grid:
                    if w1 + w2 + w3 > 1: continue
                    for w4 in grid:
                        if w1 + w2 + w3 + w4 > 1: continue
                        w5 = 1 - (w1 + w2 + w3 + w4)
                        if w5 < 0 or w5 > 1: continue
                        acc, f1, s2 = score_blend((w1, w2, w3, w4, w5))
                        if s2 > best_local[0]:
                            best_local = (s2, (w1, w2, w3, w4, w5, acc, f1))
        s2, packed = best_local
        w1, w2, w3, w4, w5, acc, f1 = packed
        delta = s2 - baseline_s2
        sign = "+" if delta >= 0 else ""
        print(f"  best OOF: Acc={acc:.4f}  F1={f1:.4f}  S_2={s2:.4f}  (Δ={sign}{delta:.4f})")
        results.append({"conf_thr": conf_thr, "n_pseudo": len(keep), "s2": s2, "acc": acc, "f1": f1,
                         "weights": (w1, w2, w3, w4, w5)})
        if s2 > best[1]:
            best = ({"conf_thr": conf_thr, "weights": (w1, w2, w3, w4, w5), "s2": s2,
                     "augmented_size": len(augmented)}, s2)

    print()
    if best[0] is None:
        print(f"Pseudo-labeling did NOT improve S_2 (baseline {baseline_s2:.4f}). Discarding.")
    else:
        b = best[0]
        print(f"PSEUDO-LABELING IMPROVED S_2: conf_thr={b['conf_thr']}, S_2={b['s2']:.4f}, augmented={b['augmented_size']}")
        print(f"  weights (lgb,knn,lr,evgs,heur)={b['weights']}")

    (cfg.CACHE_DIR / "track2_pseudo_summary.json").write_text(json.dumps({
        "baseline_s2": baseline_s2,
        "best": best[0],
        "all_results": results,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
