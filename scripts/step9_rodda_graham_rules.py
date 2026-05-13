"""Step 9: Rodda-Graham clinical rules for Track 2.

Use continuous kinematic features (ankle/knee/hip flex angles) and EVGS item
predictions to apply the published Rodda-Graham classification rules directly.

Rodda-Graham subtype defining features:
  type1 True Equinus:   ankle plantarflexed in stance, knees/hips extended
  type2 Jump Gait:      ankle equinus + knee/hip flexed early stance, extending late
  type3 Apparent Equinus: normal ankle DF + knee/hip flexed in stance
  type4 Crouch Gait:    excessive ankle DF + extreme knee/hip flexion
  WNL:                  all close to normal (low EVGS Total)

Compares LOPO S_2 to the existing 5-model ensemble (current 0.65).
"""
from __future__ import annotations

import json
import sys
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
from src.track2_model import build_track2_dataset, CLASSES, EVGS_ITEMS


# Curated discriminative features per Rodda-Graham logic
CLINICAL_FEATS = [
    "ipsi_ankle_flex_mean_clip_mean",   # ankle plantarflexion in stance
    "ipsi_ankle_flex_max_clip_mean",    # peak plantarflexion
    "ipsi_ankle_flex_min_clip_mean",    # max DF (lower = more DF)
    "ipsi_knee_flex_mean_clip_mean",    # knee flexion in stance
    "ipsi_knee_flex_min_clip_mean",     # max knee flexion (lower = more flexed)
    "ipsi_hip_flex_mean_clip_mean",     # hip flexion in stance
    "ipsi_hip_flex_min_clip_mean",      # max hip flexion (lower = more flexed)
    "contra_ankle_flex_mean_clip_mean",
    "contra_knee_flex_mean_clip_mean",
    "contra_hip_flex_mean_clip_mean",
    "trunk_lean_mean_clip_mean",        # forward lean (crouch indicator)
    "trunk_lean_std_clip_mean",
    "pelvic_obliquity_std_clip_mean",   # obliquity variation
]


def main():
    train_df, all_df, feature_cols = build_track2_dataset()
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    # Filter curated clinical features
    avail = [c for c in CLINICAL_FEATS if c in train_df.columns]
    print(f"Available clinical features: {len(avail)}/{len(CLINICAL_FEATS)}")
    evgs_cols = [f"evgs_{it}" for it in EVGS_ITEMS]
    feat_cols = avail + evgs_cols
    print(f"Total features (clinical + EVGS): {len(feat_cols)}")

    X = train_df[feat_cols].to_numpy(dtype=np.float32)
    # Add EVGS Total
    evgs_total = train_df[evgs_cols].sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1)
    X = np.hstack([X, evgs_total])
    print(f"X shape: {X.shape}")

    y = np.array([label_to_idx[s] for s in train_df["y"].values], dtype=np.int64)
    pids = train_df["patient_id"].to_numpy()

    # LOPO with multinomial LR on curated features
    oof_lr = np.zeros((len(y), 5))
    for tr_idx, va_idx in cvmod.leave_one_patient_out(pids):
        sc = StandardScaler().fit(X[tr_idx])
        Xtr_s = sc.transform(X[tr_idx])
        Xva_s = sc.transform(X[va_idx])
        m = LogisticRegression(
            penalty="l2", C=0.5, multi_class="multinomial",
            solver="lbfgs", max_iter=3000, class_weight="balanced",
            random_state=cfg.CFG.seed,
        )
        m.fit(Xtr_s, y[tr_idx])
        for i, cls in enumerate(m.classes_):
            oof_lr[va_idx, int(cls)] = m.predict_proba(Xva_s)[:, i]

    def score(probs):
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, labels=list(range(5)), average="macro", zero_division=0)
        s2 = (acc + f1) / 2
        per_class = {
            idx_to_label[i]: f1_score(y, preds, labels=[i], average="macro", zero_division=0)
            for i in range(5)
        }
        return acc, f1, s2, per_class

    acc, f1, s2, pc = score(oof_lr)
    print(f"\nCurated-feature LR LOPO: acc={acc:.3f}  f1_macro={f1:.3f}  S_2={s2:.3f}")
    print(f"  per-class F1: {pc}")
    print(f"\n(Baseline ensemble OOF S_2 ~0.65)")

    # Save OOF for use in ensembling
    out = pd.DataFrame({
        "patient_id": train_df["patient_id"].values,
        "side": train_df["side"].values,
    })
    for i, c in enumerate(CLASSES):
        out[f"prob_{c}"] = oof_lr[:, i]
    out.to_parquet(cfg.CACHE_DIR / "track2_clinical_lr_oof.parquet", index=False)

    # Also: refit on full train and produce test predictions
    sc_full = StandardScaler().fit(X)
    X_all_evgs = all_df[evgs_cols].to_numpy(dtype=np.float32)
    X_all_clinical = all_df[avail].to_numpy(dtype=np.float32)
    X_all = np.hstack([X_all_clinical, X_all_evgs, X_all_evgs.sum(axis=1, keepdims=True)])
    X_all_s = sc_full.transform(X_all)
    m_full = LogisticRegression(
        penalty="l2", C=0.5, multi_class="multinomial",
        solver="lbfgs", max_iter=3000, class_weight="balanced",
        random_state=cfg.CFG.seed,
    )
    m_full.fit(sc_full.transform(X), y)
    preds_all = np.zeros((X_all.shape[0], 5))
    for i, cls in enumerate(m_full.classes_):
        preds_all[:, int(cls)] = m_full.predict_proba(X_all_s)[:, i]

    pred_df = pd.DataFrame({
        "patient_id": all_df["patient_id"].values,
        "side": all_df["side"].values,
        "subtype": [idx_to_label[int(i)] for i in preds_all.argmax(axis=1)],
    })
    for i, c in enumerate(CLASSES):
        pred_df[f"prob_{c}"] = preds_all[:, i]
    pred_df.to_parquet(cfg.CACHE_DIR / "track2_clinical_lr_preds.parquet", index=False)

    summary = {
        "lopo_acc": float(acc), "lopo_f1_macro": float(f1), "lopo_s2": float(s2),
        "per_class_f1": pc,
        "n_features": int(X.shape[1]),
    }
    (cfg.CACHE_DIR / "step9_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
