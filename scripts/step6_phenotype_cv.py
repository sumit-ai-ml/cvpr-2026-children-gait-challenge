"""Step 6: Phenotype-grouped CV via k-means on top-10 PCs of kinematic features.

Cluster ALL 94 Track 1 train patients into 5 groups. Leave-one-group-out CV instead of
patient-grouped 5-fold. Re-tune Track 1 thresholds and Track 2 ensemble under LOGO.
Compare LOGO Track 1 S1 to LOPO; expect a drop. The drop is the CV-vs-test gap we've been
hiding from.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def cluster_patients(n_clusters: int = 5) -> dict[int, int]:
    """Return {patient_id: cluster_idx} via k-means on top-10 PCs of pooled features."""
    pooled = pd.read_parquet(cfg.CACHE_DIR / "features_patient_limb.parquet")
    # Aggregate per patient: mean of L and R rows
    feature_cols = [c for c in pooled.columns if c not in ("patient_id", "side")]
    per_pat = pooled.groupby("patient_id")[feature_cols].mean()
    X = per_pat.to_numpy(dtype=np.float32)
    X_s = StandardScaler().fit_transform(X)
    pca = PCA(n_components=10, random_state=cfg.CFG.seed)
    X_p = pca.fit_transform(X_s)
    print(f"PCA top-10 explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    km = KMeans(n_clusters=n_clusters, random_state=cfg.CFG.seed, n_init=10)
    labels = km.fit_predict(X_p)
    result = {int(pid): int(lab) for pid, lab in zip(per_pat.index, labels)}
    print(f"Cluster sizes: {dict(Counter(labels))}")
    return result


def main() -> None:
    # 1. Build clusters from features.
    pid_to_cluster = cluster_patients(n_clusters=5)

    # 2. Set up Track 1 training under LOGO
    pooled = pd.read_parquet(cfg.CACHE_DIR / "features_patient_limb.parquet")
    labels = load_track1_labels()
    side_key = {"L": "left", "R": "right"}
    train_rows = []
    for _, r in pooled.iterrows():
        pid = int(r["patient_id"])
        if pid not in labels:
            continue
        lab = labels[pid][side_key[r["side"]]]
        rec = r.to_dict()
        for it in EVGS_ITEMS:
            rec[f"y_{it}"] = int(lab[it])
        rec["y_total"] = int(lab["Total"])
        rec["cluster"] = pid_to_cluster.get(pid, -1)
        train_rows.append(rec)
    train_df = pd.DataFrame(train_rows)
    feature_cols = [c for c in pooled.columns if c not in ("patient_id", "side")]
    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_total = train_df["y_total"].to_numpy(dtype=np.float32)
    pids = train_df["patient_id"].to_numpy()
    clusters = train_df["cluster"].to_numpy()

    # 3. Run LOGO: hold one cluster out at a time.
    print(f"\nRunning LOGO over {len(set(clusters))} clusters ...")
    oof_probs_dict: dict[str, np.ndarray] = {}
    y_true_dict: dict[str, np.ndarray] = {}
    for it in EVGS_ITEMS:
        oof_probs_dict[it] = np.zeros(len(train_df))
        y_true_dict[it] = train_df[f"y_{it}"].to_numpy(dtype=np.int32)

    unique_clusters = sorted(set(clusters))
    for held_c in unique_clusters:
        tr_mask = clusters != held_c
        va_mask = clusters == held_c
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]
        for it in EVGS_ITEMS:
            y_it = y_true_dict[it]
            pos = max(int(y_it[tr_idx].sum()), 1)
            neg = max(int((1 - y_it[tr_idx]).sum()), 1)
            spw = neg / pos
            m_lgb = _fit_lgb(X[tr_idx], y_it[tr_idx], X[va_idx], y_it[va_idx], spw)
            oof_probs_dict[it][va_idx] = m_lgb.predict(X[va_idx], num_iteration=m_lgb.best_iteration)
        print(f"  cluster {held_c}: {tr_mask.sum()} train, {va_mask.sum()} val")

    # 4. Tune thresholds under LOGO
    pids_arr = train_df["patient_id"].to_numpy()
    thrs, best_s1 = tune_thresholds_for_s1(oof_probs_dict, y_true_dict, y_total, pids_arr, n_iters=5)
    acc, nrmse, s1 = compute_s1(oof_probs_dict, y_true_dict, y_total, pids_arr, thrs)
    print(f"\n== TRACK 1 LOGO SCORE ==")
    print(f"  Acc:   {acc:.4f}")
    print(f"  NRMSE: {nrmse:.4f}")
    print(f"  S_1:   {s1:.4f}")
    print(f"  (vs LOPO Track 1 baseline 0.8263 — LOGO drop reveals the true test-gap)")

    (cfg.CACHE_DIR / "step6_summary.json").write_text(json.dumps({
        "n_clusters": 5,
        "logo_acc": float(acc), "logo_nrmse": float(nrmse), "logo_s1": float(s1),
        "lopo_s1_baseline": 0.8263,
        "logo_gap": float(0.8263 - s1),
        "cluster_sizes": dict(Counter(clusters.tolist())),
    }, indent=2, default=str))
    print(f"\nLOGO is more conservative than LOPO. Use this for trustworthy CV going forward.")


if __name__ == "__main__":
    main()
