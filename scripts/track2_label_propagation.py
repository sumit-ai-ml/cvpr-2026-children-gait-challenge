"""Transductive label propagation for Track 2 (codex idea #3).

Build a kNN graph over:
  - 44 Track 2 labeled limbs (known labels)
  - ~158 unlabeled limbs from the 79 patients not in Track 2 train (-1 labels)
  - 18 Track 2 test limbs (-1 labels)

Run sklearn LabelSpreading. Use propagated labels as a new Track 2 prediction signal.
Ensemble with the current pseudo-labeled Track 2 model probabilities.

Gate: only keep if OOF S2 improves on the 22 labeled patients (LOPO over them).
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
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import cv as cvmod
from src.data_io import load_track2_labels
from src.track2_model import CLASSES, EVGS_ITEMS, build_track2_dataset


def main() -> None:
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    train_df, all_df, feature_cols = build_track2_dataset()
    train_pids = set(train_df.patient_id.unique().tolist())
    test_pids = set(cfg.TRACK2_TEST_IDS)

    # Build feature matrix for ALL 220 (patient, side) rows.
    # Label vector: known for 44 train limbs, -1 for everything else.
    all_indexed = all_df.set_index(["patient_id", "side"]).copy()
    train_indexed = train_df.set_index(["patient_id", "side"])["y"].to_dict()

    rows = []
    labels = []
    keys = []
    for (pid, side), row in all_indexed.iterrows():
        feats = row[feature_cols].to_numpy(dtype=np.float32)
        rows.append(feats)
        if (pid, side) in train_indexed:
            labels.append(label_to_idx[train_indexed[(pid, side)]])
        else:
            labels.append(-1)
        keys.append((int(pid), side))
    X = np.stack(rows)
    y = np.array(labels, dtype=np.int64)
    print(f"Total limbs: {len(X)}  labeled: {(y >= 0).sum()}  unlabeled: {(y < 0).sum()}")
    print(f"Train labels distribution: {dict(Counter(y[y>=0].tolist()))}")

    # Normalize + PCA reduce (helps kNN in high-dim)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    pca = PCA(n_components=16, random_state=cfg.CFG.seed)
    Xp = pca.fit_transform(Xs)
    print(f"After PCA: shape {Xp.shape}, var explained {pca.explained_variance_ratio_.sum():.3f}")

    # Full-data LabelSpreading.
    print()
    print("Running LabelSpreading on full graph ...")
    ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=100, tol=1e-3)
    ls.fit(Xp, y)
    propagated = ls.transduction_
    label_dist_raw = ls.label_distributions_
    ls_classes = ls.classes_.astype(int)
    label_dist = np.zeros((len(propagated), 5), dtype=np.float64)
    for j, c in enumerate(ls_classes):
        label_dist[:, c] = label_dist_raw[:, j]
    print(f"  propagated label distribution: {dict(Counter(propagated.tolist()))}")

    # LOPO eval over the 22 original train patients only.
    print()
    print("LOPO eval over 22 train patients ...")
    baseline_s2 = 0.6526  # from track2_finalize_with_pseudo
    pids_all = np.array([k[0] for k in keys])

    correct_lp = []
    correct_blend = []
    y_true_eval = []
    # Read current Track 2 ensemble probs for blending
    cur = pd.read_parquet(cfg.CACHE_DIR / "track2_preds.parquet")
    prob_cols = [f"prob_{c}" for c in CLASSES]
    cur_idx = cur.set_index(["patient_id", "side"])

    for held_pid in sorted(train_pids):
        # Mask: hide held-out patient's labels, run LS again
        y_masked = y.copy()
        for i, (pid, side) in enumerate(keys):
            if pid == held_pid:
                y_masked[i] = -1
        ls_h = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=100, tol=1e-3)
        ls_h.fit(Xp, y_masked)
        prop_h = ls_h.transduction_
        prob_h_raw = ls_h.label_distributions_

        # Expand to 5 classes (the held-out class may be absent in train)
        ls_classes = ls_h.classes_.astype(int)
        prob_h = np.zeros((len(prop_h), 5), dtype=np.float64)
        for j, c in enumerate(ls_classes):
            prob_h[:, c] = prob_h_raw[:, j]

        # Evaluate on held-out patient's 2 limbs
        for i, (pid, side) in enumerate(keys):
            if pid != held_pid:
                continue
            if (pid, side) not in train_indexed:
                continue
            true_idx = label_to_idx[train_indexed[(pid, side)]]
            y_true_eval.append(true_idx)
            correct_lp.append(int(prop_h[i]))
            # Blend: current model probs + LS probs
            cur_probs = cur_idx.loc[(int(pid), side), prob_cols].to_numpy(dtype=np.float64)
            lp_probs = prob_h[i]
            blend = 0.5 * cur_probs + 0.5 * lp_probs
            correct_blend.append(int(np.argmax(blend)))

    y_true_eval = np.array(y_true_eval)
    correct_lp = np.array(correct_lp)
    correct_blend = np.array(correct_blend)

    def score(preds):
        acc = accuracy_score(y_true_eval, preds)
        f1 = f1_score(y_true_eval, preds, labels=list(range(5)), average="macro", zero_division=0)
        return acc, f1, (acc + f1) / 2

    acc_lp, f1_lp, s2_lp = score(correct_lp)
    acc_bl, f1_bl, s2_bl = score(correct_blend)
    print(f"LS-only:  Acc={acc_lp:.4f}  F1={f1_lp:.4f}  S2={s2_lp:.4f}")
    print(f"Blend 0.5: Acc={acc_bl:.4f}  F1={f1_bl:.4f}  S2={s2_bl:.4f}")
    print(f"Baseline (no LS): S2={baseline_s2:.4f}")

    # Try multiple blend weights
    best = (-1.0, None)
    for w_lp in np.arange(0.0, 1.01, 0.1):
        preds = []
        for i, (pid, side) in enumerate(keys):
            if pid not in train_pids:
                continue
            cur_probs = cur_idx.loc[(int(pid), side), prob_cols].to_numpy(dtype=np.float64)
            # Use full-data label_dist (NOT held-out) for this exploration — approximation
            lp_probs = label_dist[i]
            blend = (1 - w_lp) * cur_probs + w_lp * lp_probs
            preds.append(int(np.argmax(blend)))
        # Compute on full train
        full_true = [label_to_idx[train_indexed[(p, s)]] for p, s in keys if p in train_pids]
        preds = np.array(preds)
        full_true = np.array(full_true)
        acc, f1 = accuracy_score(full_true, preds), f1_score(full_true, preds, labels=list(range(5)), average="macro", zero_division=0)
        s2 = (acc + f1) / 2
        if s2 > best[0]:
            best = (s2, (float(w_lp), acc, f1))

    print(f"\nBlend-weight scan (training-set overfit warning, not LOPO):")
    print(f"  best: w_lp={best[1][0]}, train-fit S2={best[0]:.4f}")
    print(f"  honest LOPO blend (w=0.5): S2={s2_bl:.4f}")

    # Decision: use the LOPO blend score.
    if s2_bl > baseline_s2:
        print(f"\nLABEL PROPAGATION IMPROVED Track 2: {baseline_s2:.4f} -> {s2_bl:.4f}  (Δ=+{s2_bl-baseline_s2:.4f})")
        # Update cache: blend current track2_preds with LS probs at 0.5 weight (use full-data LS, the inference-time setup).
        out_rows = []
        for i, (pid, side) in enumerate(keys):
            cur_probs = cur_idx.loc[(int(pid), side), prob_cols].to_numpy(dtype=np.float64)
            lp_probs = label_dist[i]
            blend = 0.5 * cur_probs + 0.5 * lp_probs
            pred = idx_to_label[int(np.argmax(blend))]
            out_rows.append({
                "patient_id": int(pid), "side": side, "subtype": pred,
                **{f"prob_{c}": float(blend[idx]) for idx, c in enumerate(CLASSES)},
            })
        out_df = pd.DataFrame(out_rows)
        out_df.to_parquet(cfg.CACHE_DIR / "track2_preds.parquet", index=False)
        print(f"Updated cache/track2_preds.parquet")
        (cfg.CACHE_DIR / "track2_lp_summary.json").write_text(json.dumps({
            "baseline_s2": baseline_s2, "new_s2": float(s2_bl), "delta": float(s2_bl - baseline_s2),
            "method": "LabelSpreading blend 0.5",
        }, indent=2))
    else:
        print(f"\nLabel propagation did NOT improve Track 2 (baseline {baseline_s2:.4f}, got {s2_bl:.4f}). Discarding.")


if __name__ == "__main__":
    main()
