"""Step 3: Codify manual corrections as automatic rules.

Apply rules on top of Step 1's predictions:
  Rule 1 (WNL recovery):     if pred_total <= thr_total AND top1_conf < thr_conf_low  -> override to WNL.
  Rule 2 (bilateral fix):    if L != R AND neither side has top1_conf > thr_conf_high -> set both = higher-conf side's class.

Tune (thr_total, thr_conf_low, thr_conf_high) on LOPO using Step 1's OOF predictions.
Verify pids 13 and 4 are recovered when test patients are scored.
"""
from __future__ import annotations

import json
import sys
from itertools import product
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
from src.evgs_bridge import bridge_feature_cols, build_bridge_table
from src.track2_model import CLASSES, build_track2_dataset, _fit_knn, _fit_lgb_multi, _fit_lr


def _fit_bridge_lr(Xb, y, C=10.0):
    return LogisticRegression(
        penalty="l2", C=C, multi_class="multinomial",
        solver="lbfgs", max_iter=2000, class_weight="balanced",
        random_state=cfg.CFG.seed,
    ).fit(Xb, y)


def main() -> None:
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    WNL_IDX = label_to_idx["WNL"]

    # Build augmented dataset (same as Step 1)
    train_df, all_df, _ = build_track2_dataset()
    bridge_df = build_bridge_table(use_oof_for_train=True)
    bridge_cols = bridge_feature_cols()
    train_df = train_df.merge(bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left")
    all_df_b = all_df.merge(bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left")
    feature_cols = [c for c in train_df.columns if c not in ("patient_id", "side", "y") and c not in bridge_cols]

    train_pids_set = set(train_df.patient_id.unique().tolist())
    test_pids_set = set(cfg.TRACK2_TEST_IDS)

    # Sourcing pseudo-labels (replicate Step 1's flow)
    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    Xb = train_df[bridge_cols].to_numpy(dtype=np.float32)
    y = np.array([label_to_idx[s] for s in train_df["y"].values], dtype=np.int64)

    sc_init = StandardScaler().fit(X)
    m_lgb_init = _fit_lgb_multi(X, y, n_class=5)
    m_knn_init = _fit_knn(sc_init.transform(X), y, k=3)
    m_lr_init = _fit_lr(sc_init.transform(X), y)
    m_b_init = _fit_bridge_lr(Xb, y)
    X_all = all_df_b[feature_cols].to_numpy(dtype=np.float32)
    Xb_all = all_df_b[bridge_cols].to_numpy(dtype=np.float32)

    def predict_probs(m, X_in):
        out = np.zeros((X_in.shape[0], 5))
        for j, c in enumerate(m.classes_):
            out[:, int(c)] = m.predict_proba(X_in)[:, j]
        return out

    p_knn_a = predict_probs(m_knn_init, sc_init.transform(X_all))
    p_b_a = predict_probs(m_b_init, Xb_all)
    blend_a = 0.2 * p_knn_a + 0.8 * p_b_a  # step1's weights

    aug_rows = []
    all_b_indexed = all_df_b.set_index(["patient_id", "side"])
    for i in range(len(all_df_b)):
        pid = int(all_df_b.iloc[i]["patient_id"])
        side = all_df_b.iloc[i]["side"]
        if pid in train_pids_set or pid in test_pids_set:
            continue
        if blend_a[i].max() >= 0.5:
            base = all_b_indexed.loc[(pid, side)].to_dict()
            base["patient_id"] = pid; base["side"] = side
            base["y"] = idx_to_label[int(blend_a[i].argmax())]
            aug_rows.append(base)
    augmented = pd.concat([train_df, pd.DataFrame(aug_rows)], ignore_index=True)
    Xa = augmented[feature_cols].to_numpy(dtype=np.float32)
    Xab = augmented[bridge_cols].to_numpy(dtype=np.float32)
    ya = np.array([label_to_idx[s] for s in augmented["y"].values], dtype=np.int64)
    pids_a = augmented["patient_id"].to_numpy()
    totals_a = augmented["bridge_total"].to_numpy()
    original_mask = np.array([int(p) in train_pids_set for p in pids_a])

    print(f"Augmented training: {len(augmented)} limbs")

    # OOF on original 22 with augmented training pool (Step 1's ensemble)
    oof_blend = np.zeros((len(ya), 5))
    for held_pid in sorted(train_pids_set):
        tr_idx = np.where(pids_a != held_pid)[0]
        va_idx = np.where((pids_a == held_pid) & original_mask)[0]
        if len(va_idx) == 0:
            continue
        sc = StandardScaler().fit(Xa[tr_idx])
        m_knn = _fit_knn(sc.transform(Xa[tr_idx]), ya[tr_idx], k=3)
        m_b = _fit_bridge_lr(Xab[tr_idx], ya[tr_idx])
        kp = np.zeros((len(va_idx), 5))
        for j, c in enumerate(m_knn.classes_):
            kp[:, int(c)] = m_knn.predict_proba(sc.transform(Xa[va_idx]))[:, j]
        bp = np.zeros((len(va_idx), 5))
        for j, c in enumerate(m_b.classes_):
            bp[:, int(c)] = m_b.predict_proba(Xab[va_idx])[:, j]
        oof_blend[va_idx] = 0.2 * kp + 0.8 * bp

    # Now apply post-hoc rules. Map oof_blend[original] back to (pid, side) for L/R lookup.
    orig_rows = augmented[original_mask].reset_index(drop=True)
    orig_indices = np.where(original_mask)[0]

    def decode(thr_total, thr_conf_low, thr_conf_high):
        base_preds = oof_blend.argmax(axis=1)
        base_confs = oof_blend.max(axis=1)
        out_preds = base_preds.copy()
        # Rule 1 (WNL recovery)
        for k in range(len(ya)):
            if not original_mask[k]:
                continue
            if totals_a[k] <= thr_total and base_confs[k] < thr_conf_low:
                out_preds[k] = WNL_IDX
        # Rule 2 (bilateral consistency): per-patient L/R fix
        # Group by patient_id
        pat_to_indices: dict[int, dict[str, int]] = {}
        for k in range(len(ya)):
            if not original_mask[k]:
                continue
            pid = int(pids_a[k])
            side = augmented.iloc[k]["side"]
            pat_to_indices.setdefault(pid, {})[side] = k
        for pid, sides in pat_to_indices.items():
            if "L" not in sides or "R" not in sides:
                continue
            kL, kR = sides["L"], sides["R"]
            if out_preds[kL] == out_preds[kR]:
                continue
            if base_confs[kL] > thr_conf_high or base_confs[kR] > thr_conf_high:
                continue
            # Neither side is highly confident: force both to higher-confidence side's class.
            if base_confs[kL] >= base_confs[kR]:
                out_preds[kR] = out_preds[kL]
            else:
                out_preds[kL] = out_preds[kR]
        return out_preds[original_mask]

    y_orig = ya[original_mask]
    def score_p(preds):
        acc = accuracy_score(y_orig, preds)
        f1 = f1_score(y_orig, preds, labels=list(range(5)), average="macro", zero_division=0)
        return acc, f1, (acc + f1) / 2

    baseline_preds = oof_blend.argmax(axis=1)[original_mask]
    base_acc, base_f1, base_s2 = score_p(baseline_preds)
    print(f"No-rules OOF baseline: Acc={base_acc:.4f} F1={base_f1:.4f} S2={base_s2:.4f}")

    print()
    print("Tuning (thr_total, thr_conf_low, thr_conf_high)...")
    best = (-1.0, None)
    for thr_total in (4, 5, 6, 7, 8, 9):
        for thr_cl in np.arange(0.40, 0.81, 0.05):
            for thr_ch in np.arange(0.50, 0.91, 0.05):
                preds = decode(thr_total, float(thr_cl), float(thr_ch))
                acc, f1, s2 = score_p(preds)
                if s2 > best[0]:
                    best = (s2, (int(thr_total), float(thr_cl), float(thr_ch), acc, f1))
    s2, (thr_total, thr_conf_low, thr_conf_high, acc, f1) = best
    print(f"BEST: thr_total={thr_total}  thr_conf_low={thr_conf_low:.2f}  thr_conf_high={thr_conf_high:.2f}")
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  S2={s2:.4f}")
    baseline = 0.5405
    print(f"\nStep 1 baseline: {baseline:.4f}")
    print(f"Delta: {s2 - baseline:+.4f}")

    # Apply to test set (using full-trained ensemble)
    print("\nApplying rules to test-set predictions (full-trained ensemble)...")
    sc_a = StandardScaler().fit(Xa)
    m_knn_a = _fit_knn(sc_a.transform(Xa), ya, k=3)
    m_b_a = _fit_bridge_lr(Xab, ya)
    p_knn = predict_probs(m_knn_a, sc_a.transform(X_all))
    p_b = predict_probs(m_b_a, Xb_all)
    p_blend = 0.2 * p_knn + 0.8 * p_b
    base_preds_all = p_blend.argmax(axis=1)
    base_confs_all = p_blend.max(axis=1)
    totals_all = all_df_b["bridge_total"].to_numpy()

    out_preds_all = base_preds_all.copy()
    for i in range(len(X_all)):
        if totals_all[i] <= thr_total and base_confs_all[i] < thr_conf_low:
            out_preds_all[i] = WNL_IDX

    # Bilateral fix per test patient
    test_pid_to_idx: dict[int, dict[str, int]] = {}
    for i in range(len(X_all)):
        pid = int(all_df_b.iloc[i]["patient_id"])
        side = all_df_b.iloc[i]["side"]
        test_pid_to_idx.setdefault(pid, {})[side] = i
    for pid, sides in test_pid_to_idx.items():
        if pid not in test_pids_set or "L" not in sides or "R" not in sides:
            continue
        iL, iR = sides["L"], sides["R"]
        if out_preds_all[iL] == out_preds_all[iR]:
            continue
        if base_confs_all[iL] > thr_conf_high or base_confs_all[iR] > thr_conf_high:
            continue
        if base_confs_all[iL] >= base_confs_all[iR]:
            out_preds_all[iR] = out_preds_all[iL]
        else:
            out_preds_all[iL] = out_preds_all[iR]

    # Verify pid 13 -> WNL and pid 4 bilateral fix
    print()
    for verify_pid in (13, 4):
        for side in ("L", "R"):
            idx = test_pid_to_idx.get(verify_pid, {}).get(side, None)
            if idx is None:
                continue
            label = idx_to_label[int(out_preds_all[idx])]
            base = idx_to_label[int(base_preds_all[idx])]
            conf = float(base_confs_all[idx])
            total = int(totals_all[idx])
            print(f"  pid={verify_pid:>3} {side}: base={base:<6} -> rule={label:<6}  (conf={conf:.2f}, pred_total={total})")

    # Persist
    rows = []
    for i in range(len(X_all)):
        rec = {
            "patient_id": int(all_df_b.iloc[i]["patient_id"]),
            "side": all_df_b.iloc[i]["side"],
            "subtype": idx_to_label[int(out_preds_all[i])],
        }
        for ci, cn in enumerate(CLASSES):
            rec[f"prob_{cn}"] = float(p_blend[i, ci])
        rows.append(rec)
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(cfg.CACHE_DIR / "track2_preds_step3.parquet", index=False)

    if s2 > baseline:
        out_df.to_parquet(cfg.CACHE_DIR / "track2_preds.parquet", index=False)
        print(f"\nSTEP 3 IMPROVED OOF: {baseline:.4f} -> {s2:.4f}. Updated cache/track2_preds.parquet")
    else:
        print(f"\nStep 3 did NOT improve OOF over Step 1. cache/track2_preds.parquet unchanged.")
        print(f"(Rules may still recover pids 13/4 on test even without OOF lift.)")

    (cfg.CACHE_DIR / "step3_summary.json").write_text(json.dumps({
        "thr_total": thr_total, "thr_conf_low": thr_conf_low, "thr_conf_high": thr_conf_high,
        "oof_acc": float(acc), "oof_f1": float(f1), "oof_s2": float(s2),
        "baseline_s2": baseline, "delta": float(s2 - baseline),
    }, indent=2))


if __name__ == "__main__":
    main()
