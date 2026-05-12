"""Step 4: Pseudo-label using EVGS-consistency, not raw model confidence.

Filter: keep a pseudo-label only when
  (a) hierarchical head (or 35-dim bridge ensemble) outputs same subtype with conf >= 0.6
  (b) pred_total is within the training-set range for that subtype.

Then refit Step 1's ensemble on this cleaner pool. Gate on OOF S2 vs Step 3 (0.5753).
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import cv as cvmod
from src.data_io import load_track2_labels
from src.evgs_bridge import bridge_feature_cols, build_bridge_table
from src.track2_model import CLASSES, build_track2_dataset, _fit_knn, _fit_lgb_multi, _fit_lr


def _fit_bridge_lr(Xb, y, C=10.0):
    return LogisticRegression(
        penalty="l2", C=C, multi_class="multinomial",
        solver="lbfgs", max_iter=2000, class_weight="balanced",
        random_state=cfg.CFG.seed,
    ).fit(Xb, y)


def main(conf_thr: float = 0.6) -> None:
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    train_df, all_df, _ = build_track2_dataset()
    bridge_df = build_bridge_table(use_oof_for_train=True)
    bridge_cols = bridge_feature_cols()
    train_df = train_df.merge(bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left")
    all_df_b = all_df.merge(bridge_df[["patient_id", "side"] + bridge_cols], on=["patient_id", "side"], how="left")
    feature_cols = [c for c in train_df.columns if c not in ("patient_id", "side", "y") and c not in bridge_cols]

    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    Xb = train_df[bridge_cols].to_numpy(dtype=np.float32)
    y = np.array([label_to_idx[s] for s in train_df["y"].values], dtype=np.int64)
    train_pids_set = set(train_df.patient_id.unique().tolist())
    test_pids_set = set(cfg.TRACK2_TEST_IDS)

    # Compute per-subtype training Total range (min, max).
    train_total_ranges: dict[str, tuple[int, int]] = {}
    totals_train = train_df["bridge_total"].to_numpy()
    for cls_name in CLASSES:
        mask = train_df["y"].values == cls_name
        if mask.sum() == 0:
            train_total_ranges[cls_name] = (0, 34)
            continue
        ts = totals_train[mask]
        # tolerant range (extend by 1 to allow some slack)
        train_total_ranges[cls_name] = (max(int(ts.min()) - 1, 0), int(ts.max()) + 1)
    print(f"Training total ranges per subtype:")
    for c, r in train_total_ranges.items():
        print(f"  {c}: total in [{r[0]}, {r[1]}]")

    # Initial Step 1 ensemble for sourcing pseudo-labels
    sc_init = StandardScaler().fit(X)
    m_knn_init = _fit_knn(sc_init.transform(X), y, k=3)
    m_b_init = _fit_bridge_lr(Xb, y)
    X_all = all_df_b[feature_cols].to_numpy(dtype=np.float32)
    Xb_all = all_df_b[bridge_cols].to_numpy(dtype=np.float32)

    def predict_probs(m, X_in):
        out = np.zeros((X_in.shape[0], 5))
        for j, c in enumerate(m.classes_):
            out[:, int(c)] = m.predict_proba(X_in)[:, j]
        return out

    p_knn_all = predict_probs(m_knn_init, sc_init.transform(X_all))
    p_b_all = predict_probs(m_b_init, Xb_all)
    blend_all = 0.2 * p_knn_all + 0.8 * p_b_all

    totals_all = all_df_b["bridge_total"].to_numpy()
    aug_rows = []
    all_b_indexed = all_df_b.set_index(["patient_id", "side"])
    n_conf_pass = n_total_pass = n_both_pass = 0
    for i in range(len(all_df_b)):
        pid = int(all_df_b.iloc[i]["patient_id"])
        side = all_df_b.iloc[i]["side"]
        if pid in train_pids_set or pid in test_pids_set:
            continue
        cls_idx = int(blend_all[i].argmax())
        cls_name = idx_to_label[cls_idx]
        conf = float(blend_all[i].max())
        total_i = int(totals_all[i])

        conf_ok = conf >= conf_thr
        total_lo, total_hi = train_total_ranges[cls_name]
        total_ok = total_lo <= total_i <= total_hi
        n_conf_pass += int(conf_ok)
        n_total_pass += int(total_ok)
        if conf_ok and total_ok:
            n_both_pass += 1
            base = all_b_indexed.loc[(pid, side)].to_dict()
            base["patient_id"] = pid; base["side"] = side
            base["y"] = cls_name
            aug_rows.append(base)

    print(f"\nPseudo-label filtering (conf_thr={conf_thr}):")
    print(f"  passed conf only:        {n_conf_pass}")
    print(f"  passed total-range only: {n_total_pass}")
    print(f"  passed BOTH (kept):      {n_both_pass}")
    print(f"  distribution: {dict(Counter(r['y'] for r in aug_rows))}")

    augmented = pd.concat([train_df, pd.DataFrame(aug_rows)], ignore_index=True)
    Xa = augmented[feature_cols].to_numpy(dtype=np.float32)
    Xab = augmented[bridge_cols].to_numpy(dtype=np.float32)
    ya = np.array([label_to_idx[s] for s in augmented["y"].values], dtype=np.int64)
    pids_a = augmented["patient_id"].to_numpy()
    totals_a = augmented["bridge_total"].to_numpy()
    original_mask = np.array([int(p) in train_pids_set for p in pids_a])

    # LOPO on original 22
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

    # Apply Step 3 rules to OOF
    WNL_IDX = label_to_idx["WNL"]
    thr_total, thr_conf_low, thr_conf_high = 4, 0.40, 0.80  # from Step 3 best
    base_preds = oof_blend.argmax(axis=1)
    base_confs = oof_blend.max(axis=1)
    out_preds = base_preds.copy()
    for k in range(len(ya)):
        if not original_mask[k]:
            continue
        if totals_a[k] <= thr_total and base_confs[k] < thr_conf_low:
            out_preds[k] = WNL_IDX
    pat_to_idx: dict[int, dict[str, int]] = {}
    for k in range(len(ya)):
        if not original_mask[k]:
            continue
        pid = int(pids_a[k]); side = augmented.iloc[k]["side"]
        pat_to_idx.setdefault(pid, {})[side] = k
    for pid, sides in pat_to_idx.items():
        if "L" not in sides or "R" not in sides:
            continue
        kL, kR = sides["L"], sides["R"]
        if out_preds[kL] == out_preds[kR]:
            continue
        if base_confs[kL] > thr_conf_high or base_confs[kR] > thr_conf_high:
            continue
        if base_confs[kL] >= base_confs[kR]:
            out_preds[kR] = out_preds[kL]
        else:
            out_preds[kL] = out_preds[kR]

    y_orig = ya[original_mask]
    preds_eval = out_preds[original_mask]
    acc = accuracy_score(y_orig, preds_eval)
    f1 = f1_score(y_orig, preds_eval, labels=list(range(5)), average="macro", zero_division=0)
    s2 = (acc + f1) / 2
    print(f"\nLOPO with Step 3 rules:  Acc={acc:.4f}  F1={f1:.4f}  S2={s2:.4f}")
    baseline = 0.5753
    delta = s2 - baseline
    print(f"Step 3 baseline: {baseline:.4f}")
    print(f"Delta: {delta:+.4f}")

    # Apply to test set
    sc_a = StandardScaler().fit(Xa)
    m_knn_a = _fit_knn(sc_a.transform(Xa), ya, k=3)
    m_b_a = _fit_bridge_lr(Xab, ya)
    p_knn = predict_probs(m_knn_a, sc_a.transform(X_all))
    p_b = predict_probs(m_b_a, Xb_all)
    p_blend = 0.2 * p_knn + 0.8 * p_b
    base_preds_all = p_blend.argmax(axis=1)
    base_confs_all = p_blend.max(axis=1)

    out_preds_all = base_preds_all.copy()
    for i in range(len(X_all)):
        if totals_all[i] <= thr_total and base_confs_all[i] < thr_conf_low:
            out_preds_all[i] = WNL_IDX
    test_pid_to_idx: dict[int, dict[str, int]] = {}
    for i in range(len(X_all)):
        pid = int(all_df_b.iloc[i]["patient_id"]); side = all_df_b.iloc[i]["side"]
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

    # Show test predictions
    print("\nTest set predictions:")
    for pid in cfg.TRACK2_TEST_IDS:
        sides = test_pid_to_idx.get(pid, {})
        if "L" in sides and "R" in sides:
            iL, iR = sides["L"], sides["R"]
            print(f"  pid={pid:>3}: L={idx_to_label[int(out_preds_all[iL])]:<6} R={idx_to_label[int(out_preds_all[iR])]:<6}")

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
    out_df.to_parquet(cfg.CACHE_DIR / "track2_preds_step4.parquet", index=False)
    if s2 > baseline:
        out_df.to_parquet(cfg.CACHE_DIR / "track2_preds.parquet", index=False)
        print(f"\nSTEP 4 IMPROVED OOF: {baseline:.4f} -> {s2:.4f}. Updated cache/track2_preds.parquet")
    else:
        print(f"\nStep 4 did NOT improve over Step 3. cache/track2_preds.parquet unchanged.")
    (cfg.CACHE_DIR / "step4_summary.json").write_text(json.dumps({
        "conf_thr": conf_thr, "n_pseudo_kept": n_both_pass,
        "oof_acc": float(acc), "oof_f1": float(f1), "oof_s2": float(s2),
        "baseline_s2": baseline, "delta": float(delta),
    }, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--conf", type=float, default=0.6)
    args = p.parse_args()
    main(conf_thr=args.conf)
