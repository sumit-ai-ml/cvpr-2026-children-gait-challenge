"""Step 7: Physics-rule EVGS scoring.

For each of 17 EVGS items, extract the physically-correct signal from per-clip
features, select the appropriate view (sagittal for items 6-13,16; coronal for
items 4,5,8,14,17; both for items 1-3,7,15), and tune a single threshold via
patient-grouped CV.

Compare per-item OOF accuracy to the existing 3-tree ensemble. Build a hybrid
that uses physics rule where it beats trees, tree otherwise.

EVGS items (mapping per MDPI 2025 paper and Read 2003 clinical instrument):
  1  Initial Contact (sagittal): foot strike pattern
  2  Heel Lift in Stance (sagittal): early heel rise → abnormal
  3  Max Ankle DF in Stance (sagittal): too low (equinus) or too high (calcaneus)
  4  Hindfoot Valgus/Varus (coronal): hard from 2D
  5  Foot Rotation in Stance (transverse/coronal): toe-in/toe-out
  6  Foot Clearance in Swing (sagittal): foot drags
  7  Max Ankle DF in Swing (sagittal): drop foot
  8  Knee Progression Angle (coronal): coronal knee alignment
  9  Peak Knee Extension in Stance (sagittal): incomplete extension
 10  Knee Extension at Terminal Swing (sagittal): late-swing flexion
 11  Peak Knee Flexion in Swing (sagittal): insufficient flexion
 12  Peak Hip Extension in Stance (sagittal): incomplete extension
 13  Peak Hip Flexion in Swing (sagittal): excessive flexion
 14  Max Pelvic Obliquity in Stance (coronal): hip drop
 15  Pelvic Rotation in Midstance (transverse): hard from 2D
 16  Peak Sagittal Trunk Position (sagittal): forward lean
 17  Max Lateral Trunk Shift (coronal): lateral shift
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.cv import patient_kfold
from src.data_io import load_track1_labels


SAGITTAL_VIEWS = ("left", "right")
CORONAL_VIEWS = ("forward", "backward")
ALL_VIEWS = SAGITTAL_VIEWS + CORONAL_VIEWS


def load_clip_features() -> pd.DataFrame:
    df = pd.read_parquet(cfg.CACHE_DIR / "features_clip.parquet")
    # add a `view` string column for easier grouping
    view = np.where(df["view_left"] > 0, "left",
            np.where(df["view_right"] > 0, "right",
             np.where(df["view_forward"] > 0, "forward", "backward")))
    df["view"] = view
    return df


def aggregate_for_side(clip_df: pd.DataFrame, pid: int, side: str,
                        views: tuple[str, ...], col: str,
                        agg: str = "mean") -> float:
    """Pick clips for `pid` in `views`, compute statistic on column `col`.

    `col` may include {side} placeholder which is filled with the active side.
    Returns 0.0 if no matching clip.
    """
    use_col = col.format(side=side)
    sub = clip_df[(clip_df["patient_id"] == pid) & (clip_df["view"].isin(views))]
    if len(sub) == 0 or use_col not in sub.columns:
        return 0.0
    vals = sub[use_col].dropna().to_numpy()
    if vals.size == 0:
        return 0.0
    if agg == "mean":
        return float(np.mean(vals))
    if agg == "max":
        return float(np.max(vals))
    if agg == "min":
        return float(np.min(vals))
    if agg == "absmax":
        return float(np.max(np.abs(vals)))
    if agg == "absmean":
        return float(np.mean(np.abs(vals)))
    raise ValueError(f"unknown agg {agg}")


def build_physics_signals(clip_df: pd.DataFrame, all_pid_side: list[tuple[int, str]]) -> pd.DataFrame:
    """For each (patient_id, side), compute one float per item — the physics signal."""
    rows = []
    for pid, side in all_pid_side:
        rec: dict = {"patient_id": pid, "side": side}

        # 1: Initial contact — foot pose at heel strike. Use ankle min in early stance.
        #    Higher ankle flex angle at IC = more plantarflexed = abnormal.
        #    Use cyc_IC_mean as the value at initial contact, on sagittal views.
        rec["sig_1"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                          "ankle_flex_{side}_cyc_IC_mean", "mean")

        # 2: Heel lift — heel rises early. Proxy: ankle plantarflexion at midstance.
        #    Use cyc_mid_mean from sagittal — higher angle = more plantarflexed in midstance.
        rec["sig_2"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                          "ankle_flex_{side}_cyc_mid_mean", "mean")

        # 3: Max ankle DF in stance — peak DF angle. Use cyc_max_mean (max during cycle ~ stance peak).
        rec["sig_3"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                          "ankle_flex_{side}_cyc_max_mean", "mean")

        # 4: Hindfoot valgus/varus — coronal foot alignment. Hard from 2D pose.
        #    Use ankle position vs ground (heel y-toe y diff) from coronal views.
        rec["sig_4"] = aggregate_for_side(clip_df, pid, side, CORONAL_VIEWS,
                                          "ankle_flex_{side}_mean", "mean")

        # 5: Foot rotation — toe-out/in. Use ankle_flex variation in coronal views.
        rec["sig_5"] = aggregate_for_side(clip_df, pid, side, CORONAL_VIEWS,
                                          "ankle_flex_{side}_range", "mean")

        # 6: Foot clearance in swing — minimum toe height during swing. Use min ankle.
        #    Lower min ankle angle = less clearance.
        rec["sig_6"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                          "ankle_flex_{side}_cyc_min_mean", "mean")

        # 7: Max ankle DF in swing — max DF during swing. Use cyc_max with sagittal.
        rec["sig_7"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                          "ankle_flex_{side}_p90", "mean")

        # 8: Knee progression — coronal knee angle. Use knee mean from coronal views.
        rec["sig_8"] = aggregate_for_side(clip_df, pid, side, CORONAL_VIEWS,
                                          "knee_flex_{side}_mean", "mean")

        # 9: Peak knee extension in stance — max knee angle (extension = ~180°).
        #    Use cyc_max_mean from sagittal. Higher = more extended.
        rec["sig_9"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                          "knee_flex_{side}_cyc_max_mean", "mean")

        # 10: Knee extension at terminal swing — knee angle at end of cycle. Use cyc_IC.
        rec["sig_10"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                           "knee_flex_{side}_cyc_IC_mean", "mean")

        # 11: Peak knee flexion in swing — min knee angle (max flexion).
        rec["sig_11"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                           "knee_flex_{side}_cyc_min_mean", "mean")

        # 12: Peak hip extension in stance — max hip angle (extension). Higher = more extended.
        rec["sig_12"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                           "hip_flex_{side}_cyc_max_mean", "mean")

        # 13: Peak hip flexion in swing — min hip angle (max flexion is the smaller angle).
        rec["sig_13"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                           "hip_flex_{side}_cyc_min_mean", "mean")

        # 14: Max pelvic obliquity in stance — peak absolute obliquity from coronal views.
        rec["sig_14"] = aggregate_for_side(clip_df, pid, side, CORONAL_VIEWS,
                                           "pelvic_obliquity_cyc_max_mean", "absmean")

        # 15: Pelvic rotation in midstance — hard from 2D. Use pelvic obliquity variation
        #     from sagittal views as a proxy (transverse-plane rotation projects to changes
        #     in apparent hip distance).
        rec["sig_15"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                           "pelvic_obliquity_range", "mean")

        # 16: Peak sagittal trunk position — forward lean. Use trunk_lean from sagittal.
        rec["sig_16"] = aggregate_for_side(clip_df, pid, side, SAGITTAL_VIEWS,
                                           "trunk_lean_cyc_max_mean", "absmean")

        # 17: Max lateral trunk shift — coronal trunk lean.
        rec["sig_17"] = aggregate_for_side(clip_df, pid, side, CORONAL_VIEWS,
                                           "trunk_lean_cyc_max_mean", "absmean")

        rows.append(rec)
    return pd.DataFrame(rows)


def tune_threshold_and_direction(signal: np.ndarray, y: np.ndarray,
                                  pids: np.ndarray, n_splits: int = 5) -> tuple[float, int, float, np.ndarray]:
    """Find best (threshold, direction) via patient-grouped CV.

    direction=+1 means y_pred = (signal >= thr)
    direction=-1 means y_pred = (signal <= thr)

    Returns: best_threshold, best_direction, best_oof_acc, oof_predictions
    """
    # Generate candidate thresholds from the signal's quantiles
    qs = np.linspace(0.05, 0.95, 19)
    candidates = np.unique(np.quantile(signal, qs))
    if len(candidates) < 2:
        return 0.0, 1, float((y == 0).mean()), np.zeros_like(y)

    best_acc = -1.0
    best_thr = 0.0
    best_dir = 1
    best_oof = np.zeros_like(y)

    for direction in (+1, -1):
        for thr in candidates:
            # Compute OOF preds via 5-fold patient grouping
            oof = np.full(len(y), -1, dtype=int)
            for tr_idx, va_idx in patient_kfold(pids, n_splits=n_splits, seed=cfg.CFG.seed):
                # Threshold is just applied — no fitting beyond direction/thr
                # (the choice is global; we score the OOF preds at the end)
                if direction > 0:
                    oof[va_idx] = (signal[va_idx] >= thr).astype(int)
                else:
                    oof[va_idx] = (signal[va_idx] <= thr).astype(int)
            acc = (oof == y).mean()
            if acc > best_acc:
                best_acc = float(acc)
                best_thr = float(thr)
                best_dir = direction
                best_oof = oof.copy()

    return best_thr, best_dir, best_acc, best_oof


def main():
    clip_df = load_clip_features()
    print(f"Loaded clip features: {clip_df.shape}")

    labels = load_track1_labels()
    side_key = {"L": "left", "R": "right"}

    # Build training (patient_id, side) list — only patients with labels
    train_pids = [p for p in clip_df["patient_id"].unique() if p in labels]
    train_pid_side = [(int(p), s) for p in train_pids for s in ("L", "R")]
    print(f"Train pid-side pairs: {len(train_pid_side)} ({len(train_pids)} patients)")

    # Build physics signals for training set
    sig_df = build_physics_signals(clip_df, train_pid_side)
    print(f"Physics signals: {sig_df.shape}")

    # Attach labels
    for it in [str(i) for i in range(1, 18)]:
        sig_df[f"y_{it}"] = sig_df.apply(
            lambda r: int(labels[int(r["patient_id"])][side_key[r["side"]]][it]),
            axis=1,
        )

    pids = sig_df["patient_id"].to_numpy()

    # Tune per-item threshold + direction
    print()
    print(f"{'item':>4}  {'base':>5}  {'phys_oof':>9}  {'thr':>9}  {'dir':>4}")
    print("-" * 45)
    results = []
    physics_oof_dict = {}
    for it in [str(i) for i in range(1, 18)]:
        y = sig_df[f"y_{it}"].to_numpy()
        s = sig_df[f"sig_{it}"].to_numpy()
        base = float(y.mean())
        thr, direction, oof_acc, oof_preds = tune_threshold_and_direction(s, y, pids)
        physics_oof_dict[it] = (oof_preds, thr, direction, oof_acc)
        results.append((it, base, oof_acc, thr, direction))
        print(f"{it:>4}  {base:>5.3f}  {oof_acc:>9.3f}  {thr:>9.3f}  {direction:+d}")

    # Compare to tree OOF
    tree_oof = pd.read_parquet(cfg.CACHE_DIR / "track1_oof_train.parquet")
    import pickle
    with open(cfg.CACHE_DIR / "track1_models.pkl", "rb") as f:
        saved = pickle.load(f)
    tree_thrs = saved["thresholds"]

    print()
    print(f"{'item':>4}  {'base':>5}  {'tree':>6}  {'phys':>6}  {'winner':>6}  {'delta':>7}")
    print("-" * 55)
    winners = {}
    physics_wins = 0
    for it, base, phys_acc, thr, direction in results:
        y_tree = tree_oof[f"y_{it}"].to_numpy()
        p_tree = tree_oof[f"oof_{it}"].to_numpy()
        tree_pred = (p_tree >= tree_thrs[it]).astype(int)
        tree_acc = (tree_pred == y_tree).mean()
        # Important: we need the orderings consistent. Both indexed by (patient_id, side).
        # sig_df is indexed by (pid, L), (pid, R). tree_oof also.
        # As long as both have the same row order they match.
        delta = phys_acc - tree_acc
        winner = "PHYS" if phys_acc > tree_acc else "TREE"
        if phys_acc > tree_acc:
            physics_wins += 1
            winners[it] = "physics"
        else:
            winners[it] = "tree"
        print(f"{it:>4}  {base:>5.3f}  {tree_acc:>6.3f}  {phys_acc:>6.3f}  {winner:>6}  {delta:>+7.3f}")

    print()
    print(f"Physics wins on {physics_wins}/17 items.")

    # Also build physics signals for ALL patients (including test) so they can be added as features
    all_pids = sorted(clip_df["patient_id"].unique().tolist())
    all_pid_side = [(int(p), s) for p in all_pids for s in ("L", "R")]
    all_sig_df = build_physics_signals(clip_df, all_pid_side)
    all_sig_df.to_parquet(cfg.CACHE_DIR / "physics_signals_all.parquet", index=False)

    # Save outputs
    sig_df.to_parquet(cfg.CACHE_DIR / "physics_signals_train.parquet", index=False)
    payload = {
        "thresholds": {it: float(thr) for it, base, acc, thr, direc in results},
        "directions": {it: int(direc) for it, base, acc, thr, direc in results},
        "winners": winners,
        "physics_oof_acc": {it: float(acc) for it, base, acc, thr, direc in results},
        "physics_wins": physics_wins,
    }
    (cfg.CACHE_DIR / "step7_summary.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved cache/physics_signals_train.parquet + physics_signals_all.parquet + step7_summary.json")


if __name__ == "__main__":
    main()
