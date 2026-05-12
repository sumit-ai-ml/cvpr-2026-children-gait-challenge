"""Per-clip and per-patient feature extraction.

Pipeline per clip:
  1. Load (T, 23, 2) keypoints + bbox.
  2. Mask low confidence, interpolate NaNs, smooth.
  3. Normalize by bbox (height-scale).
  4. Compute per-frame joint angles (hip/knee/ankle for L+R, trunk lean, pelvic obliquity).
  5. Detect gait cycles via heel y.
  6. Aggregate per-cycle stats (min/max/range/mean/value-at-IC/value-at-mid/value-at-TO).
  7. Pool across cycles (mean + std).
  8. Append clip-level summary stats from the full sequence as a fallback.

If gait events fail (<2 cycles), fall back to whole-clip stats only — never produce all-NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import gait_events as ge
from . import kinematics as kn
from . import spatiotemporal as st
from . import waveform as wf
from .data_io import ClipMeta, load_clip_sequence


# ---- core per-frame angle computation -------------------------------------

def compute_per_frame_angles(kpts: np.ndarray) -> dict[str, np.ndarray]:
    """kpts: (T, 23, 2) normalized. Returns dict name -> (T,) array of angles in degrees."""
    L = lambda i: kpts[:, i, :]  # noqa: E731
    return {
        "hip_flex_L": kn.hip_flex_angle(L(cfg.L_SHO), L(cfg.L_HIP), L(cfg.L_KNE)),
        "hip_flex_R": kn.hip_flex_angle(L(cfg.R_SHO), L(cfg.R_HIP), L(cfg.R_KNE)),
        "knee_flex_L": kn.knee_flex_angle(L(cfg.L_HIP), L(cfg.L_KNE), L(cfg.L_ANK)),
        "knee_flex_R": kn.knee_flex_angle(L(cfg.R_HIP), L(cfg.R_KNE), L(cfg.R_ANK)),
        "ankle_flex_L": kn.ankle_flex_angle(L(cfg.L_KNE), L(cfg.L_ANK), L(cfg.L_BIGTOE)),
        "ankle_flex_R": kn.ankle_flex_angle(L(cfg.R_KNE), L(cfg.R_ANK), L(cfg.R_BIGTOE)),
        "trunk_lean": kn.trunk_lean(L(cfg.L_SHO), L(cfg.R_SHO), L(cfg.L_HIP), L(cfg.R_HIP)),
        "pelvic_obliquity": kn.pelvic_obliquity(L(cfg.L_HIP), L(cfg.R_HIP)),
    }


# ---- aggregation primitives -----------------------------------------------

def _safe_stat(fn, arr: np.ndarray, default: float = 0.0) -> float:
    a = arr[~np.isnan(arr)]
    if a.size == 0:
        return default
    return float(fn(a))


def _whole_clip_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return mean/std/min/max/range/p10/p90 for one (T,) series."""
    return {
        f"{prefix}_mean": _safe_stat(np.mean, values),
        f"{prefix}_std": _safe_stat(np.std, values),
        f"{prefix}_min": _safe_stat(np.min, values),
        f"{prefix}_max": _safe_stat(np.max, values),
        f"{prefix}_range": _safe_stat(lambda a: np.max(a) - np.min(a), values),
        f"{prefix}_p10": _safe_stat(lambda a: np.percentile(a, 10), values),
        f"{prefix}_p90": _safe_stat(lambda a: np.percentile(a, 90), values),
    }


def _per_cycle_stats(values: np.ndarray, cycles: list[tuple[int, int]], prefix: str) -> dict[str, float]:
    """Aggregate min/max/range/mean/IC/mid/TO across detected cycles, then take mean + std across cycles."""
    if not cycles:
        return {
            f"{prefix}_cyc_min_mean": 0.0, f"{prefix}_cyc_min_std": 0.0,
            f"{prefix}_cyc_max_mean": 0.0, f"{prefix}_cyc_max_std": 0.0,
            f"{prefix}_cyc_range_mean": 0.0, f"{prefix}_cyc_range_std": 0.0,
            f"{prefix}_cyc_IC_mean": 0.0, f"{prefix}_cyc_IC_std": 0.0,
            f"{prefix}_cyc_mid_mean": 0.0, f"{prefix}_cyc_mid_std": 0.0,
            f"{prefix}_n_cycles": 0.0,
        }
    mins, maxs, ranges, ics, mids = [], [], [], [], []
    for s, e in cycles:
        seg = values[s:e]
        seg_good = seg[~np.isnan(seg)]
        if seg_good.size == 0:
            continue
        mins.append(float(np.min(seg_good)))
        maxs.append(float(np.max(seg_good)))
        ranges.append(float(np.max(seg_good) - np.min(seg_good)))
        ics.append(float(seg[0]) if not np.isnan(seg[0]) else float(np.nanmean(seg)))
        mid_idx = (s + e) // 2 - s
        mid_idx = min(max(mid_idx, 0), seg.size - 1)
        m = seg[mid_idx]
        mids.append(float(m) if not np.isnan(m) else float(np.nanmean(seg)))

    def m_s(xs):
        if not xs:
            return 0.0, 0.0
        return float(np.mean(xs)), float(np.std(xs))

    mins_m, mins_s = m_s(mins)
    maxs_m, maxs_s = m_s(maxs)
    ranges_m, ranges_s = m_s(ranges)
    ics_m, ics_s = m_s(ics)
    mids_m, mids_s = m_s(mids)
    return {
        f"{prefix}_cyc_min_mean": mins_m, f"{prefix}_cyc_min_std": mins_s,
        f"{prefix}_cyc_max_mean": maxs_m, f"{prefix}_cyc_max_std": maxs_s,
        f"{prefix}_cyc_range_mean": ranges_m, f"{prefix}_cyc_range_std": ranges_s,
        f"{prefix}_cyc_IC_mean": ics_m, f"{prefix}_cyc_IC_std": ics_s,
        f"{prefix}_cyc_mid_mean": mids_m, f"{prefix}_cyc_mid_std": mids_s,
        f"{prefix}_n_cycles": float(len(mins)),
    }


# ---- main clip-level feature builder --------------------------------------

def build_clip_features(clip: ClipMeta) -> dict[str, float]:
    seq = load_clip_sequence(clip)
    kpts = seq["keypoints"]          # (T, 23, 2) pixel coords
    scores = seq["keypoint_scores"]  # (T, 23)
    bbox = seq["bbox_xywh"]          # (T, 4)
    T = seq["n_frames"]

    out: dict[str, float] = {
        "patient_id": float(clip.patient_id),
        "session": float(clip.session),
        "n_frames": float(T),
        "fps": float(seq["fps"]),
        "view_forward": float(clip.view == "forward"),
        "view_backward": float(clip.view == "backward"),
        "view_left": float(clip.view == "left"),
        "view_right": float(clip.view == "right"),
    }

    # 1. Mask, interpolate, smooth keypoints in pixel space first.
    kpts_masked = kn.mask_low_confidence(kpts, scores)
    kpts_filled = kn.interp_nans(kpts_masked)
    kpts_smooth = kn.smooth_trajectories(kpts_filled)

    # 2. Normalize by bbox height. NaNs in bbox -> NaN row (rare).
    # Fill bbox NaNs via forward-fill so normalization is stable.
    if np.isnan(bbox).any():
        bbox = pd.DataFrame(bbox).ffill().bfill().to_numpy(dtype=np.float32)
    kpts_norm = kn.normalize_keypoints(kpts_smooth, bbox)

    # 3. Per-frame angles.
    angles = compute_per_frame_angles(kpts_norm)

    # 4. Whole-clip stats for each angle.
    for name, v in angles.items():
        out.update(_whole_clip_stats(v, name))

    # 5. Gait events from heels (one per leg). Also detect toe-offs for spatiotemporal params.
    cycles_L: list[tuple[int, int]] = []
    cycles_R: list[tuple[int, int]] = []
    import numpy as _np
    hs_L = _np.array([], dtype=_np.int64)
    hs_R = _np.array([], dtype=_np.int64)
    to_L = _np.array([], dtype=_np.int64)
    to_R = _np.array([], dtype=_np.int64)
    try:
        heel_L_y = kpts_norm[:, cfg.L_HEEL, 1]
        heel_R_y = kpts_norm[:, cfg.R_HEEL, 1]
        toe_L_y = kpts_norm[:, cfg.L_BIGTOE, 1]
        toe_R_y = kpts_norm[:, cfg.R_BIGTOE, 1]
        hs_L = ge.detect_heel_strikes(heel_L_y)
        hs_R = ge.detect_heel_strikes(heel_R_y)
        to_L = ge.detect_toe_offs(toe_L_y, hs_L)
        to_R = ge.detect_toe_offs(toe_R_y, hs_R)
        cycles_L = ge.segment_cycles(hs_L)
        cycles_R = ge.segment_cycles(hs_R)
    except Exception:
        pass

    out["n_cycles_L"] = float(len(cycles_L))
    out["n_cycles_R"] = float(len(cycles_R))

    # 5b. Spatiotemporal gait parameters (cadence, stride/step length, stance/swing, clearance...).
    try:
        st_feats = st.compute_spatiotemporal(kpts_norm, float(seq["fps"]), hs_L, hs_R, to_L, to_R)
        out.update(st_feats)
    except Exception:
        pass

    # 6. Per-cycle stats using L cycles for L-side angles and R cycles for R-side angles.
    side_to_cycles = {"L": cycles_L, "R": cycles_R}
    for name, v in angles.items():
        if name.endswith("_L"):
            out.update(_per_cycle_stats(v, side_to_cycles["L"], name))
        elif name.endswith("_R"):
            out.update(_per_cycle_stats(v, side_to_cycles["R"], name))
        else:
            # bilateral metrics (trunk_lean, pelvic_obliquity): use whichever side has more cycles
            chosen = cycles_L if len(cycles_L) >= len(cycles_R) else cycles_R
            out.update(_per_cycle_stats(v, chosen, name))

    # NOTE: Waveform features (DCT, stance/swing summaries) were tested and discarded:
    # OOF S_1 dropped 0.8267 -> 0.8239 due to dim explosion noise. Kept code in src/waveform.py
    # for the report's ablation table but not wired in.

    return out


# ---- patient-limb pooling --------------------------------------------------

# Regex helpers to identify side-suffixed column names. Examples that match:
#   hip_flex_L_mean, hip_flex_L_cyc_max_std, n_cycles_L
# We treat a name as L-side iff it contains "_L_" or ends with "_L".
def _is_side_col(col: str, side: str) -> bool:
    return f"_{side}_" in col or col.endswith(f"_{side}")


def _strip_side(col: str, side: str) -> str:
    """hip_flex_L_mean -> hip_flex_mean, n_cycles_L -> n_cycles."""
    if col.endswith(f"_{side}"):
        return col[: -2]
    return col.replace(f"_{side}_", "_", 1)


def pool_to_patient_limb(clip_df: pd.DataFrame, view_separate: bool = True) -> pd.DataFrame:
    """One row per (patient_id, side) with features named consistently for both sides.

    Side-suffixed columns (e.g. `hip_flex_L_mean`) are renamed:
       - for L-side row: hip_flex_L_mean -> ipsi_hip_flex_mean, hip_flex_R_mean -> contra_hip_flex_mean
       - for R-side row: hip_flex_R_mean -> ipsi_hip_flex_mean, hip_flex_L_mean -> contra_hip_flex_mean
    Bilateral columns are kept unchanged. View-specific pooling: mean over each view,
    plus an overall mean and std across all clips for the patient.
    """
    numeric_cols = [
        c for c in clip_df.columns
        if c not in ("patient_id", "session") and pd.api.types.is_numeric_dtype(clip_df[c])
    ]
    rows = []
    for pid, group in clip_df.groupby("patient_id"):
        for side in ("L", "R"):
            opp = "R" if side == "L" else "L"
            row: dict[str, float] = {"patient_id": int(pid), "side": side}

            # Overall (across all clips) mean + std
            for col in numeric_cols:
                if _is_side_col(col, side):
                    out_name = "ipsi_" + _strip_side(col, side)
                elif _is_side_col(col, opp):
                    out_name = "contra_" + _strip_side(col, opp)
                else:
                    out_name = col
                row[f"{out_name}_clip_mean"] = float(group[col].mean()) if not group[col].isna().all() else 0.0
                row[f"{out_name}_clip_std"] = float(group[col].std(ddof=0)) if group[col].shape[0] > 1 else 0.0

            # Per-view means (forward, backward, left, right)
            if view_separate:
                for v in cfg.VIEWS:
                    sub = group[group[f"view_{v}"] > 0]
                    for col in numeric_cols:
                        if col.startswith("view_"):
                            continue
                        if _is_side_col(col, side):
                            out_name = "ipsi_" + _strip_side(col, side)
                        elif _is_side_col(col, opp):
                            out_name = "contra_" + _strip_side(col, opp)
                        else:
                            out_name = col
                        key = f"{out_name}_v{v}_mean"
                        row[key] = float(sub[col].mean()) if len(sub) > 0 and not sub[col].isna().all() else 0.0
            rows.append(row)

    df = pd.DataFrame(rows)
    # Replace any inf/NaN with 0 for tree models (LightGBM handles NaN but we'll be explicit).
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Symmetry features: for each ipsi_X column, if a contra_X exists, add sym_X and avg_X.
    ipsi_cols = [c for c in df.columns if c.startswith("ipsi_")]
    sym_dict: dict[str, pd.Series] = {}
    for ic in ipsi_cols:
        cc = "contra_" + ic[len("ipsi_"):]
        if cc in df.columns:
            sym_dict["sym_abs_" + ic[len("ipsi_"):]] = (df[ic] - df[cc]).abs()
            sym_dict["sym_avg_" + ic[len("ipsi_"):]] = (df[ic] + df[cc]) / 2.0
    if sym_dict:
        df = pd.concat([df, pd.DataFrame(sym_dict)], axis=1)
    return df
