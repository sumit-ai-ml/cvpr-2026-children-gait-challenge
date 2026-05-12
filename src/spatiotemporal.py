"""Spatiotemporal gait parameters — cadence, stride, step, stance/swing, double support.

All durations are expressed in seconds (using clip fps) and frames. All distances
are in normalized units (bbox-height-scaled).
"""
from __future__ import annotations

import numpy as np

from . import config as cfg
from . import gait_events as ge


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b is None or b == 0 or np.isnan(b):
        return default
    return float(a / b)


def compute_spatiotemporal(
    kpts_norm: np.ndarray,
    fps: float,
    hs_L: np.ndarray,
    hs_R: np.ndarray,
    to_L: np.ndarray,
    to_R: np.ndarray,
) -> dict[str, float]:
    """Compute spatiotemporal gait parameters from heel-strike / toe-off events.

    Returns a flat dict of feature_name -> float (0.0 when undefined).
    Side-suffixed entries follow the convention used elsewhere (e.g. cadence_L).
    """
    T = kpts_norm.shape[0]
    out: dict[str, float] = {}

    # --- per-side cycle duration + cadence ---
    for side, hs in (("L", hs_L), ("R", hs_R)):
        if len(hs) >= 2:
            cycle_frames = np.diff(hs)
            cycle_sec = cycle_frames / max(fps, 1.0)
            out[f"stride_duration_sec_{side}"] = float(np.mean(cycle_sec))
            out[f"stride_duration_std_{side}"] = float(np.std(cycle_sec))
            out[f"cadence_steps_per_min_{side}"] = float(60.0 / np.mean(cycle_sec) * 2.0)  # 2 steps per stride
        else:
            out[f"stride_duration_sec_{side}"] = 0.0
            out[f"stride_duration_std_{side}"] = 0.0
            out[f"cadence_steps_per_min_{side}"] = 0.0

    # --- stance / swing durations ---
    for side, hs, to in (("L", hs_L, to_L), ("R", hs_R, to_R)):
        stance = []
        swing = []
        for i, h in enumerate(hs):
            t = to[i] if i < len(to) else -1
            if t < 0:
                continue
            stance.append((t - h) / max(fps, 1.0))
            # swing = TO to next HS
            if i + 1 < len(hs):
                swing.append((hs[i + 1] - t) / max(fps, 1.0))
        out[f"stance_sec_mean_{side}"] = float(np.mean(stance)) if stance else 0.0
        out[f"swing_sec_mean_{side}"] = float(np.mean(swing)) if swing else 0.0
        out[f"stance_swing_ratio_{side}"] = _safe_div(np.mean(stance) if stance else 0.0,
                                                      np.mean(swing) if swing else 0.0)

    # --- double-support: a frame where BOTH heels are in stance (heel below mean) ---
    if T > 0:
        heel_L_y = kpts_norm[:, cfg.L_HEEL, 1]
        heel_R_y = kpts_norm[:, cfg.R_HEEL, 1]
        # "in stance" heuristic = heel below the per-foot median (deeper = on ground).
        med_L = np.nanmedian(heel_L_y)
        med_R = np.nanmedian(heel_R_y)
        in_stance_L = heel_L_y > med_L
        in_stance_R = heel_R_y > med_R
        both = in_stance_L & in_stance_R
        out["double_support_frac"] = float(np.mean(both)) if T > 0 else 0.0
    else:
        out["double_support_frac"] = 0.0

    # --- stride length and step length (in normalized units) ---
    # Use ankle x as the proxy for foot position.
    for side, hs, ank_idx in (("L", hs_L, cfg.L_ANK), ("R", hs_R, cfg.R_ANK)):
        if len(hs) >= 2:
            xs = kpts_norm[hs, ank_idx, 0]
            strides = np.abs(np.diff(xs))
            out[f"stride_length_mean_{side}"] = float(np.nanmean(strides))
            out[f"stride_length_std_{side}"] = float(np.nanstd(strides))
        else:
            out[f"stride_length_mean_{side}"] = 0.0
            out[f"stride_length_std_{side}"] = 0.0

    # --- step length: distance between contralateral ankles at heel-strike (x-axis displacement) ---
    if len(hs_L) > 0:
        x_L = kpts_norm[hs_L, cfg.L_ANK, 0]
        x_R_at_L_HS = kpts_norm[hs_L, cfg.R_ANK, 0]
        step_len_L = np.abs(x_L - x_R_at_L_HS)
        out["step_length_L_mean"] = float(np.nanmean(step_len_L)) if step_len_L.size else 0.0
    else:
        out["step_length_L_mean"] = 0.0
    if len(hs_R) > 0:
        x_R = kpts_norm[hs_R, cfg.R_ANK, 0]
        x_L_at_R_HS = kpts_norm[hs_R, cfg.L_ANK, 0]
        step_len_R = np.abs(x_R - x_L_at_R_HS)
        out["step_length_R_mean"] = float(np.nanmean(step_len_R)) if step_len_R.size else 0.0
    else:
        out["step_length_R_mean"] = 0.0

    # --- step width: lateral (y is image-y; for coronal views patient walks toward/away from camera
    # so x is lateral). Use abs(L_ank.x - R_ank.x) averaged over all stance frames.
    # NB: in sagittal views this is meaningless (~zero); features.py keeps view tags so models can learn.
    if T > 0:
        dx = np.abs(kpts_norm[:, cfg.L_ANK, 0] - kpts_norm[:, cfg.R_ANK, 0])
        dy = np.abs(kpts_norm[:, cfg.L_ANK, 1] - kpts_norm[:, cfg.R_ANK, 1])
        out["foot_lateral_separation_mean"] = float(np.nanmean(dx))
        out["foot_vertical_separation_mean"] = float(np.nanmean(dy))
    else:
        out["foot_lateral_separation_mean"] = 0.0
        out["foot_vertical_separation_mean"] = 0.0

    # --- foot clearance during swing (max foot height - min foot height per cycle, taken on toe) ---
    for side, hs, toe_idx in (("L", hs_L, cfg.L_BIGTOE), ("R", hs_R, cfg.R_BIGTOE)):
        clearances = []
        for i in range(len(hs) - 1):
            s, e = hs[i], hs[i + 1]
            seg = kpts_norm[s:e, toe_idx, 1]
            seg_good = seg[~np.isnan(seg)]
            if seg_good.size >= 3:
                # In image coords, lower y = higher off the ground.
                clearances.append(float(np.max(seg_good) - np.min(seg_good)))
        out[f"foot_clearance_mean_{side}"] = float(np.mean(clearances)) if clearances else 0.0
        out[f"foot_clearance_max_{side}"] = float(np.max(clearances)) if clearances else 0.0

    # --- cadence asymmetry between L and R ---
    out["cadence_asymmetry"] = float(abs(out["cadence_steps_per_min_L"] - out["cadence_steps_per_min_R"]))
    out["stride_length_asymmetry"] = float(abs(out["stride_length_mean_L"] - out["stride_length_mean_R"]))
    out["stance_swing_asymmetry"] = float(abs(out["stance_swing_ratio_L"] - out["stance_swing_ratio_R"]))

    return out
