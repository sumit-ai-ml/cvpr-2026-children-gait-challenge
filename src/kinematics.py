"""Joint-angle math + trajectory normalization + smoothing.

All angles in degrees. All coordinate inputs assume (x, y) image pixels
unless noted otherwise. Image-y increases downward.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from . import config as cfg


def joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Angle at p2 formed by rays p2->p1 and p2->p3, in degrees [0, 180].

    Works on arrays of shape (..., 2). Returns NaN where any input is NaN
    or where any ray has zero length.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    dot = np.sum(v1 * v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = dot / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    bad = (n1 == 0) | (n2 == 0) | np.isnan(n1) | np.isnan(n2)
    ang = np.where(bad, np.nan, ang)
    return ang


def signed_angle_from_vertical(p_top: np.ndarray, p_bot: np.ndarray) -> np.ndarray:
    """Angle of vector (p_top - p_bot) from the vertical (-y) axis, in degrees.

    Positive = leaning to the right in image coords. Useful for trunk lean and
    pelvic obliquity diagnostics. Returns degrees in [-180, 180].
    """
    dx = p_top[..., 0] - p_bot[..., 0]
    dy = p_top[..., 1] - p_bot[..., 1]
    # vertical-up in image is (0, -1). atan2(x, -y) gives signed angle from up.
    return np.degrees(np.arctan2(dx, -dy))


def normalize_keypoints(kpts: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Normalize (T, K, 2) keypoints by bbox so output is roughly in [0, 1].

    Subtracts bbox (x, y) and divides by bbox height (preserves aspect).
    NaNs propagate.
    """
    if kpts.ndim != 3 or kpts.shape[-1] != 2:
        raise ValueError(f"kpts must be (T, K, 2), got {kpts.shape}")
    if bbox.ndim != 2 or bbox.shape[-1] != 4:
        raise ValueError(f"bbox must be (T, 4), got {bbox.shape}")

    bx = bbox[:, 0:1][:, None, :]  # (T, 1, 1)
    by = bbox[:, 1:2][:, None, :]
    bh = bbox[:, 3:4]              # (T, 1)
    bh_safe = np.where(bh > 1.0, bh, np.nan)[:, None, :]
    out = np.stack([(kpts[..., 0:1] - bx) / bh_safe, (kpts[..., 1:2] - by) / bh_safe], axis=-1)
    out = out.squeeze(-2)  # back to (T, K, 2)
    return out.astype(np.float32)


def mask_low_confidence(kpts: np.ndarray, scores: np.ndarray, thr: float = cfg.KPT_SCORE_THR) -> np.ndarray:
    """Set kpts to NaN where score < thr. Returns a new array (does not mutate)."""
    mask = scores < thr  # (T, K)
    out = kpts.copy()
    out[mask] = np.nan
    return out


def _interp_nan_1d(y: np.ndarray) -> np.ndarray:
    """Linear-interp NaNs in a 1D array. Leading/trailing NaNs are nearest-filled.
    All-NaN input is returned unchanged.
    """
    y = y.astype(np.float64, copy=True)
    n = y.shape[0]
    good = ~np.isnan(y)
    if not good.any():
        return y
    idx = np.arange(n)
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


def interp_nans(kpts: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs along time for each (kpt, coord) channel.
    kpts: (T, K, 2). Returns (T, K, 2)."""
    out = kpts.copy()
    T, K, C = out.shape
    for k in range(K):
        for c in range(C):
            out[:, k, c] = _interp_nan_1d(out[:, k, c])
    return out


def smooth_trajectories(
    kpts: np.ndarray,
    window: int = cfg.SMOOTH_WINDOW,
    polyorder: int = cfg.SMOOTH_POLYORDER,
) -> np.ndarray:
    """Apply Savitzky-Golay along the time axis. Handles short sequences by shrinking the window.

    Assumes NaNs already interpolated. Pass-through for sequences shorter than polyorder+2.
    """
    T = kpts.shape[0]
    if T < polyorder + 2:
        return kpts.copy()
    w = min(window, T if T % 2 == 1 else T - 1)
    if w < polyorder + 2:
        return kpts.copy()
    if w % 2 == 0:
        w -= 1
    return savgol_filter(kpts, window_length=w, polyorder=polyorder, axis=0, mode="nearest").astype(kpts.dtype)


# ---- compound named angles -------------------------------------------------

def hip_flex_angle(shoulder: np.ndarray, hip: np.ndarray, knee: np.ndarray) -> np.ndarray:
    """Hip flexion: angle at hip between trunk (shoulder->hip) and thigh (hip->knee)."""
    return joint_angle(shoulder, hip, knee)


def knee_flex_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    return joint_angle(hip, knee, ankle)


def ankle_flex_angle(knee: np.ndarray, ankle: np.ndarray, toe: np.ndarray) -> np.ndarray:
    return joint_angle(knee, ankle, toe)


def trunk_lean(l_sho: np.ndarray, r_sho: np.ndarray, l_hip: np.ndarray, r_hip: np.ndarray) -> np.ndarray:
    """Trunk lean signed angle from vertical."""
    sho_mid = 0.5 * (l_sho + r_sho)
    hip_mid = 0.5 * (l_hip + r_hip)
    return signed_angle_from_vertical(sho_mid, hip_mid)


def pelvic_obliquity(l_hip: np.ndarray, r_hip: np.ndarray) -> np.ndarray:
    """Angle of the hip line from horizontal, in degrees. Positive = right hip lower."""
    dx = r_hip[..., 0] - l_hip[..., 0]
    dy = r_hip[..., 1] - l_hip[..., 1]
    return np.degrees(np.arctan2(dy, dx))
