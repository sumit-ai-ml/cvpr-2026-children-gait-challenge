"""Phase-normalized waveform features per gait cycle per joint angle.

For each detected gait cycle, resample the angle time-series to 101 points (% of cycle).
Then per-angle: DCT coefficients 1-5, max/min/range, peak timing index, stance-only mean,
swing-only mean, L-R cross-correlation lag (when paired).

This matches what clinical gait kinematics literature uses (Schwartz et al, Gait Profile
Score) and what EVGS items actually measure.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.fft import dct


N_PHASE = 101            # gait cycle normalized to 0..100 %
STANCE_FRAC = 0.6        # rough proportion of cycle in stance (0-60%)
DCT_COEFS = 5            # number of low-frequency DCT coefficients to keep


def _resample_cycle(values: np.ndarray, n: int = N_PHASE) -> np.ndarray:
    """Linear-interpolate a 1D series of length L to n points.
    NaNs are forward/backward filled.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.full(n, np.nan)
    # Fill NaNs
    if np.isnan(v).any():
        good = ~np.isnan(v)
        if not good.any():
            return np.full(n, 0.0)
        idx = np.arange(v.size)
        v = np.interp(idx, idx[good], v[good])
    if v.size == 1:
        return np.full(n, float(v[0]))
    src_t = np.linspace(0, 1, v.size)
    dst_t = np.linspace(0, 1, n)
    return np.interp(dst_t, src_t, v)


def _cycle_features(wave: np.ndarray) -> dict[str, float]:
    """Per-cycle features of one (101,) angle waveform."""
    if wave.size != N_PHASE or np.isnan(wave).all():
        return {
            "mean": 0.0, "max": 0.0, "min": 0.0, "range": 0.0,
            "argmax_phase": 0.0, "argmin_phase": 0.0,
            "stance_mean": 0.0, "swing_mean": 0.0, "stance_swing_diff": 0.0,
            **{f"dct_{k}": 0.0 for k in range(1, DCT_COEFS + 1)},
        }
    out = {
        "mean": float(np.mean(wave)),
        "max": float(np.max(wave)),
        "min": float(np.min(wave)),
        "range": float(np.max(wave) - np.min(wave)),
        "argmax_phase": float(np.argmax(wave)) / N_PHASE,
        "argmin_phase": float(np.argmin(wave)) / N_PHASE,
    }
    stance_end = int(STANCE_FRAC * N_PHASE)
    stance_part = wave[:stance_end]
    swing_part = wave[stance_end:]
    out["stance_mean"] = float(np.mean(stance_part)) if stance_part.size else 0.0
    out["swing_mean"] = float(np.mean(swing_part)) if swing_part.size else 0.0
    out["stance_swing_diff"] = out["stance_mean"] - out["swing_mean"]
    # Low-frequency DCT (Type-II, ortho)
    coefs = dct(wave, type=2, norm="ortho")
    for k in range(1, DCT_COEFS + 1):
        out[f"dct_{k}"] = float(coefs[k]) if k < coefs.size else 0.0
    return out


def _aggregate_cycles(rows: list[dict[str, float]]) -> dict[str, float]:
    """Take mean + std of each per-cycle feature across all cycles."""
    if not rows:
        keys = list(_cycle_features(np.zeros(N_PHASE)).keys())
        return {**{f"{k}_mean": 0.0 for k in keys}, **{f"{k}_std": 0.0 for k in keys}, "n_cycles": 0.0}
    keys = list(rows[0].keys())
    arr = np.array([[r.get(k, 0.0) for k in keys] for r in rows], dtype=np.float64)
    out = {}
    for i, k in enumerate(keys):
        out[f"{k}_mean"] = float(np.mean(arr[:, i]))
        out[f"{k}_std"] = float(np.std(arr[:, i]))
    out["n_cycles"] = float(len(rows))
    return out


def waveform_features_for_angle(
    angle_series: np.ndarray,
    cycles: list[tuple[int, int]],
) -> dict[str, float]:
    """Top-level: given per-frame angle series and list of (start, end) cycle boundaries,
    return aggregated waveform features.
    """
    per_cycle = []
    for s, e in cycles:
        if e - s < 5:
            continue
        wave = _resample_cycle(angle_series[s:e])
        per_cycle.append(_cycle_features(wave))
    return _aggregate_cycles(per_cycle)


def cross_corr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 30) -> float:
    """L-R cross-correlation lag at peak. Range: [-max_lag/N_PHASE, +max_lag/N_PHASE]."""
    if a.size == 0 or b.size == 0:
        return 0.0
    a = (a - np.mean(a)) / (np.std(a) + 1e-8)
    b = (b - np.mean(b)) / (np.std(b) + 1e-8)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag >= 0:
            corr = np.dot(a[lag:], b[: len(a) - lag])
        else:
            corr = np.dot(a[: len(a) + lag], b[-lag:])
        corrs.append(corr / max(len(a) - abs(lag), 1))
    return float(lags[int(np.argmax(corrs))]) / N_PHASE
