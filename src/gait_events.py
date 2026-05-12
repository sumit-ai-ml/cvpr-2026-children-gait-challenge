"""Gait event detection from 2D keypoints.

Strategy:
- Heel y-coordinate (image-y, increases downward) oscillates with each step.
  Stance phase = heel near its local maximum (lowest point in image).
- Heel strike (HS): local maximum of heel-y (foot first touches ground).
- Toe-off (TO):     local maximum of toe-y after HS, before the next HS.

This works across views — patient walking direction affects x more than y.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from . import config as cfg


def detect_heel_strikes(heel_y: np.ndarray, min_distance: int = cfg.MIN_CYCLE_FRAMES,
                        prominence: float = cfg.MIN_PEAK_PROMINENCE) -> np.ndarray:
    """Find heel-strike frame indices via local maxima of (normalized) heel-y.

    Args:
        heel_y: 1D array of heel y over time (normalized by bbox height).
        min_distance: minimum frames between heel strikes.
        prominence: minimum peak prominence.
    Returns:
        np.ndarray of int frame indices, possibly empty.
    """
    y = np.asarray(heel_y, dtype=np.float64)
    if y.size < min_distance + 1:
        return np.array([], dtype=np.int64)
    # Replace NaNs with the series mean for peak finding (does not affect detected indices' validity check below).
    if np.isnan(y).any():
        m = np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0
        y = np.where(np.isnan(y), m, y)
    peaks, _ = find_peaks(y, distance=min_distance, prominence=prominence)
    return peaks.astype(np.int64)


def detect_toe_offs(toe_y: np.ndarray, heel_strikes: np.ndarray,
                    min_distance: int = cfg.MIN_CYCLE_FRAMES // 2) -> np.ndarray:
    """For each heel strike, find the next toe-off (local max of toe-y) before the next HS.

    Returns an array of the same length as heel_strikes, with -1 where no TO was found.
    """
    if heel_strikes.size == 0:
        return np.array([], dtype=np.int64)
    y = np.asarray(toe_y, dtype=np.float64)
    if np.isnan(y).any():
        m = np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0
        y = np.where(np.isnan(y), m, y)
    n = y.size
    out = np.full(heel_strikes.shape, -1, dtype=np.int64)
    for i, hs in enumerate(heel_strikes):
        end = heel_strikes[i + 1] if i + 1 < len(heel_strikes) else n
        if end - hs < min_distance:
            continue
        window = y[hs:end]
        peaks, _ = find_peaks(window)
        if peaks.size == 0:
            continue
        # Choose the peak with the largest value (deepest toe-y = foot most lifted off the ground after rolling).
        best_local = peaks[np.argmax(window[peaks])]
        out[i] = hs + int(best_local)
    return out


def segment_cycles(heel_strikes: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start_frame, end_frame_exclusive), ...] for each detected gait cycle.

    A cycle is HS_i to HS_{i+1}.
    """
    if heel_strikes.size < 2:
        return []
    return [(int(heel_strikes[i]), int(heel_strikes[i + 1])) for i in range(len(heel_strikes) - 1)]
