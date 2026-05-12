"""Spatiotemporal feature smoke tests."""
from __future__ import annotations

import numpy as np

from src import config as cfg
from src import spatiotemporal as st


def _fake_kpts(T: int = 90) -> np.ndarray:
    """Make a (T, 23, 2) array with sinusoidal heels/toes."""
    t = np.arange(T)
    K = len(cfg.KEPT_KPTS)
    kpts = np.zeros((T, K, 2), dtype=np.float32)
    # Heels: peaks at t=15, 45, 75 (cycle = 30 frames)
    heel = 0.5 + 0.3 * np.sin(2 * np.pi * t / 30 - np.pi / 2)
    toe = 0.5 + 0.3 * np.sin(2 * np.pi * (t - 5) / 30 - np.pi / 2)
    kpts[:, cfg.L_HEEL, 1] = heel
    kpts[:, cfg.R_HEEL, 1] = heel  # same heel pattern for simplicity
    kpts[:, cfg.L_BIGTOE, 1] = toe
    kpts[:, cfg.R_BIGTOE, 1] = toe
    # Ankles move along x for stride length
    kpts[:, cfg.L_ANK, 0] = 0.1 * t / T
    kpts[:, cfg.R_ANK, 0] = 0.1 * t / T + 0.05
    return kpts


def test_spatiotemporal_cadence_and_stride():
    T = 90
    kpts = _fake_kpts(T)
    fps = 30.0
    # Manual events.
    hs = np.array([15, 45, 75], dtype=np.int64)
    to = np.array([25, 55, 85], dtype=np.int64)
    feats = st.compute_spatiotemporal(kpts, fps, hs, hs, to, to)
    # Stride = 30 frames / 30 fps = 1s
    assert abs(feats["stride_duration_sec_L"] - 1.0) < 0.01
    # Cadence = 60 / stride * 2 steps = 120 steps/min
    assert abs(feats["cadence_steps_per_min_L"] - 120.0) < 1.0
    # Stance (HS to TO = 10 frames / 30 fps ≈ 0.333s)
    assert abs(feats["stance_sec_mean_L"] - 10.0 / 30.0) < 0.01
    # Swing (TO to next HS = 20 frames / 30 fps ≈ 0.667s)
    assert abs(feats["swing_sec_mean_L"] - 20.0 / 30.0) < 0.01


def test_spatiotemporal_empty_events():
    T = 30
    kpts = _fake_kpts(T)
    empty = np.array([], dtype=np.int64)
    feats = st.compute_spatiotemporal(kpts, 30.0, empty, empty, empty, empty)
    # All zero, no crash
    assert feats["cadence_steps_per_min_L"] == 0.0
    assert feats["stride_length_mean_L"] == 0.0
