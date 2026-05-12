"""Gait event detection tests. The synthetic-sinusoid test is ★★★ critical."""
from __future__ import annotations

import numpy as np

from src import gait_events as ge


def test_detect_heel_strikes_sinusoid():
    # 3 cycles of 30 frames. sin(2π*t/30 - π/2) peaks where 2π*t/30 - π/2 = π/2 → t = 15, 45, 75.
    n = 90
    t = np.arange(n)
    heel_y = 0.5 + 0.4 * np.sin(2 * np.pi * t / 30 - np.pi / 2)
    peaks = ge.detect_heel_strikes(heel_y, min_distance=15, prominence=0.05)
    assert len(peaks) == 3
    assert all(abs(p - target) < 3 for p, target in zip(sorted(peaks), [15, 45, 75]))


def test_detect_heel_strikes_flat_input():
    heel_y = np.full(60, 0.5, dtype=np.float64)
    peaks = ge.detect_heel_strikes(heel_y)
    assert len(peaks) == 0


def test_detect_heel_strikes_short_input():
    heel_y = np.array([0.1, 0.2, 0.3])
    peaks = ge.detect_heel_strikes(heel_y, min_distance=5)
    assert len(peaks) == 0


def test_detect_heel_strikes_with_nans():
    n = 90
    t = np.arange(n)
    heel_y = 0.5 + 0.4 * np.sin(2 * np.pi * t / 30 - np.pi / 2)
    heel_y[10:15] = np.nan  # gap that overlaps one peak
    peaks = ge.detect_heel_strikes(heel_y, min_distance=15, prominence=0.05)
    # Should still find ~3 peaks (NaNs were filled in)
    assert len(peaks) >= 2


def test_detect_toe_offs_returns_one_per_hs():
    n = 90
    t = np.arange(n)
    heel_y = 0.5 + 0.4 * np.sin(2 * np.pi * t / 30 - np.pi / 2)
    # Toe peaks shifted ~10 frames after heel peaks.
    toe_y = 0.5 + 0.4 * np.sin(2 * np.pi * (t - 10) / 30 - np.pi / 2)
    hs = ge.detect_heel_strikes(heel_y, min_distance=15, prominence=0.05)
    to = ge.detect_toe_offs(toe_y, hs)
    assert to.shape == hs.shape
    # All but possibly the last (no next HS) should be valid.
    assert (to[:-1] >= 0).all()


def test_segment_cycles_basic():
    hs = np.array([10, 40, 70, 100], dtype=np.int64)
    cycles = ge.segment_cycles(hs)
    assert cycles == [(10, 40), (40, 70), (70, 100)]


def test_segment_cycles_too_few():
    assert ge.segment_cycles(np.array([5], dtype=np.int64)) == []
    assert ge.segment_cycles(np.array([], dtype=np.int64)) == []
