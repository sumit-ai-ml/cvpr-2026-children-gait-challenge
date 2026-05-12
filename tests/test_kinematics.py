"""Math-correctness tests for src.kinematics. The joint_angle test is ★★★ critical:
if it's wrong, every feature is wrong."""
from __future__ import annotations

import numpy as np
import pytest

from src import kinematics as kn


def test_joint_angle_right_angle():
    p1 = np.array([0.0, 1.0])
    p2 = np.array([0.0, 0.0])
    p3 = np.array([1.0, 0.0])
    assert np.isclose(kn.joint_angle(p1, p2, p3), 90.0)


def test_joint_angle_straight():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 0.0])
    p3 = np.array([2.0, 0.0])
    assert np.isclose(kn.joint_angle(p1, p2, p3), 180.0)


def test_joint_angle_zero():
    # Collinear, same direction => 0 deg between rays.
    p1 = np.array([2.0, 0.0])
    p2 = np.array([0.0, 0.0])
    p3 = np.array([1.0, 0.0])
    assert np.isclose(kn.joint_angle(p1, p2, p3), 0.0)


def test_joint_angle_array():
    # Batch of three: right angle, straight, 45 deg.
    p1 = np.array([[0, 1], [0, 0], [1, 1]], dtype=float)
    p2 = np.array([[0, 0], [1, 0], [0, 0]], dtype=float)
    p3 = np.array([[1, 0], [2, 0], [1, 0]], dtype=float)
    out = kn.joint_angle(p1, p2, p3)
    assert np.allclose(out, [90.0, 180.0, 45.0])


def test_joint_angle_degenerate_returns_nan():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([0.0, 0.0])  # degenerate: ray of zero length
    p3 = np.array([1.0, 0.0])
    assert np.isnan(kn.joint_angle(p1, p2, p3))


def test_joint_angle_nan_input_returns_nan():
    p1 = np.array([np.nan, 0.0])
    p2 = np.array([0.0, 0.0])
    p3 = np.array([1.0, 0.0])
    assert np.isnan(kn.joint_angle(p1, p2, p3))


def test_signed_angle_from_vertical():
    top = np.array([0.0, 0.0])
    bot = np.array([0.0, 1.0])  # bot is below top in image coords -> straight up
    assert np.isclose(kn.signed_angle_from_vertical(top, bot), 0.0)
    # Lean to right: top displaced +x relative to bot
    top2 = np.array([1.0, 0.0])
    assert np.isclose(kn.signed_angle_from_vertical(top2, bot), 45.0)
    # Lean to left
    top3 = np.array([-1.0, 0.0])
    assert np.isclose(kn.signed_angle_from_vertical(top3, bot), -45.0)


def test_normalize_keypoints_basic():
    # bbox at (100, 50) with w=200, h=300. Kpt at (200, 200) -> ((100/300), (150/300)) = (0.333, 0.5)
    kpts = np.array([[[200.0, 200.0]]], dtype=np.float32)  # (T=1, K=1, 2)
    bbox = np.array([[100.0, 50.0, 200.0, 300.0]], dtype=np.float32)
    out = kn.normalize_keypoints(kpts, bbox)
    assert out.shape == (1, 1, 2)
    assert np.isclose(out[0, 0, 0], 100.0 / 300.0)
    assert np.isclose(out[0, 0, 1], 150.0 / 300.0)


def test_normalize_keypoints_zero_height_safe():
    kpts = np.array([[[10.0, 10.0]]], dtype=np.float32)
    bbox = np.array([[0.0, 0.0, 100.0, 0.0]], dtype=np.float32)
    out = kn.normalize_keypoints(kpts, bbox)
    assert np.isnan(out).all()


def test_mask_low_confidence():
    kpts = np.array([[[1.0, 1.0], [2.0, 2.0]]], dtype=np.float32)
    scores = np.array([[0.5, 0.1]], dtype=np.float32)
    out = kn.mask_low_confidence(kpts, scores, thr=0.2)
    assert np.array_equal(out[0, 0], [1.0, 1.0])
    assert np.isnan(out[0, 1]).all()


def test_interp_nans_fills_middle_gap():
    # x = [0, NaN, 2] -> [0, 1, 2]
    kpts = np.array([[[0.0, 0.0]], [[np.nan, np.nan]], [[2.0, 2.0]]], dtype=np.float64)
    out = kn.interp_nans(kpts)
    assert np.allclose(out[1, 0], [1.0, 1.0])


def test_interp_nans_all_nan_passes_through():
    kpts = np.full((3, 1, 2), np.nan, dtype=np.float64)
    out = kn.interp_nans(kpts)
    assert np.isnan(out).all()


def test_smooth_trajectories_preserves_constant():
    kpts = np.full((40, 1, 2), 5.0, dtype=np.float32)
    out = kn.smooth_trajectories(kpts, window=11, polyorder=3)
    assert np.allclose(out, 5.0)


def test_smooth_trajectories_short_sequence_passthrough():
    kpts = np.array([[[1.0, 1.0]], [[2.0, 2.0]]], dtype=np.float32)
    out = kn.smooth_trajectories(kpts)
    assert out.shape == kpts.shape
