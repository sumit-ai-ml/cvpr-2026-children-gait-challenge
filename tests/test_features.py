"""Feature builder tests. Includes an end-to-end smoke test on a real clip."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as cfg
from src import features as feat
from src.data_io import list_patient_clips


def test_compute_per_frame_angles_shape():
    T = 30
    kpts = np.random.RandomState(0).randn(T, 23, 2).astype(np.float32)
    angles = feat.compute_per_frame_angles(kpts)
    expected = {"hip_flex_L", "hip_flex_R", "knee_flex_L", "knee_flex_R",
                "ankle_flex_L", "ankle_flex_R", "trunk_lean", "pelvic_obliquity"}
    assert set(angles.keys()) == expected
    for v in angles.values():
        assert v.shape == (T,)


def test_build_clip_features_smoke():
    clips = list_patient_clips(1)
    assert clips, "patient 0001 should exist in Dataset/dataset/"
    feats = feat.build_clip_features(clips[0])
    # Sanity: dictionary, has expected keys, no NaN/Inf
    assert isinstance(feats, dict)
    assert "patient_id" in feats and feats["patient_id"] == 1.0
    for k, v in feats.items():
        assert isinstance(v, (int, float)), f"{k} is {type(v)}"
        assert np.isfinite(v), f"{k} = {v} is not finite"


def test_build_clip_features_idempotent():
    clips = list_patient_clips(1)
    f1 = feat.build_clip_features(clips[0])
    f2 = feat.build_clip_features(clips[0])
    assert f1.keys() == f2.keys()
    for k in f1:
        assert f1[k] == f2[k], f"{k}: {f1[k]} != {f2[k]}"


def test_pool_to_patient_limb_shape():
    # Build features for one patient, two clips (or whatever exists).
    clips = list_patient_clips(1)
    rows = [feat.build_clip_features(c) for c in clips[:2]]
    df = pd.DataFrame(rows)
    pooled = feat.pool_to_patient_limb(df)
    assert len(pooled) == 2  # L + R rows
    assert set(pooled["side"]) == {"L", "R"}
    assert (pooled["patient_id"] == 1).all()
    # Side-suffixed columns should not appear in pooled output
    for c in pooled.columns:
        # We expect no raw '_L_' or '_R_' or trailing '_L'/'_R' in pooled column names
        assert not c.endswith("_L"), c
        assert not c.endswith("_R"), c
