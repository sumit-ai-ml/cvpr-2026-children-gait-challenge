"""Mirror augmentation: swap L↔R keypoint indices + horizontally flip x.

Gait is roughly mirror-symmetric across the sagittal plane. A horizontally-flipped
clip with L/R swapped is a valid, label-consistent additional training example.

Doubles N for Track 1 (and indirectly Track 2). Labels: swap left/right EVGS items.
"""
from __future__ import annotations

import numpy as np

from . import config as cfg

# COCO-WholeBody body+feet (0-22) L↔R index swap map.
# Body: 1<->2 (eyes), 3<->4 (ears), 5<->6 (shoulders), 7<->8 (elbows), 9<->10 (wrists),
#       11<->12 (hips), 13<->14 (knees), 15<->16 (ankles)
# Feet: 17<->20 (big toes), 18<->21 (small toes), 19<->22 (heels)
SWAP_MAP: dict[int, int] = {
    1: 2, 2: 1,
    3: 4, 4: 3,
    5: 6, 6: 5,
    7: 8, 8: 7,
    9: 10, 10: 9,
    11: 12, 12: 11,
    13: 14, 14: 13,
    15: 16, 16: 15,
    17: 20, 20: 17,
    18: 21, 21: 18,
    19: 22, 22: 19,
}


def mirror_keypoints(kpts: np.ndarray, bbox: np.ndarray | None = None) -> np.ndarray:
    """Mirror (T, K=23, 2) keypoints horizontally and swap L↔R indices.

    Flip x in image coordinates: x' = image_width - x. If bbox is provided,
    flip relative to image width (1920 default per challenge spec).
    """
    out = kpts.copy()
    # Apply index swap
    indices = np.arange(out.shape[1])
    for src, dst in SWAP_MAP.items():
        if src < out.shape[1] and dst < out.shape[1]:
            indices[src] = dst
    out = out[:, indices, :]
    # Flip x relative to image width
    # CGPS clips are all 1920x1080 per spec.
    image_width = 1920.0
    out[..., 0] = image_width - out[..., 0]
    return out


def mirror_bbox(bbox: np.ndarray, image_width: float = 1920.0) -> np.ndarray:
    """Mirror (T, 4) bbox xywh: new_x = image_width - (x + w)."""
    out = bbox.copy()
    out[:, 0] = image_width - (out[:, 0] + out[:, 2])
    return out


def mirror_evgs_labels(left: dict, right: dict) -> tuple[dict, dict]:
    """For a patient with (left, right) EVGS dicts, the mirrored version has them swapped."""
    return right.copy(), left.copy()


def mirror_subtype_labels(left_subtype: str, right_subtype: str) -> tuple[str, str]:
    """Track 2 labels also swap L/R."""
    return right_subtype, left_subtype


def mirror_scores(scores: np.ndarray) -> np.ndarray:
    """Apply the same L↔R swap to the keypoint scores (T, K)."""
    out = scores.copy()
    indices = np.arange(out.shape[1])
    for src, dst in SWAP_MAP.items():
        if src < out.shape[1] and dst < out.shape[1]:
            indices[src] = dst
    out = out[:, indices]
    return out
