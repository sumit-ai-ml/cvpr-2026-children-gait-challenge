"""Single source of truth for all pipeline knobs.

Anything tuneable lives here. Re-running with the same config = same outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Dataset"
DATASET_DIR = DATA_DIR / "dataset"
CACHE_DIR = ROOT / "cache"
SUBMIT_DIR = ROOT / "submissions"

TRACK1_TRAIN_JSON = DATA_DIR / "track1_train.json"
TRACK2_TRAIN_JSON = DATA_DIR / "track2_train.json"

TRACK1_TEST_IDS = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
TRACK2_TEST_IDS = [4, 6, 7, 13, 26, 35, 39, 42, 50]

# Keypoint indices we keep from COCO-WholeBody-133.
# 0-16: body, 17-22: feet. Drop face (23-90) and hands (91-132).
BODY_KPTS = list(range(17))
FEET_KPTS = list(range(17, 23))
KEPT_KPTS = BODY_KPTS + FEET_KPTS  # 23 keypoints total

# Named indices for readability.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16
L_BIGTOE, L_SMALLTOE, L_HEEL = 17, 18, 19
R_BIGTOE, R_SMALLTOE, R_HEEL = 20, 21, 22

# Views
VIEWS = ("forward", "backward", "left", "right")

# Confidence threshold below which a keypoint is masked.
KPT_SCORE_THR = 0.2

# Savitzky-Golay smoothing.
SMOOTH_WINDOW = 11
SMOOTH_POLYORDER = 3

# Gait event detection.
MIN_CYCLE_FRAMES = 15        # at 59 fps, that's ~0.25s — shortest plausible gait cycle
MIN_PEAK_PROMINENCE = 0.02   # in normalized units (fraction of bbox height)


@dataclass(frozen=True)
class Config:
    """Frozen run config so logs/cache keys can include it."""
    kpt_score_thr: float = KPT_SCORE_THR
    smooth_window: int = SMOOTH_WINDOW
    smooth_polyorder: int = SMOOTH_POLYORDER
    min_cycle_frames: int = MIN_CYCLE_FRAMES
    min_peak_prominence: float = MIN_PEAK_PROMINENCE
    seed: int = 42


CFG = Config()
