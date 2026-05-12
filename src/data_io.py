"""Load per-frame JSON, labels, and enumerate patients/clips/views.

Clip directory naming: `{patient_id4}-{session2}_{view}_{start}-{end}_filtered_pose`
Example: `0004-0001_backward_679-873_filtered_pose`
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from . import config as cfg

CLIP_RE = re.compile(
    r"^(?P<pid>\d{4})-(?P<sess>\d{4})_(?P<view>forward|backward|left|right)_"
    r"(?P<start>\d+)-(?P<end>\d+)_filtered_pose$"
)


@dataclass(frozen=True)
class ClipMeta:
    patient_id: int
    session: int
    view: str
    start: int
    end: int
    path: Path

    @property
    def name(self) -> str:
        return self.path.name


def parse_clip_dir(p: Path) -> ClipMeta | None:
    m = CLIP_RE.match(p.name)
    if not m:
        return None
    return ClipMeta(
        patient_id=int(m["pid"]),
        session=int(m["sess"]),
        view=m["view"],
        start=int(m["start"]),
        end=int(m["end"]),
        path=p,
    )


def patient_dir(patient_id: int, root: Path = cfg.DATASET_DIR) -> Path:
    return root / f"{patient_id:04d}"


def list_patient_clips(patient_id: int, root: Path = cfg.DATASET_DIR) -> list[ClipMeta]:
    pdir = patient_dir(patient_id, root)
    if not pdir.exists():
        return []
    out: list[ClipMeta] = []
    for d in sorted(pdir.iterdir()):
        if d.is_dir():
            meta = parse_clip_dir(d)
            if meta is not None:
                out.append(meta)
    return out


def list_all_patient_ids(root: Path = cfg.DATASET_DIR) -> list[int]:
    if not root.exists():
        return []
    ids: list[int] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.isdigit():
            ids.append(int(d.name))
    return ids


def load_frame(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_clip_frames(clip: ClipMeta) -> Iterator[Path]:
    """Yield frame JSON paths in frame_index order."""
    yield from sorted(clip.path.glob("frame_*.json"))


def load_clip_sequence(clip: ClipMeta) -> dict:
    """Return a dict with stacked arrays:
    {
      'keypoints':       (T, K, 2) float32  — K = 23 (body + feet)
      'keypoint_scores': (T, K)    float32
      'bbox_xywh':       (T, 4)    float32
      'frame_index':     (T,)      int32
      'video_info':      dict
      'n_frames':        int
      'video_height':    int
      'video_width':     int
      'fps':             float
    }
    """
    frame_paths = list(iter_clip_frames(clip))
    if not frame_paths:
        raise FileNotFoundError(f"No frames in clip {clip.path}")

    # video_info isn't always on frame 0 — some clips only carry it on the last frames.
    # Scan a few from each end before falling back to defaults.
    vi: dict = {}
    scan_candidates = list(frame_paths[:5]) + list(frame_paths[-5:])
    for fp in scan_candidates:
        d = load_frame(fp)
        if "video_info" in d and d["video_info"]:
            vi = d["video_info"]
            break
    if not vi:
        # Per spec: 1920x1080. fps varies but 30 is a safe default for stats not derived from fps.
        vi = {"width": 1920, "height": 1080, "fps": 30.0, "video_name": clip.name + ".mp4", "total_frames": len(frame_paths)}
    K = len(cfg.KEPT_KPTS)
    T = len(frame_paths)

    kpts = np.full((T, K, 2), np.nan, dtype=np.float32)
    scores = np.zeros((T, K), dtype=np.float32)
    bbox = np.full((T, 4), np.nan, dtype=np.float32)
    fidx = np.zeros((T,), dtype=np.int32)

    for t, fp in enumerate(frame_paths):
        d = load_frame(fp)
        fidx[t] = int(d.get("frame_index", t))
        insts = d.get("instance_info", [])
        if not insts:
            continue
        inst = insts[0]  # one patient per frame
        all_kpts = np.asarray(inst.get("keypoints", []), dtype=np.float32)
        all_scores = np.asarray(inst.get("keypoint_scores", []), dtype=np.float32).reshape(-1)
        if all_kpts.ndim != 2 or all_kpts.shape[0] < max(cfg.KEPT_KPTS) + 1:
            continue
        kpts[t] = all_kpts[cfg.KEPT_KPTS, :2]
        if all_scores.shape[0] >= max(cfg.KEPT_KPTS) + 1:
            scores[t] = all_scores[cfg.KEPT_KPTS]
        bb = inst.get("gt_bbox_xywh_px", None)
        if bb is not None and len(bb) >= 4:
            bbox[t] = np.asarray(bb[:4], dtype=np.float32)

    return {
        "keypoints": kpts,
        "keypoint_scores": scores,
        "bbox_xywh": bbox,
        "frame_index": fidx,
        "video_info": vi,
        "n_frames": T,
        "video_height": int(vi.get("height", 1080)),
        "video_width": int(vi.get("width", 1920)),
        "fps": float(vi.get("fps", 30.0)),
    }


def load_track1_labels() -> dict[int, dict]:
    raw = json.loads(cfg.TRACK1_TRAIN_JSON.read_text())
    return {p["patient_id"]: p for p in raw}


def load_track2_labels() -> dict[int, dict]:
    raw = json.loads(cfg.TRACK2_TRAIN_JSON.read_text())
    return {p["patient_id"]: p for p in raw}
