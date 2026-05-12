"""Augmented feature build: original clips + mirrored clips.

Mirrored features are computed in-process (no need to write mirrored frame JSONs).
Labels: when a (patient, side) row is mirrored, its EVGS items swap L↔R.

Output:
  cache/features_patient_limb_mirrored.parquet  — 220 rows (mirrored versions paired)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import features as feat
from src import kinematics as kn
from src import mirror_aug as ma
from src.data_io import ClipMeta, list_all_patient_ids, list_patient_clips, load_clip_sequence


def _features_for_clip(clip: ClipMeta, mirrored: bool) -> dict | None:
    """Compute features for a clip. If mirrored=True, flip x and swap L↔R indices first."""
    try:
        if not mirrored:
            return feat.build_clip_features(clip)
        # Manually load + mirror + run feature builder.
        # Simplest: rebuild feature builder with mirrored inputs.
        seq = load_clip_sequence(clip)
        kpts = ma.mirror_keypoints(seq["keypoints"])
        scores = ma.mirror_scores(seq["keypoint_scores"])
        bbox = ma.mirror_bbox(seq["bbox_xywh"])
        # Now call feat with these mirrored arrays. We need to inline the build_clip_features
        # to override the loaded values. Easiest: monkey-patch by writing a small helper.
        return _build_from_arrays(
            patient_id=clip.patient_id,
            session=clip.session,
            view=clip.view,
            kpts=kpts, scores=scores, bbox=bbox,
            fps=float(seq["fps"]),
            n_frames=seq["n_frames"],
        )
    except Exception as e:
        print(f"[ERROR] {clip.path.name} mirrored={mirrored}: {e}", file=sys.stderr)
        return None


def _build_from_arrays(
    patient_id: int, session: int, view: str,
    kpts: np.ndarray, scores: np.ndarray, bbox: np.ndarray, fps: float, n_frames: int,
) -> dict:
    """Replica of features.build_clip_features but accepting raw arrays (post-mirror)."""
    from src import config as cfg
    from src import features as feat
    from src import gait_events as ge
    from src import kinematics as kn
    from src import spatiotemporal as st

    out = {
        "patient_id": float(patient_id),
        "session": float(session),
        "n_frames": float(n_frames),
        "fps": float(fps),
        "view_forward": float(view == "forward"),
        "view_backward": float(view == "backward"),
        "view_left": float(view == "left"),
        "view_right": float(view == "right"),
    }
    kpts_masked = kn.mask_low_confidence(kpts, scores)
    kpts_filled = kn.interp_nans(kpts_masked)
    kpts_smooth = kn.smooth_trajectories(kpts_filled)
    if np.isnan(bbox).any():
        bbox = pd.DataFrame(bbox).ffill().bfill().to_numpy(dtype=np.float32)
    kpts_norm = kn.normalize_keypoints(kpts_smooth, bbox)
    angles = feat.compute_per_frame_angles(kpts_norm)
    for name, v in angles.items():
        out.update(feat._whole_clip_stats(v, name))
    # gait events
    cycles_L, cycles_R = [], []
    hs_L = np.array([], dtype=np.int64); hs_R = np.array([], dtype=np.int64)
    to_L = np.array([], dtype=np.int64); to_R = np.array([], dtype=np.int64)
    try:
        heel_L_y = kpts_norm[:, cfg.L_HEEL, 1]
        heel_R_y = kpts_norm[:, cfg.R_HEEL, 1]
        toe_L_y = kpts_norm[:, cfg.L_BIGTOE, 1]
        toe_R_y = kpts_norm[:, cfg.R_BIGTOE, 1]
        hs_L = ge.detect_heel_strikes(heel_L_y)
        hs_R = ge.detect_heel_strikes(heel_R_y)
        to_L = ge.detect_toe_offs(toe_L_y, hs_L)
        to_R = ge.detect_toe_offs(toe_R_y, hs_R)
        cycles_L = ge.segment_cycles(hs_L)
        cycles_R = ge.segment_cycles(hs_R)
    except Exception:
        pass
    out["n_cycles_L"] = float(len(cycles_L))
    out["n_cycles_R"] = float(len(cycles_R))
    try:
        st_feats = st.compute_spatiotemporal(kpts_norm, fps, hs_L, hs_R, to_L, to_R)
        out.update(st_feats)
    except Exception:
        pass
    side_to_cycles = {"L": cycles_L, "R": cycles_R}
    for name, v in angles.items():
        if name.endswith("_L"):
            out.update(feat._per_cycle_stats(v, side_to_cycles["L"], name))
        elif name.endswith("_R"):
            out.update(feat._per_cycle_stats(v, side_to_cycles["R"], name))
        else:
            chosen = cycles_L if len(cycles_L) >= len(cycles_R) else cycles_R
            out.update(feat._per_cycle_stats(v, chosen, name))
    return out


def main(n_workers: int = 12) -> None:
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_clips: list[ClipMeta] = []
    pids = list_all_patient_ids()
    for pid in pids:
        all_clips.extend(list_patient_clips(pid))
    print(f"Found {len(all_clips)} clips. Building original + mirrored = {len(all_clips)*2} feature sets.")

    rows: list[dict] = []
    t0 = time.time()
    tasks = [(c, False) for c in all_clips] + [(c, True) for c in all_clips]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_features_for_clip, c, m): (c, m) for c, m in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is not None:
                c, m = futs[fut]
                r["mirrored"] = int(m)
                rows.append(r)
            if i % 200 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] done in {time.time()-t0:.1f}s")

    clip_df = pd.DataFrame(rows)
    print(f"Combined clip table: {clip_df.shape}")

    # Pool by (patient_id, side, mirrored=0/1). For mirrored rows, the L side in features
    # corresponds to the original RIGHT limb's label, and vice versa.
    # We'll handle this in the train script by re-labeling the mirrored row.
    pool_orig = feat.pool_to_patient_limb(clip_df[clip_df.mirrored == 0])
    pool_orig["mirrored"] = 0
    pool_mirr = feat.pool_to_patient_limb(clip_df[clip_df.mirrored == 1])
    pool_mirr["mirrored"] = 1
    pooled = pd.concat([pool_orig, pool_mirr], ignore_index=True)
    print(f"Pooled (with mirrored): {pooled.shape}")

    out_path = cfg.CACHE_DIR / "features_patient_limb_mirrored.parquet"
    pooled.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
