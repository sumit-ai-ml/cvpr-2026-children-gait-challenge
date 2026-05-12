"""Train ST-GCN at clip level. Aggregate predictions per (patient, side) for OOF eval."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import config as cfg
from . import cv as cvmod
from . import kinematics as kn
from .data_io import ClipMeta, list_all_patient_ids, list_patient_clips, load_clip_sequence, load_track1_labels
from .stgcn import STGCN, NUM_NODES


T_TARGET = 64  # frames per clip (resampled)


def _prepare_clip(clip: ClipMeta) -> np.ndarray | None:
    """Return (2, T_TARGET, 23) float32 normalized pose sequence, or None on failure."""
    try:
        seq = load_clip_sequence(clip)
    except Exception:
        return None
    kpts = seq["keypoints"]
    scores = seq["keypoint_scores"]
    bbox = seq["bbox_xywh"]
    T = seq["n_frames"]
    if T < 3:
        return None
    kpts = kn.mask_low_confidence(kpts, scores)
    kpts = kn.interp_nans(kpts)
    kpts = kn.smooth_trajectories(kpts)
    if np.isnan(bbox).any():
        bbox = pd.DataFrame(bbox).ffill().bfill().to_numpy(dtype=np.float32)
    kpts_norm = kn.normalize_keypoints(kpts, bbox)  # (T, 23, 2)
    # Recenter around mid-hip and clip to safe range.
    midhip = 0.5 * (kpts_norm[:, cfg.L_HIP] + kpts_norm[:, cfg.R_HIP])
    kpts_norm = kpts_norm - midhip[:, None, :]
    kpts_norm = np.clip(kpts_norm, -3.0, 3.0)
    if np.isnan(kpts_norm).any() or np.isinf(kpts_norm).any():
        return None
    # Resample to T_TARGET frames by linear interpolation along time.
    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, T_TARGET)
    out = np.zeros((NUM_NODES, T_TARGET, 2), dtype=np.float32)
    for v in range(NUM_NODES):
        for c in range(2):
            out[v, :, c] = np.interp(t_new, t_old, kpts_norm[:, v, c])
    # Reshape to (C=2, T, V=23)
    out = out.transpose(2, 1, 0)  # (2, T, V)
    return out.astype(np.float32)


class ClipPoseDataset(Dataset):
    """Each item: (pose (2,T,V), label_L (17,), label_R (17,), patient_id, side_flag)."""

    def __init__(self, clip_metas: list[ClipMeta], labels: dict[int, dict], cache: dict[Path, np.ndarray] | None = None):
        self.clips = clip_metas
        self.labels = labels
        self.cache = cache if cache is not None else {}

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip = self.clips[idx]
        key = clip.path
        if key in self.cache:
            pose = self.cache[key]
        else:
            pose = _prepare_clip(clip)
            if pose is None:
                pose = np.zeros((2, T_TARGET, NUM_NODES), dtype=np.float32)
            self.cache[key] = pose
        lab = self.labels.get(clip.patient_id, None)
        if lab is not None:
            yL = np.array([int(lab["left"][str(i)]) for i in range(1, 18)], dtype=np.float32)
            yR = np.array([int(lab["right"][str(i)]) for i in range(1, 18)], dtype=np.float32)
        else:
            yL = np.full(17, -1.0, dtype=np.float32)
            yR = np.full(17, -1.0, dtype=np.float32)
        return (
            torch.from_numpy(pose),
            torch.from_numpy(np.concatenate([yL, yR])),  # (34,)
            int(clip.patient_id),
        )


def _build_clip_pool() -> tuple[list[ClipMeta], dict[int, dict]]:
    clips: list[ClipMeta] = []
    for pid in list_all_patient_ids():
        clips.extend(list_patient_clips(pid))
    labels = load_track1_labels()
    return clips, labels


def _train_one_fold(
    train_clips: list[ClipMeta],
    val_clips: list[ClipMeta],
    labels: dict[int, dict],
    pos_weight: torch.Tensor,
    device: torch.device,
    epochs: int = 25,
    cache: dict | None = None,
) -> tuple[STGCN, np.ndarray, list[int]]:
    """Return (model, val_probs (N_val, 34), val_pids)."""
    train_ds = ClipPoseDataset(train_clips, labels, cache=cache)
    val_ds = ClipPoseDataset(val_clips, labels, cache=cache)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = STGCN(in_channels=2, num_classes=34).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    for ep in range(epochs):
        model.train()
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    all_probs: list[np.ndarray] = []
    all_pids: list[int] = []
    with torch.no_grad():
        for x, y, pid in val_dl:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_pids.extend(pid.tolist())
    val_probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 34))
    return model, val_probs, all_pids


def train_stgcn() -> dict:
    """5-fold patient-grouped CV at clip level. Aggregates per (patient, side) for OOF."""
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ST-GCN training on {device}")

    all_clips, labels = _build_clip_pool()
    print(f"Total clips: {len(all_clips)}")
    # Only train clips for Track 1 train patients (94 patients).
    train_clips = [c for c in all_clips if c.patient_id in labels]
    print(f"Training clips: {len(train_clips)}")

    # Compute pos_weight per task (17 L + 17 R).
    pos_count = np.zeros(34, dtype=np.float32)
    neg_count = np.zeros(34, dtype=np.float32)
    for c in train_clips:
        lab = labels[c.patient_id]
        for i, it in enumerate(range(1, 18)):
            v = int(lab["left"][str(it)])
            pos_count[i] += v
            neg_count[i] += 1 - v
            v = int(lab["right"][str(it)])
            pos_count[17 + i] += v
            neg_count[17 + i] += 1 - v
    pos_weight = torch.from_numpy(np.clip(neg_count / np.clip(pos_count, 1, None), 1.0, 20.0).astype(np.float32))
    print(f"pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")

    pids_per_clip = np.array([c.patient_id for c in train_clips])
    unique_pids = np.unique(pids_per_clip)

    # Patient-grouped 5-fold (same seed as track1_model for fold compat)
    cache: dict[Path, np.ndarray] = {}

    oof = pd.DataFrame({
        "patient_id": pids_per_clip,
        "clip_path": [c.path.name for c in train_clips],
    })
    for it in range(1, 18):
        oof[f"oof_L_{it}"] = np.nan
        oof[f"oof_R_{it}"] = np.nan

    t0 = time.time()
    for fi, (tr_pids_idx, va_pids_idx) in enumerate(_pat_fold_indices(unique_pids, n_splits=5)):
        tr_pids = set(unique_pids[tr_pids_idx].tolist())
        va_pids = set(unique_pids[va_pids_idx].tolist())
        tr_clips_fold = [c for c in train_clips if c.patient_id in tr_pids]
        va_clips_fold = [c for c in train_clips if c.patient_id in va_pids]
        print(f"\n[fold {fi+1}/5] train_clips={len(tr_clips_fold)} val_clips={len(va_clips_fold)}")
        _, va_probs, va_pids_clip = _train_one_fold(
            tr_clips_fold, va_clips_fold, labels, pos_weight, device, epochs=25, cache=cache,
        )
        # Write OOF probs for those clips
        va_mask = np.zeros(len(train_clips), dtype=bool)
        for i, c in enumerate(train_clips):
            if c.patient_id in va_pids:
                va_mask[i] = True
        idxs = np.where(va_mask)[0]
        for k, idx in enumerate(idxs):
            for it in range(17):
                oof.at[idx, f"oof_L_{it+1}"] = float(va_probs[k, it])
                oof.at[idx, f"oof_R_{it+1}"] = float(va_probs[k, 17 + it])
        print(f"  elapsed: {time.time() - t0:.1f}s")

    # Aggregate clip-level OOF to (patient, side) level by averaging.
    print("\nAggregating clip-level OOF to (patient, side) ...")
    agg_rows = []
    for pid in sorted(set(pids_per_clip.tolist())):
        sub = oof[oof.patient_id == pid]
        for side in ("L", "R"):
            row = {"patient_id": pid, "side": side}
            for it in range(1, 18):
                col = f"oof_{side}_{it}"
                row[f"oof_{it}"] = float(sub[col].mean()) if not sub[col].isna().all() else 0.5
            agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_parquet(cfg.CACHE_DIR / "track1_stgcn_oof.parquet", index=False)
    print(f"Wrote cache/track1_stgcn_oof.parquet  shape={agg.shape}")

    # Refit on ALL train clips for inference.
    print("\nRefitting on full train set ...")
    full_model, _, _ = _train_one_fold(train_clips, train_clips[:64], labels, pos_weight, device, epochs=25, cache=cache)

    # Predict on ALL clips (for both train and test patients, all 110).
    all_ds = ClipPoseDataset(all_clips, labels, cache=cache)
    all_dl = DataLoader(all_ds, batch_size=64, shuffle=False, num_workers=0)
    full_model.eval()
    all_probs = []
    all_pids = []
    with torch.no_grad():
        for x, y, pid in all_dl:
            x = x.to(device)
            p = torch.sigmoid(full_model(x)).cpu().numpy()
            all_probs.append(p)
            all_pids.extend(pid.tolist())
    pcat = np.concatenate(all_probs, axis=0)

    # Aggregate to (patient_id, side)
    pred_rows = []
    pid_arr = np.array(all_pids)
    for pid in sorted(set(all_pids)):
        mask = pid_arr == pid
        avg = pcat[mask].mean(axis=0)  # (34,)
        pred_rows.append({"patient_id": int(pid), "side": "L", **{f"prob_{i}": float(avg[i - 1]) for i in range(1, 18)}})
        pred_rows.append({"patient_id": int(pid), "side": "R", **{f"prob_{i}": float(avg[17 + i - 1]) for i in range(1, 18)}})
    full_preds = pd.DataFrame(pred_rows)
    full_preds.to_parquet(cfg.CACHE_DIR / "track1_stgcn_full.parquet", index=False)
    print(f"Wrote cache/track1_stgcn_full.parquet  shape={full_preds.shape}")

    return {"oof_path": "cache/track1_stgcn_oof.parquet", "full_path": "cache/track1_stgcn_full.parquet"}


def _pat_fold_indices(unique_pids: np.ndarray, n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.CFG.seed)
    yield from kf.split(unique_pids)
