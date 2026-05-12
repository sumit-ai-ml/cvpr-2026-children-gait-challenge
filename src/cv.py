"""Cross-validation helpers — strictly patient-grouped to prevent leakage.

A patient is NEVER split across train and val folds. Both sides (L/R) of
the same patient go into the same fold.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut


def patient_kfold(patient_ids: np.ndarray, n_splits: int = 5, seed: int = 42) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) folds where each patient appears in exactly one val fold.

    Args:
        patient_ids: array of length N (the dataset rows). Two rows for the same
            patient (e.g., L and R sides) must end up in the SAME fold.
    """
    patient_ids = np.asarray(patient_ids)
    unique_pids = np.unique(patient_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_pids_idx, val_pids_idx in kf.split(unique_pids):
        train_pids = set(unique_pids[train_pids_idx].tolist())
        val_pids = set(unique_pids[val_pids_idx].tolist())
        train_mask = np.array([p in train_pids for p in patient_ids])
        val_mask = np.array([p in val_pids for p in patient_ids])
        yield np.where(train_mask)[0], np.where(val_mask)[0]


def leave_one_patient_out(patient_ids: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """LOPO CV — one patient held out at a time. Both sides of held-out patient go to val."""
    patient_ids = np.asarray(patient_ids)
    logo = LeaveOneGroupOut()
    yield from logo.split(np.zeros((len(patient_ids), 1)), groups=patient_ids)
