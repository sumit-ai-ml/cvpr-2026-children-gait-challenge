"""★★★ critical: a patient must never appear in both train and val of the same fold.
This is the single most common silent CV bug."""
from __future__ import annotations

import numpy as np

from src import cv as cvmod


def test_patient_kfold_no_leak():
    # 20 patients, 2 sides each => 40 rows
    pids = np.repeat(np.arange(20), 2)
    sides = np.tile(["L", "R"], 20)
    seen_val_pids: set[int] = set()
    n_folds = 0
    for tr, va in cvmod.patient_kfold(pids, n_splits=5, seed=42):
        n_folds += 1
        tr_pids = set(pids[tr].tolist())
        va_pids = set(pids[va].tolist())
        assert tr_pids.isdisjoint(va_pids), f"leak: {tr_pids & va_pids}"
        # both sides of a held-out patient must be in val
        for pid in va_pids:
            mask = pids == pid
            assert mask.sum() == 2  # L and R
            assert np.all(np.isin(np.where(mask)[0], va)), f"patient {pid} split across folds"
        seen_val_pids |= va_pids
    assert n_folds == 5
    assert seen_val_pids == set(range(20)), "every patient must appear in exactly one val fold"


def test_patient_kfold_returns_indices():
    pids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    for tr, va in cvmod.patient_kfold(pids, n_splits=2, seed=0):
        assert isinstance(tr, np.ndarray)
        assert isinstance(va, np.ndarray)
        assert len(set(tr.tolist()) & set(va.tolist())) == 0


def test_leave_one_patient_out():
    pids = np.array([1, 1, 2, 2, 3, 3])
    folds = list(cvmod.leave_one_patient_out(pids))
    assert len(folds) == 3
    for tr, va in folds:
        va_pids = set(pids[va].tolist())
        assert len(va_pids) == 1
        # both sides of held-out patient in val
        assert len(va) == 2
