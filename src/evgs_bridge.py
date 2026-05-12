"""Build the 35-dim EVGS bridge feature vector for Track 2.

Per (patient, limb) row:
  - 17 raw OOF probabilities (oof_1..oof_17)
  - 17 thresholded binaries (oof_X >= threshold_X) -> pred_1..pred_17
  - 1 predicted Total = sum(binaries)

For Track 2 train patients (94 in Track 1): use OOF (leak-free).
For Track 2 test patients: use full-trained Track 1 predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

EVGS_ITEMS = [str(i) for i in range(1, 18)]


def load_thresholds() -> dict[str, float]:
    """Load per-item thresholds tuned on OOF S_1."""
    summary = json.loads((cfg.CACHE_DIR / "track1_summary.json").read_text())
    return {k: float(v) for k, v in summary["thresholds"].items()}


def build_bridge_table(use_oof_for_train: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns:
      patient_id, side,
      bridge_p_1..bridge_p_17  (raw probs),
      bridge_b_1..bridge_b_17  (binaries at tuned threshold),
      bridge_total             (sum of binaries)

    For each (pid, side) where pid is in Track 1 train (94 patients) AND
    use_oof_for_train=True: use OOF probs. For everyone else: use full-trained probs.
    """
    thrs = load_thresholds()
    oof = pd.read_parquet(cfg.CACHE_DIR / "track1_oof_train.parquet")
    full = pd.read_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet")

    oof_keys = set((int(r.patient_id), r.side) for _, r in oof.iterrows())
    rows = []
    for _, r in full.iterrows():
        pid = int(r["patient_id"])
        side = r["side"]
        key = (pid, side)
        if use_oof_for_train and key in oof_keys:
            o = oof[(oof.patient_id == pid) & (oof.side == side)].iloc[0]
            probs = np.array([float(o[f"oof_{i}"]) for i in EVGS_ITEMS])
        else:
            probs = np.array([float(r[f"prob_{i}"]) for i in EVGS_ITEMS])
        bins = (probs >= np.array([thrs[i] for i in EVGS_ITEMS])).astype(int)
        rec = {"patient_id": pid, "side": side}
        for k, it in enumerate(EVGS_ITEMS):
            rec[f"bridge_p_{it}"] = probs[k]
            rec[f"bridge_b_{it}"] = int(bins[k])
        rec["bridge_total"] = int(bins.sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def bridge_feature_cols() -> list[str]:
    return (
        [f"bridge_p_{i}" for i in EVGS_ITEMS]
        + [f"bridge_b_{i}" for i in EVGS_ITEMS]
        + ["bridge_total"]
    )
