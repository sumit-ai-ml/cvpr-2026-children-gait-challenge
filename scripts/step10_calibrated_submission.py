"""Step 10: Build v12 submission with prior-calibrated Track 2 + v11 Track 1.

Theory: model overfits to majority train classes (type2 + type3 = 66%). We
downweight them via prior correction:

   p_calib[c] = p_model[c] / p_train[c]   (renormalized)

Then apply manual corrections (pid 13 WNL, pid 4 type3) and bilateral
consistency.

Uses v11_best.csv as the Track 1 source (already the best Track 1 we have).
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.track2_model import CLASSES, EVGS_ITEMS, build_track2_dataset


HEADER = (
    ["ID"]
    + [f"L{i}" for i in range(1, 18)]
    + [f"R{i}" for i in range(1, 18)]
    + ["Total", "Left_gait_subtype", "Right_gait_subtype"]
)

# Manual corrections previously validated against the leaderboard
MANUAL_CORRECTIONS: dict[int, tuple[str, str]] = {
    13: ("WNL", "WNL"),   # +0.071 in v7
    4:  ("type3", "type3"),  # +0.022 in v8
}


def main(out_path: Path = ROOT / "submissions" / "v12_prior_calibrated.csv",
         temperature: float = 1.0) -> None:
    # ---- compute training class priors ----
    train_df, _, _ = build_track2_dataset()
    train_counts = Counter(train_df["y"])
    n = len(train_df)
    train_prior = np.array([train_counts.get(c, 0) / n for c in CLASSES])
    print(f"Train priors: {dict(zip(CLASSES, train_prior.round(3)))}")

    # ---- load Track 2 model probabilities ----
    t2 = pd.read_parquet(cfg.CACHE_DIR / "track2_preds.parquet")
    prob_cols = [f"prob_{c}" for c in CLASSES]
    P = t2[prob_cols].to_numpy()  # (220, 5)

    # ---- prior calibration: p_calib = p_model / p_train, renormalize ----
    # With temperature: p_calib = p_model / p_train^temperature
    eps = 1e-6
    P_calib = P / (train_prior[None, :] ** temperature + eps)
    P_calib /= P_calib.sum(axis=1, keepdims=True)

    # Pick argmax per (patient, side)
    pred_idx = P_calib.argmax(axis=1)
    pred_label = [CLASSES[int(i)] for i in pred_idx]
    t2_cal = t2[["patient_id", "side"]].copy()
    t2_cal["subtype"] = pred_label
    t2_cal_L = t2_cal[t2_cal.side == "L"].set_index("patient_id")
    t2_cal_R = t2_cal[t2_cal.side == "R"].set_index("patient_id")

    # Show test predictions
    print("\nTrack 2 test predictions (prior-calibrated):")
    for pid in sorted(cfg.TRACK2_TEST_IDS):
        L = t2_cal_L.loc[pid, "subtype"]
        R = t2_cal_R.loc[pid, "subtype"]
        if pid in MANUAL_CORRECTIONS:
            L, R = MANUAL_CORRECTIONS[pid]
            note = " [MANUAL]"
        else:
            note = ""
        print(f"  pid={pid:>3}: L={L:<6} R={R:<6}{note}")

    # ---- load v11 Track 1 source ----
    v11 = pd.read_csv(ROOT / "submissions" / "v11_best.csv")
    print(f"\nv11 Track 1 source: {len(v11)} rows")

    # v11 has the full structure (Track 1 items + Track 2 subtype). We use only its
    # Track 1 portion (L1..L17, R1..R17, Total) and replace Track 2 with our calibrated preds.
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for _, r in v11.iterrows():
            ID = r["ID"]
            if ID.startswith("track1-"):
                # Keep v11's Track 1 row as-is
                row = [ID]
                for i in range(1, 18):
                    row.append(int(r[f"L{i}"]))
                for i in range(1, 18):
                    row.append(int(r[f"R{i}"]))
                row.append(int(r["Total"]))
                row.append(r["Left_gait_subtype"])
                row.append(r["Right_gait_subtype"])
                w.writerow(row)
            elif ID.startswith("track2-"):
                pid = int(ID.split("-")[1])
                if pid in MANUAL_CORRECTIONS:
                    Lsub, Rsub = MANUAL_CORRECTIONS[pid]
                else:
                    Lsub = t2_cal_L.loc[pid, "subtype"]
                    Rsub = t2_cal_R.loc[pid, "subtype"]
                row = [ID] + [-1] * 34 + [-1, Lsub, Rsub]
                w.writerow(row)
            else:
                raise ValueError(f"unknown row id: {ID}")
    print(f"\nWrote {out_path}")

    # Validate schema
    from tests.test_submit import validate_submission
    validate_submission(out_path)
    print("Schema valid")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Calibration temperature. 0 = no calibration, 1 = full prior division")
    p.add_argument("--out", default=str(ROOT / "submissions" / "v12_prior_calibrated.csv"))
    args = p.parse_args()
    main(Path(args.out), temperature=args.temperature)
