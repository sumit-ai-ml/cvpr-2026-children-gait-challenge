"""Build a submission combining current Track 1 cache + v8 Track 2 predictions
(preserved on disk as submissions/v8_pid4_l_type3.csv). Used to safely test new
Track 1 model variations without losing the verified Track 2 lift from v8.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg

V8_PATH = ROOT / "submissions" / "v8_pid4_l_type3.csv"
HEADER = (
    ["ID"]
    + [f"L{i}" for i in range(1, 18)]
    + [f"R{i}" for i in range(1, 18)]
    + ["Total", "Left_gait_subtype", "Right_gait_subtype"]
)


def build(out_path: Path) -> None:
    if not V8_PATH.exists():
        raise FileNotFoundError(f"v8 reference submission missing: {V8_PATH}")
    v8 = pd.read_csv(V8_PATH)
    # Extract v8 Track 2 predictions (the source of our 0.639 score)
    v8_t2 = v8[v8.ID.str.startswith("track2-")].set_index("ID")[
        ["Left_gait_subtype", "Right_gait_subtype"]
    ]

    t1 = pd.read_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet")
    t1_L = t1[t1.side == "L"].set_index("patient_id")
    t1_R = t1[t1.side == "R"].set_index("patient_id")

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for pid in sorted(cfg.TRACK1_TEST_IDS):
            L = [int(t1_L.loc[pid, f"pred_{i}"]) for i in range(1, 18)]
            R = [int(t1_R.loc[pid, f"pred_{i}"]) for i in range(1, 18)]
            w.writerow([f"track1-{pid}"] + L + R + [sum(L) + sum(R), -1, -1])
        for pid in sorted(cfg.TRACK2_TEST_IDS):
            row_id = f"track2-{pid}"
            Lsub = v8_t2.loc[row_id, "Left_gait_subtype"]
            Rsub = v8_t2.loc[row_id, "Right_gait_subtype"]
            w.writerow([row_id] + [-1] * 34 + [-1, Lsub, Rsub])

    print(f"Wrote {out_path}")
    # validate
    from tests.test_submit import validate_submission
    validate_submission(out_path)
    print("Schema valid")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    build(Path(args.out))
