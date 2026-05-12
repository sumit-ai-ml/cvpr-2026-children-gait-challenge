"""Build a submission CSV from cached predictions.

By default combines: Track 1 = cached LightGBM predictions
                     Track 2 = cached Track 2 predictions if present, else Day-0 majority
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg


HEADER = (
    ["ID"]
    + [f"L{i}" for i in range(1, 18)]
    + [f"R{i}" for i in range(1, 18)]
    + ["Total", "Left_gait_subtype", "Right_gait_subtype"]
)


def _track2_fallback() -> tuple[str, str]:
    """Majority class from track2_train.json. Used until Track 2 model is trained."""
    raw = json.loads(cfg.TRACK2_TRAIN_JSON.read_text())
    from collections import Counter
    left = Counter(p["left"]["gait_subtype"] for p in raw)
    right = Counter(p["right"]["gait_subtype"] for p in raw)
    return max(left.items(), key=lambda kv: kv[1])[0], max(right.items(), key=lambda kv: kv[1])[0]


def build_submission(out_path: Path, track2_csv: Path | None = None) -> None:
    t1 = pd.read_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet")

    # Pivot: one row per patient with L1..L17, R1..R17, Total.
    by_side = {}
    for side in ("L", "R"):
        sub = t1[t1["side"] == side].set_index("patient_id")
        by_side[side] = sub
    patient_ids = sorted(set(by_side["L"].index) | set(by_side["R"].index))

    track1_pred = {}
    for pid in patient_ids:
        rec = {}
        for it in range(1, 18):
            rec[f"L{it}"] = int(by_side["L"].loc[pid, f"pred_{it}"]) if pid in by_side["L"].index else 0
            rec[f"R{it}"] = int(by_side["R"].loc[pid, f"pred_{it}"]) if pid in by_side["R"].index else 0
        # Total invariant: Total = sum(L_items) + sum(R_items). Matches ground-truth definition.
        rec["Total"] = int(sum(rec[f"L{i}"] for i in range(1, 18)) + sum(rec[f"R{i}"] for i in range(1, 18)))
        track1_pred[pid] = rec

    if track2_csv is not None and track2_csv.exists():
        t2_df = pd.read_csv(track2_csv)
        track2_pred = {int(r["patient_id"]): (r["left_subtype"], r["right_subtype"]) for _, r in t2_df.iterrows()}
    else:
        left_mode, right_mode = _track2_fallback()
        track2_pred = {pid: (left_mode, right_mode) for pid in cfg.TRACK2_TEST_IDS}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for pid in sorted(cfg.TRACK1_TEST_IDS):
            r = track1_pred.get(pid, {f"L{i}": 0 for i in range(1, 18)} | {f"R{i}": 0 for i in range(1, 18)} | {"Total": 0})
            row = [f"track1-{pid}"]
            row += [r[f"L{i}"] for i in range(1, 18)]
            row += [r[f"R{i}"] for i in range(1, 18)]
            row += [r["Total"], -1, -1]
            w.writerow(row)
        for pid in sorted(cfg.TRACK2_TEST_IDS):
            lsub, rsub = track2_pred.get(pid, _track2_fallback())
            row = [f"track2-{pid}"]
            row += [-1] * 17
            row += [-1] * 17
            row += [-1, lsub, rsub]
            w.writerow(row)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(cfg.SUBMIT_DIR / "v1_track1.csv"))
    ap.add_argument("--track2-csv", default=None, help="Optional CSV with columns patient_id,left_subtype,right_subtype")
    args = ap.parse_args()
    build_submission(Path(args.out), Path(args.track2_csv) if args.track2_csv else None)


if __name__ == "__main__":
    main()
