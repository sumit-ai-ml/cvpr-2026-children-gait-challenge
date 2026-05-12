"""Day-0 base-rate baseline submission.

Track 1: per-item majority class from train labels (binarized 0/1).
Track 2: modal subtype from train labels.
Output: submissions/v0_baseline.csv matching the exact spec schema.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset"
OUT = ROOT / "submissions" / "v0_baseline.csv"

TRACK1_TEST_IDS = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
TRACK2_TEST_IDS = [4, 6, 7, 13, 26, 35, 39, 42, 50]

HEADER = (
    ["ID"]
    + [f"L{i}" for i in range(1, 18)]
    + [f"R{i}" for i in range(1, 18)]
    + ["Total", "Left_gait_subtype", "Right_gait_subtype"]
)


def compute_track1_majority(track1_train: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    left = {str(i): Counter() for i in range(1, 18)}
    right = {str(i): Counter() for i in range(1, 18)}
    for p in track1_train:
        for k in left:
            left[k][int(p["left"][k])] += 1
            right[k][int(p["right"][k])] += 1
    left_maj = {k: max(c.items(), key=lambda kv: kv[1])[0] for k, c in left.items()}
    right_maj = {k: max(c.items(), key=lambda kv: kv[1])[0] for k, c in right.items()}
    return left_maj, right_maj


def compute_track2_modal(track2_train: list[dict]) -> tuple[str, str]:
    left = Counter(p["left"]["gait_subtype"] for p in track2_train)
    right = Counter(p["right"]["gait_subtype"] for p in track2_train)
    return max(left.items(), key=lambda kv: kv[1])[0], max(right.items(), key=lambda kv: kv[1])[0]


def main() -> None:
    track1_train = json.loads((DATA / "track1_train.json").read_text())
    track2_train = json.loads((DATA / "track2_train.json").read_text())

    left_maj, right_maj = compute_track1_majority(track1_train)
    left_mode, right_mode = compute_track2_modal(track2_train)

    print("Track 1 per-item majority (Left):", left_maj)
    print("Track 1 per-item majority (Right):", right_maj)
    print(f"Track 2 modal: left={left_mode} right={right_mode}")

    t1_total = sum(left_maj.values()) + sum(right_maj.values())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for pid in sorted(TRACK1_TEST_IDS):
            row = [f"track1-{pid}"]
            row += [left_maj[str(i)] for i in range(1, 18)]
            row += [right_maj[str(i)] for i in range(1, 18)]
            row += [t1_total, -1, -1]
            w.writerow(row)
        for pid in sorted(TRACK2_TEST_IDS):
            row = [f"track2-{pid}"]
            row += [-1] * 17  # L1..L17
            row += [-1] * 17  # R1..R17
            row += [-1, left_mode, right_mode]
            w.writerow(row)

    print(f"Wrote {OUT} with {len(TRACK1_TEST_IDS) + len(TRACK2_TEST_IDS)} rows.")


if __name__ == "__main__":
    main()
