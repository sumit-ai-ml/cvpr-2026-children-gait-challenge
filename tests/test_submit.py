"""IRON RULE test — the submission CSV schema MUST match the Kaggle spec exactly.

A drift here = a 0 score on submission. If this test ever fails, do not submit.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACK1_TEST_IDS = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
TRACK2_TEST_IDS = [4, 6, 7, 13, 26, 35, 39, 42, 50]

EXPECTED_HEADER = (
    "ID,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15,L16,L17,"
    "R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,R14,R15,R16,R17,"
    "Total,Left_gait_subtype,Right_gait_subtype"
)
VALID_SUBTYPES = {"type1", "type2", "type3", "type4", "WNL"}


def validate_submission(path: Path) -> None:
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        assert ",".join(header) == EXPECTED_HEADER, f"header mismatch:\n  got: {','.join(header)}\n want: {EXPECTED_HEADER}"
        rows = list(reader)

    # Row count
    expected_rows = len(TRACK1_TEST_IDS) + len(TRACK2_TEST_IDS)
    assert len(rows) == expected_rows, f"row count {len(rows)} != {expected_rows}"

    # Row order: track1 ascending, then track2 ascending
    expected_ids = [f"track1-{i}" for i in sorted(TRACK1_TEST_IDS)] + [
        f"track2-{i}" for i in sorted(TRACK2_TEST_IDS)
    ]
    actual_ids = [r[0] for r in rows]
    assert actual_ids == expected_ids, f"id order mismatch:\n  got: {actual_ids}\n want: {expected_ids}"

    # Track 1 rows
    for r in rows[: len(TRACK1_TEST_IDS)]:
        items = r[1:35]
        total = r[35]
        subs = r[36:38]
        for v in items:
            assert v in {"0", "1"}, f"track1 item must be 0/1, got {v!r}"
        assert total.lstrip("-").isdigit(), f"track1 Total must be int, got {total!r}"
        assert 0 <= int(total) <= 34, f"track1 Total {total} out of range"
        # sum-equals-Total invariant
        assert int(total) == sum(int(v) for v in items), f"track1 Total {total} != sum(items) {sum(int(v) for v in items)}"
        for v in subs:
            assert v == "-1", f"track1 subtype col must be -1, got {v!r}"

    # Track 2 rows
    for r in rows[len(TRACK1_TEST_IDS):]:
        items = r[1:35]
        total = r[35]
        subs = r[36:38]
        for v in items:
            assert v == "-1", f"track2 item col must be -1, got {v!r}"
        assert total == "-1", f"track2 Total must be -1, got {total!r}"
        for v in subs:
            assert v in VALID_SUBTYPES, f"track2 subtype must be in {VALID_SUBTYPES}, got {v!r}"


def test_v0_baseline_schema() -> None:
    validate_submission(ROOT / "submissions" / "v0_baseline.csv")


def test_v1_track1_schema() -> None:
    path = ROOT / "submissions" / "v1_track1.csv"
    if path.exists():
        validate_submission(path)


if __name__ == "__main__":
    for p in (ROOT / "submissions").glob("*.csv"):
        validate_submission(p)
        print(f"OK — {p.name}")
