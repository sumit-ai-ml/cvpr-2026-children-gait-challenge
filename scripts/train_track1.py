"""Train Track 1 ensemble end-to-end and dump caches."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.track1_model import train_and_predict


def main() -> None:
    summary = train_and_predict()
    print()
    print("== TRACK 1 OOF SCORE ==")
    print(f"  Acc:   {summary['oof_acc']:.4f}")
    print(f"  NRMSE: {summary['oof_nrmse']:.4f}")
    print(f"  S_1:   {summary['oof_s1']:.4f}")
    (ROOT / "cache" / "track1_summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
