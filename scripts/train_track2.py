"""Train Track 2 ensemble end-to-end."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.track2_model import train_and_predict


def main() -> None:
    summary = train_and_predict()
    print()
    print("== TRACK 2 OOF SCORE ==")
    print(f"  Acc:       {summary['oof_acc']:.4f}")
    print(f"  Macro-F1:  {summary['oof_macro_f1']:.4f}")
    print(f"  S_2:       {summary['oof_s2']:.4f}")
    print(f"  weights:   {summary['ensemble_weights']}")


if __name__ == "__main__":
    main()
