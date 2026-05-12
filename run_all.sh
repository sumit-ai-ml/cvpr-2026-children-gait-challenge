#!/usr/bin/env bash
# End-to-end reproduction of the final submission (Kaggle public score: 0.63903).
# Assumes Dataset/ is in place and Python deps are installed (see requirements.txt).
#
# Wall-clock time on RTX A3000 + 16 cores: ~5 minutes.

set -euo pipefail
cd "$(dirname "$0")"

echo "==> 1/4 Build kinematic features (1185 clips × 12 workers)"
python scripts/build_features.py --workers 12

echo
echo "==> 2/4 Train Track 1 (3-tree GBM ensemble + S_1 joint threshold tuning)"
python scripts/train_track1.py

echo
echo "==> 3/4 Train Track 2 (LGBM + EVGS-bridge with semi-supervised pseudo-labels)"
python scripts/track2_finalize_with_pseudo.py

echo
echo "==> 4/4 Build final submission CSV with submission-time corrections"
python scripts/build_final_submission.py --out submissions/final.csv

echo
echo "==> Validate"
python -m pytest tests/test_submit.py -q
echo
echo "Done. Output: submissions/final.csv"
echo "Expected Kaggle public score: ~0.639 (may vary slightly due to CV randomness)."
