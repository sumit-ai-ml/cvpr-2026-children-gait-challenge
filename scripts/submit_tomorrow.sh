#!/usr/bin/env bash
# Run this tomorrow (after ~11:16 UTC) when Kaggle's daily quota refreshes.
# Submits v10 (Steps 1-3 pipeline) and waits for the score.
# If v10 lifts: also submits v11 (hybrid: new Track 1 + v8 Track 2) as a comparison.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Submitting v10 (Steps 1-3 pipeline) ==="
kaggle competitions submit \
  -c cvpr-2026-the-first-ai-children-challenge \
  -f submissions/v10_steps_1_3.csv \
  -m "v10: 35-dim EVGS bridge + codified WNL recovery + bilateral consistency rules. OOF S2=0.575."

echo
echo "Waiting for v10 score..."
until kaggle competitions submissions -c cvpr-2026-the-first-ai-children-challenge 2>&1 | head -3 | tail -1 | grep -q -v PENDING; do
  sleep 10
done

V10_SCORE=$(kaggle competitions submissions -c cvpr-2026-the-first-ai-children-challenge 2>&1 | head -3 | tail -1 | awk '{print $(NF-1)}')
echo "v10 score: $V10_SCORE"

echo
echo "=== Also submitting v11 (hybrid: new Track 1 + v8 Track 2) ==="
kaggle competitions submit \
  -c cvpr-2026-the-first-ai-children-challenge \
  -f submissions/v11_hybrid.csv \
  -m "v11: new pipeline Track 1 + v8's exact Track 2 predictions. Isolation test of Track 1 retraining."

echo
echo "Waiting for v11 score..."
until kaggle competitions submissions -c cvpr-2026-the-first-ai-children-challenge 2>&1 | head -3 | tail -1 | grep -q -v PENDING; do
  sleep 10
done

echo
echo "=== Final leaderboard ==="
kaggle competitions submissions -c cvpr-2026-the-first-ai-children-challenge 2>&1 | head -5
echo
echo "Best score (Kaggle keeps it):"
kaggle competitions leaderboard -c cvpr-2026-the-first-ai-children-challenge --show 2>&1 | grep "sumit pandey" | head -1
