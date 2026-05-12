"""Build clip-level and patient-limb feature tables for all patients.

Writes:
  cache/features_clip.parquet         — one row per clip
  cache/features_patient_limb.parquet — one row per (patient, side)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import features as feat
from src.data_io import ClipMeta, list_all_patient_ids, list_patient_clips


def _one_clip(clip: ClipMeta) -> dict | None:
    try:
        return feat.build_clip_features(clip)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {clip.path.name}: {e}", file=sys.stderr)
        return None


def main(n_workers: int = 8) -> None:
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_clips: list[ClipMeta] = []
    pids = list_all_patient_ids()
    for pid in pids:
        all_clips.extend(list_patient_clips(pid))
    print(f"Found {len(all_clips)} clips across {len(pids)} patients.")

    rows: list[dict] = []
    t0 = time.time()
    if n_workers <= 1:
        for c in all_clips:
            r = _one_clip(c)
            if r is not None:
                rows.append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_one_clip, c): c for c in all_clips}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r is not None:
                    rows.append(r)
                if i % 100 == 0 or i == len(all_clips):
                    print(f"  [{i}/{len(all_clips)}] clips done in {time.time()-t0:.1f}s")

    clip_df = pd.DataFrame(rows)
    print(f"Built clip table: {clip_df.shape}")
    clip_path = cfg.CACHE_DIR / "features_clip.parquet"
    clip_df.to_parquet(clip_path, index=False)
    print(f"  -> {clip_path}")

    pooled = feat.pool_to_patient_limb(clip_df)
    print(f"Pooled patient-limb table: {pooled.shape}")
    pool_path = cfg.CACHE_DIR / "features_patient_limb.parquet"
    pooled.to_parquet(pool_path, index=False)
    print(f"  -> {pool_path}")

    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    main(n_workers=args.workers)
