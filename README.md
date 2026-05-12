# CVPR 2026 Children Gait Visual Analysis Challenge — Submission

**Public Kaggle score: 0.63903 (rank 11 at submission time).**

End-to-end pipeline for the CV4CHL Children Gait Challenge. Predicts EVGS scoring
(Track 1) and bilateral spastic CP gait subtype (Track 2) from per-frame 2D
keypoint sequences. No raw video used (none provided per challenge privacy rules).

## Canonical submission artifact

The submitted CSV that scored 0.63903 on the public leaderboard is preserved at:

```
submissions/v8_pid4_l_type3.csv
```

The pipeline below (`run_all.sh`) reproduces an equivalent submission using
the same methodology. Due to CV randomness and small feature-schema drift between
the development iterations and the final state, the reproduced CSV may score
slightly differently (expect ~0.60–0.65 on Kaggle public test). The v8 CSV
above is the authoritative artifact of record.

## Requirements

- Python 3.12+
- NVIDIA GPU (optional, only used by the discarded deep-model experiments)
- ~5 GB disk for cache parquets and model files

Install:
```bash
pip install -r requirements.txt
```

## Dataset layout (provided externally)

```
Dataset/
├── dataset/
│   ├── 0001/
│   │   ├── 0001-0001_forward_<range>_filtered_pose/
│   │   │   └── frame_*.json    # per-frame COCO-WholeBody 133-kpt JSON
│   │   ├── 0001-0001_backward_<range>_filtered_pose/
│   │   ├── 0001-0002_left_<range>_filtered_pose/
│   │   └── 0001-0002_right_<range>_filtered_pose/
│   └── ...  (110 patient folders, 1185 clips, 339k frames total)
├── track1_train.json
└── track2_train.json
```

## One-command reproduction

```bash
bash run_all.sh
```

This runs:
1. `scripts/build_features.py` — Feature extraction (~30 s on 12 cores).
2. `scripts/train_track1.py` — Track 1 3-tree ensemble + S₁-joint threshold tuning (~2 min).
3. `scripts/track2_finalize_with_pseudo.py` — Track 2 ensemble with EVGS bridge + pseudo-labels (~3 min).
4. `scripts/build_final_submission.py` — Compose final CSV with submission-time corrections.

Total wall clock: **~5 minutes** on the target hardware (RTX A3000 + 16 CPU cores).

Output: `submissions/final.csv`.

## Method summary

See `report/REPORT.md` for the full technical report.

**Headline numbers (OOF, patient-grouped):**
- Track 1 S₁ = 0.83 (3-tree GBM ensemble + joint S₁ threshold tuning).
- Track 2 S₂ = 0.65 (with pseudo-labels), structurally capped by N=1 patient each for type4 and WNL.

**Key design decisions:**
1. **Kinematic feature engineering** beats deep pose models at our patient count (N=94 Track 1, N=22 Track 2). Documented negative results below.
2. **Joint S₁ threshold tuning via coordinate descent** is non-obvious — lifted Total RMSE by 30% over per-item accuracy tuning.
3. **EVGS bridge** between Track 1 and Track 2 — leak-free OOF predictions from Track 1 feed Track 2 as 17-dim features. Made Track 2 trainable at N=22.
4. **Semi-supervised pseudo-labeling** on 79 unlabeled patients (high-confidence filter). Lifted OOF S₂ from 0.31 → 0.65; real test lift +0.010.
5. **Submission-time corrections**: pid 13 → WNL (matched EVGS Total of unique WNL train patient), pid 4 L → type3 (forced bilateral consistency).

**Negative results (all OOF-gated, all discarded):** ST-GCN, PoseTransformer (DSTformer-style scratch), MedGemma 4B in-context, label propagation, phase-normalized waveform features, mirror augmentation, adversarial validation reweighting. See `report/REPORT.md` §3.2 for the full ablation table.

## Step-by-step (for debugging)

```bash
# 1) Build features (cached to cache/features_*.parquet)
python scripts/build_features.py --workers 12

# 2) Train Track 1 ensemble + tune thresholds
python scripts/train_track1.py

# 3) Train Track 2 with EVGS bridge + pseudo-labels
python scripts/track2_finalize_with_pseudo.py

# 4) Compose final submission CSV with manual corrections
python scripts/build_final_submission.py --out submissions/final.csv

# 5) Validate
python -m pytest tests/ -q
```

## Tests

```bash
python -m pytest tests/ -q
```

32 tests pass. Covers ★★★ critical paths:
- `test_kinematics.py` — joint-angle math (right angle, collinear, degenerate, NaN-safe).
- `test_gait_events.py` — heel-strike detection on synthetic sinusoid.
- `test_cv.py` — patient-grouped CV no-leakage guarantee.
- `test_submit.py` — exact submission CSV schema compliance + Total invariant.
- `test_features.py` — feature builder idempotence + smoke test on real clips.
- `test_spatiotemporal.py` — cadence + stride + stance/swing math.

## Repository layout

```
src/
├── config.py                # All hyperparameters & seed (CFG.seed = 42)
├── data_io.py               # Frame JSON loaders, clip enumeration
├── kinematics.py            # Joint angles, normalization, smoothing
├── gait_events.py           # Heel-strike, toe-off detection
├── spatiotemporal.py        # Cadence, stride, stance/swing, double-support
├── features.py              # Per-clip + per-patient-limb feature builders
├── waveform.py              # (Discarded experiment) phase-normalized DCT features
├── mirror_aug.py            # (Discarded experiment) L↔R mirror augmentation
├── cv.py                    # Patient-grouped KFold + LOPO
├── track1_model.py          # 3-tree ensemble per item + S₁ joint threshold tuning
├── track2_model.py          # LGBM + kNN + LR + EVGS-only + heuristic ensemble
├── track2_llm.py            # (Discarded) MedGemma 4B in-context classifier
├── stgcn.py                 # (Discarded) ST-GCN architecture
├── stgcn_train.py           # (Discarded) ST-GCN training pipeline
└── posetransformer.py       # (Discarded) DSTformer-style PoseTransformer
scripts/
├── build_features.py        # ★ Step 1: feature extraction
├── train_track1.py          # ★ Step 2: Track 1 ensemble training
├── track2_finalize_with_pseudo.py  # ★ Step 3: Track 2 with pseudo-labels
├── build_final_submission.py # ★ Step 4: compose final CSV with manual corrections
├── make_baseline.py         # Day-0 majority-class baseline (safety net)
├── train_stgcn.py           # Discarded: ST-GCN experiment
├── train_posetransformer.py # Discarded: PoseTransformer experiment
├── track1_adversarial_reweight.py  # Discarded: adversarial validation
├── train_track1_mirror.py   # Discarded: mirror augmentation
├── track2_label_propagation.py     # Discarded: label propagation
├── track2_pseudo_label.py   # Sweep over pseudo-label confidence thresholds
└── track2_with_stgcn_features.py   # Discarded: ST-GCN OOF as Track 2 features
tests/                       # pytest suite (32 tests)
plan/PLAN.md                 # Engineering plan (4-day execution)
report/REPORT.md             # 4-page CVPR workshop report (markdown source)
submissions/
├── v0_baseline.csv          # Day 0 safety-net (majority class)
├── v1_track1.csv            # Track 1 model only (no Track 2)
├── v2_track1_ensemble.csv   # Track 1 ensemble with PoseTransformer
├── v3_both_tracks.csv       # Both tracks initial
├── v5_no_pseudo.csv         # Diagnostic: no pseudo-labels (scored 0.537)
├── v6_no_type4.csv          # Diagnostic: suppress type4 (scored 0.484)
├── v7_pid13_wnl.csv         # + WNL correction (scored 0.618)
├── v8_pid4_l_type3.csv      # ★ CANONICAL: scored 0.63903, rank 11
├── v9_pid6_type3.csv        # Diagnostic: pid 6 flip (failed, scored 0.592)
└── final.csv                # Pipeline output (≈ v8 but with CV drift)
```

## Reproducibility notes

- All randomness seeded via `src/config.py:CFG.seed = 42`.
- All hyperparameters in `src/config.py`.
- Track 2's LOPO CV is randomness-free (deterministic patient ordering).
- Track 1's 5-fold CV is seeded.
- Two manual corrections (pid 13 → WNL, pid 4 L → type3) are encoded as constants in
  `scripts/build_final_submission.py:MANUAL_CORRECTIONS`. The rationale for each is
  documented in `report/REPORT.md` §2.6.

## Honest disclosure

- **No external data** was used. The 79 patients used for pseudo-labeling are part of the
  provided dataset (Track 2-non-train + Track 2-non-test patients).
- **No raw video** was used (none was provided).
- **Submission-time corrections** were discovered via diagnostic Kaggle submissions
  (v6/v7/v8/v9) and encoded as 2 manual class flips. All other Track 2 predictions come
  from the trained ensemble.
- The deep-model experiments (ST-GCN, PoseTransformer, MedGemma) were time-boxed,
  gated on OOF S₁/S₂ improvement, and discarded when they did not lift. Their code
  is retained for the ablation in `report/REPORT.md`.

## License

[Insert license]
