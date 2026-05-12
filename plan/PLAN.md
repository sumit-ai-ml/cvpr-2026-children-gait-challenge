# CVPR 2026 Children Gait Challenge — Execution Plan

**Owner:** sumitpandey171@gmail.com
**Submission deadline:** ~2026-05-16 (4 days from 2026-05-12)
**Report deadline:** 2026-05-15
**Final score:** 0.5 × Track1 + 0.5 × Track2

## Problem in one paragraph
Predict 17 binary EVGS items per limb (Track 1, 16 test patients) and 5-class CP gait subtype per limb (Track 2, 9 test patients) from per-frame COCO-WholeBody 133-keypoint sequences over 4 camera views (forward/backward/left/right). No raw video. 110 patients total; Track 1 has 94 train, Track 2 only 22 train, and 17 patients overlap both train tracks.

## Strategic anchor — the EVGS bridge
17 patients have BOTH Track 1 and Track 2 labels. We exploit this:
1. Train a Track 1 model (EVGS prediction) using 94 train patients.
2. Generate EVGS predictions for ALL 110 patients (including the 22 Track 2 train).
3. Train Track 2 using `[kinematic features] ⊕ [predicted EVGS vector]` as input.
4. At test time: predict EVGS first; for the 3 patients in both test sets (4, 26, 42) the EVGS prediction feeds Track 2.

This converts an N=22 problem into an N=22 problem with much stronger features.

## Data flow

```
Dataset/dataset/{patient}/{clip}/frame_*.json   (339k frames)
        │
        ▼
┌──────────────────────────────────────────────┐
│ STAGE 1 — Per-clip feature extraction        │
│  - keep body+feet kpts (0..22), drop face/hand│
│  - confidence mask (score >= 0.2)            │
│  - per-frame: normalize by bbox + hip center │
│  - Savitzky-Golay smooth (window=11, ord=3)  │
│  - gait event detection (heel-strike, toe-off)│
│  - per-cycle joint angles → mean/std/min/max │
│    over cycles, plus at IC / mid-stance / TO │
│  - per-clip vector ~300 dims                 │
│  → cache: cache/features_clip.parquet         │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│ STAGE 2 — Per-patient × per-limb pooling     │
│  - aggregate clips by view (mean + std)      │
│  - one row per (patient_id, limb)            │
│  → cache: cache/features_patient_limb.parquet │
└──────────────────────────────────────────────┘
        │
        ├─────────────────────┐
        ▼                     ▼
┌─────────────────┐  ┌──────────────────────────┐
│ TRACK 1 MODEL   │  │ TRACK 2 MODEL            │
│  17 × 2 = 34    │  │  Input = features ⊕      │
│  LightGBM       │  │   EVGS-pred-vector (17d) │
│  per-item bin   │  │  Ensemble:               │
│  classifiers    │  │   - LightGBM (cls-weight)│
│  + Total reg    │  │   - kNN (k=3 cosine)     │
│  CV: 5-fold by  │  │   - Logistic-Multinomial │
│   patient       │  │   - Heuristic from EVGS  │
│                 │  │  CV: leave-1-pat-out     │
└─────────────────┘  └──────────────────────────┘
        │                     │
        └──────────┬──────────┘
                   ▼
        submission.csv (track1 rows then track2 rows)
```

## Timeline

| Day | Deliverable | Validation |
|-----|-------------|------------|
| 1 | Feature pipeline (stages 1+2) on all 110 patients, cached parquet | Spot-check 3 patients: trajectories smooth, angles in [0,180°], gait cycles ≈ 2 per clip |
| 2 | Track 1 models, 5-fold CV, per-item threshold tuning, Total regressor | CV per-item accuracy reported, Total RMSE reported, sanity: sum(items) ≈ predicted Total |
| 3 | Track 2 models incl. EVGS-bridge ablation, ensemble weights | Leave-1-pat-out macro-F1 per class; ensure WNL+type4 are not collapsed |
| 4 | Submission CSV, public repo, 4-page report | CSV passes schema check; repo runs end-to-end with one command |

## Repository layout (planned)

```
.
├── Dataset/                       # provided, not in repo
├── cache/                         # generated artifacts (gitignored)
│   ├── features_clip.parquet
│   └── features_patient_limb.parquet
├── src/
│   ├── config.py                  # all knobs in one place
│   ├── data_io.py                 # load frame JSON, label JSON
│   ├── kinematics.py              # joint angles, normalization
│   ├── gait_events.py             # heel-strike / toe-off detection
│   ├── features.py                # per-clip + per-patient feature builders
│   ├── track1_model.py            # train + predict EVGS
│   ├── track2_model.py            # train + predict subtype
│   ├── ensemble.py                # late fusion across views and models
│   ├── submit.py                  # build final CSV
│   └── cv.py                      # patient-stratified CV helpers
├── scripts/
│   ├── build_features.py
│   ├── train_track1.py
│   ├── train_track2.py
│   └── make_submission.py
├── notebooks/
│   └── eda.ipynb                  # one notebook of sanity checks
├── tests/
│   ├── test_kinematics.py         # angle math on known vectors
│   ├── test_gait_events.py        # synthetic sinusoid → known events
│   ├── test_features.py           # shape + no-NaN invariants
│   └── test_submit.py             # CSV schema compliance
├── README.md
├── requirements.txt
└── PLAN.md                        # this file
```

## NOT in scope (deferred / cut)
- Training a 3D pose estimator from 2D keypoints.
- Deep temporal models from scratch (ST-GCN, MotionBERT). Considered as stretch ensemble member only on day 4 if CV scores warrant it.
- Self-supervised pretraining on keypoints.
- Hyperparameter search via Optuna. Use sensible defaults; budget 1 hr at most for tuning.
- Test-time augmentation beyond multi-view averaging.

## Decisions still open (to resolve in this review)
- Track 2 model class: pure handcrafted ensemble vs adding lightweight temporal net?
- Multi-view fusion strategy: late (per-view models) vs early (concatenated features)?
- Whether to add a clinical-heuristic class-by-class scoring for rare Track 2 classes (WNL, type4).
