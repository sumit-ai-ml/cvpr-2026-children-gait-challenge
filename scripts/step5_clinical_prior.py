"""Step 5: Clinical prior blending.

Build a 5×17 matrix P(item=1 | subtype) from Rodda 2004 / Toro 2010 clinical descriptions.
For each test patient, compute literature posterior L(subtype) = Π P(item=p_i | subtype)
over predicted item probabilities p_i. Normalize to a 5-vector posterior.

Blend with hierarchical-head's posterior at weight 0.25 (prior) / 0.75 (model).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.evgs_bridge import EVGS_ITEMS, bridge_feature_cols, build_bridge_table, load_thresholds
from src.track2_model import CLASSES, build_track2_dataset

# Clinical prior P(item=1 | subtype) — values from Rodda 2004, Toro 2010, EVGS guide.
# Approximate per literature: 0.85 = characteristic, 0.55 = sometimes, 0.20 = uncharacteristic.
# Item descriptions:
#   1=trunk lateral lean(coronal), 2=pelvic obliquity(coronal), 3=hip coronal, 4=knee coronal
#   5=hindfoot varus/valgus, 6=initial foot contact, 7=heel lift midstance, 8=max ankle DF in stance
#   9=foot rotation in stance, 10=foot clearance in swing, 11=max ankle DF in swing
#   12=knee in midswing, 13=knee in terminal swing, 14=max knee extension in stance
#   15=peak hip extension in stance, 16=pelvic rotation, 17=trunk rotation(sagittal)
# Rows: type1 (True Equinus), type2 (Jump), type3 (Apparent Equinus), type4 (Crouch), WNL
CLINICAL_PRIOR = np.array([
    # 1   2     3     4     5     6     7     8     9     10    11    12    13    14    15    16    17
    [0.55,0.55, 0.55, 0.55, 0.55, 0.85, 0.85, 0.20, 0.55, 0.55, 0.20, 0.20, 0.20, 0.55, 0.55, 0.20, 0.20],  # type1 True Equinus
    [0.55,0.55, 0.55, 0.55, 0.55, 0.85, 0.85, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.85, 0.85, 0.20, 0.20],  # type2 Jump Gait
    [0.55,0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.85, 0.85, 0.20, 0.20],  # type3 Apparent Equinus
    [0.55,0.55, 0.55, 0.85, 0.55, 0.20, 0.55, 0.85, 0.55, 0.55, 0.85, 0.85, 0.85, 0.85, 0.85, 0.55, 0.55],  # type4 Crouch
    [0.20,0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],  # WNL
])


def posterior_from_prior(item_probs: np.ndarray) -> np.ndarray:
    """item_probs: (N, 17) raw EVGS item probabilities. Returns (N, 5) normalized prior posterior.

    L(subtype) = Π_k P(item_k = p_k | subtype) approximated as
                 = Π_k [p_k * P(=1|s) + (1-p_k) * (1 - P(=1|s))]
    Take log to avoid underflow.
    """
    p = np.clip(item_probs, 1e-3, 1 - 1e-3)  # (N, 17)
    log_post = np.zeros((p.shape[0], 5))
    for s in range(5):
        prior_s = CLINICAL_PRIOR[s]  # (17,)
        # likelihood per item: p_k * prior + (1-p_k) * (1-prior)
        lik = p * prior_s + (1 - p) * (1 - prior_s)
        lik = np.clip(lik, 1e-6, 1.0)
        log_post[:, s] = np.log(lik).sum(axis=1)
    # softmax to normalize
    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    post /= post.sum(axis=1, keepdims=True)
    return post


def main(blend_weight_prior: float = 0.25) -> None:
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    WNL_IDX = label_to_idx["WNL"]

    # Load current Step 3 predictions
    cur = pd.read_parquet(cfg.CACHE_DIR / "track2_preds.parquet")
    prob_cols = [f"prob_{c}" for c in CLASSES]

    # Build bridge for raw item probabilities
    bridge_df = build_bridge_table(use_oof_for_train=True)
    bridge_indexed = bridge_df.set_index(["patient_id", "side"])

    # For each row in cur, compute prior posterior using raw item probs
    rows = []
    train_df, _, _ = build_track2_dataset()
    train_indexed = train_df.set_index(["patient_id", "side"])["y"].to_dict()

    for _, r in cur.iterrows():
        pid = int(r.patient_id); side = r.side
        bridge_row = bridge_indexed.loc[(pid, side)]
        raw_probs = np.array([float(bridge_row[f"bridge_p_{it}"]) for it in EVGS_ITEMS])
        prior_post = posterior_from_prior(raw_probs.reshape(1, -1))[0]
        rec = {"patient_id": pid, "side": side}
        for ci, c in enumerate(CLASSES):
            rec[f"prior_prob_{c}"] = float(prior_post[ci])
            rec[f"model_prob_{c}"] = float(r[f"prob_{c}"])
        rows.append(rec)
    posterior_df = pd.DataFrame(rows)

    # Try multiple blend weights, evaluate on training patients via LOPO-equivalent (just blend on current OOF preds).
    # For simplicity here: report S2 on training patients using both prior alone, model alone, and blends.
    train_pids_set = set(train_indexed.keys())
    train_mask_post = posterior_df.apply(
        lambda r: (int(r.patient_id), r.side) in train_indexed, axis=1
    )
    train_post = posterior_df[train_mask_post].reset_index(drop=True)

    y_train = np.array([
        label_to_idx[train_indexed[(int(r.patient_id), r.side)]]
        for _, r in train_post.iterrows()
    ])

    def score(preds):
        acc = accuracy_score(y_train, preds)
        f1 = f1_score(y_train, preds, labels=list(range(5)), average="macro", zero_division=0)
        return acc, f1, (acc + f1) / 2

    model_probs = train_post[[f"model_prob_{c}" for c in CLASSES]].to_numpy()
    prior_probs = train_post[[f"prior_prob_{c}" for c in CLASSES]].to_numpy()

    print("Train-set scores (NOT OOF — purely training-fit):")
    for w_prior in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        blend = (1 - w_prior) * model_probs + w_prior * prior_probs
        preds = blend.argmax(axis=1)
        acc, f1, s2 = score(preds)
        print(f"  w_prior={w_prior:.2f}: Acc={acc:.4f} F1={f1:.4f} S2={s2:.4f}")
    print()
    print("These are training-fit numbers and overfit. Real OOF would be lower.")
    print(f"Step 3 OOF baseline: 0.5753 — blending may or may not transfer.")

    # Apply blend at requested weight to ALL 220 predictions
    print()
    print(f"Applying blend at w_prior={blend_weight_prior} to full prediction set...")
    full_blend = (1 - blend_weight_prior) * posterior_df[[f"model_prob_{c}" for c in CLASSES]].to_numpy() \
                 + blend_weight_prior * posterior_df[[f"prior_prob_{c}" for c in CLASSES]].to_numpy()
    pred_idx = full_blend.argmax(axis=1)

    out_df = pd.DataFrame({
        "patient_id": posterior_df["patient_id"].values,
        "side": posterior_df["side"].values,
        "subtype": [idx_to_label[int(i)] for i in pred_idx],
    })
    for ci, cn in enumerate(CLASSES):
        out_df[f"prob_{cn}"] = full_blend[:, ci]

    # Show test predictions
    print("\nTest predictions with prior blending:")
    for pid in cfg.TRACK2_TEST_IDS:
        row_L = out_df[(out_df.patient_id == pid) & (out_df.side == "L")]
        row_R = out_df[(out_df.patient_id == pid) & (out_df.side == "R")]
        if len(row_L) and len(row_R):
            print(f"  pid={pid:>3}: L={row_L.iloc[0]['subtype']:<6} R={row_R.iloc[0]['subtype']:<6}")

    out_df.to_parquet(cfg.CACHE_DIR / "track2_preds_step5.parquet", index=False)
    (cfg.CACHE_DIR / "step5_summary.json").write_text(json.dumps({
        "blend_weight_prior": blend_weight_prior,
        "note": "Train-fit scores reported; OOF was not computed because blending uses full-trained model. Submit to Kaggle to evaluate.",
    }, indent=2))
    print(f"\nSaved cache/track2_preds_step5.parquet (NOT cache/track2_preds.parquet — manual decision needed)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--w-prior", type=float, default=0.25)
    args = p.parse_args()
    main(blend_weight_prior=args.w_prior)
