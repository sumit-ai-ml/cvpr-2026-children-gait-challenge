"""MedGemma in-context Track 2 classifier.

Encodes each patient's EVGS profile + a few key kinematic numbers as a textual
description, builds a few-shot prompt with all training examples, queries the
LLM, parses the class string.

Trade-offs explicit in the report:
- LLM has no native knowledge of the Rodda/Graham CP subtype scheme — we describe
  it in the system prompt.
- LOPO over 22 patients = 22 queries (~1 min total at 4-bit).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import config as cfg
from .data_io import load_track1_labels, load_track2_labels


MODEL_ID = "google/medgemma-1.5-4b-it"
CLASSES = ["type1", "type2", "type3", "type4", "WNL"]
EVGS_ITEMS = [str(i) for i in range(1, 18)]


# Item -> short clinical description, useful in the prompt
EVGS_ITEM_DESC = {
    "1": "trunk lean (coronal)",
    "2": "pelvic obliquity (coronal)",
    "3": "hip position (coronal)",
    "4": "knee position (coronal)",
    "5": "hindfoot varus/valgus",
    "6": "initial foot contact",
    "7": "heel lift in midstance",
    "8": "max ankle dorsiflexion in stance",
    "9": "foot rotation in stance",
    "10": "foot clearance in swing",
    "11": "max ankle dorsiflexion in swing",
    "12": "knee position in midswing",
    "13": "knee position in terminal swing",
    "14": "max knee extension in stance",
    "15": "peak hip extension in stance",
    "16": "pelvic rotation",
    "17": "trunk rotation (sagittal)",
}


SYSTEM_PROMPT = """You are a pediatric gait analysis assistant. You classify a child's gait pattern into one of five categories based on their Edinburgh Visual Gait Score (EVGS) profile.

The five gait pattern categories for Bilateral Spastic Cerebral Palsy (Rodda/Graham classification):
- type1 (True Equinus): sustained ankle plantarflexion throughout stance, knee extended.
- type2 (Jump Gait): equinus at initial contact, flexed hip and knee in early stance, knee extends in mid-stance.
- type3 (Apparent Equinus): ankle plantigrade but excessive hip and knee flexion makes the gait LOOK equinus.
- type4 (Crouch Gait): excessive hip and knee flexion throughout stance + ankle in dorsiflexion (not plantarflexion).
- WNL (within normal limits): does not clearly match the above 4 patterns.

Each EVGS item is graded 0 (normal) or 1 (deviation). 17 items are measured per limb:
""" + "\n".join(f"  {k}. {v}" for k, v in EVGS_ITEM_DESC.items()) + """

Read each patient's EVGS profile carefully, compare against the five patterns above, and output ONLY the subtype label (one of: type1, type2, type3, type4, WNL).
"""


def format_limb(items: list[int], total: int, kine: dict[str, float] | None = None) -> str:
    """Format one limb's EVGS profile + optional kinematic numbers."""
    positive = [f"{i+1}({EVGS_ITEM_DESC[str(i+1)][:30]})" for i, v in enumerate(items) if v == 1]
    pos_str = ", ".join(positive) if positive else "none"
    s = f"items_positive: [{pos_str}], total={total}/17"
    if kine:
        s += "; " + ", ".join(f"{k}={v:.2f}" for k, v in kine.items())
    return s


class MedGemmaClassifier:
    def __init__(self, model_id: str = MODEL_ID):
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="auto")
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def classify(self, examples: list[tuple[str, str]], query: str, max_new_tokens: int = 12) -> str:
        """examples = list of (patient_desc, label) pairs. Returns predicted class string."""
        body = "Training examples:\n"
        for desc, lab in examples:
            body += f"- {desc} -> {lab}\n"
        body += f"\nNow classify the following:\n- {query} -> "
        msgs = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + body},
        ]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None)
        text = self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # parse first matching class label in the text
        for cls in CLASSES:
            if re.search(rf"\b{re.escape(cls)}\b", text):
                return cls
        # Fallback: closest substring
        text_low = text.lower().strip()
        for cls in CLASSES:
            if cls.lower() in text_low:
                return cls
        return "type2"  # majority-class fallback


def build_examples_from_evgs(evgs_df: pd.DataFrame, t2_labels: dict[int, dict]) -> list[dict]:
    """Build per-limb records for Track 2 train patients with EVGS profile + label."""
    side_key = {"L": "left", "R": "right"}
    rows = []
    for _, row in evgs_df.iterrows():
        pid = int(row["patient_id"])
        side = row["side"]
        if pid not in t2_labels:
            continue
        # items: use a threshold of 0.5 on probabilities to get a binary item vector
        items = [int(row[f"oof_{it}"] >= 0.5) if f"oof_{it}" in row else int(row[f"prob_{it}"] >= 0.5) for it in EVGS_ITEMS]
        total = sum(items)
        label = t2_labels[pid][side_key[side]]["gait_subtype"]
        rows.append({
            "patient_id": pid, "side": side,
            "desc": f"patient_{pid} {side}-limb: {format_limb(items, total)}",
            "label": label,
            "items": items, "total": total,
        })
    return rows


def run_lopo() -> dict:
    """LOPO eval. Returns OOF predictions per (patient, side) and S_2 score."""
    pooled = pd.read_parquet(cfg.CACHE_DIR / "features_patient_limb.parquet")
    tree_oof = pd.read_parquet(cfg.CACHE_DIR / "track1_oof_train.parquet")
    tree_full = pd.read_parquet(cfg.CACHE_DIR / "track1_full_preds.parquet")
    t2_labels = load_track2_labels()

    # Build a merged EVGS table covering all 110 patients × 2 sides.
    # Prefer OOF for Track 1 train patients (leak-free).
    oof_map = {(int(r["patient_id"]), r["side"]): r.to_dict() for _, r in tree_oof.iterrows()}
    full_map = {(int(r["patient_id"]), r["side"]): r.to_dict() for _, r in tree_full.iterrows()}
    merged_rows = []
    for _, r in pooled.iterrows():
        key = (int(r["patient_id"]), r["side"])
        if key in oof_map and "oof_1" in oof_map[key]:
            evgs = {f"prob_{it}": oof_map[key][f"oof_{it}"] for it in EVGS_ITEMS}
        else:
            evgs = {f"prob_{it}": full_map[key][f"prob_{it}"] for it in EVGS_ITEMS}
        merged_rows.append({"patient_id": int(r["patient_id"]), "side": r["side"], **evgs})
    evgs_df = pd.DataFrame(merged_rows)

    train_examples = build_examples_from_evgs(evgs_df, t2_labels)
    print(f"Train examples (limbs): {len(train_examples)}")

    print("Loading MedGemma 4B (4-bit) ...")
    clf = MedGemmaClassifier()

    # LOPO over patients
    pids = sorted(set(r["patient_id"] for r in train_examples))
    print(f"LOPO over {len(pids)} patients (~1.5s per query) ...")
    oof_preds = {}
    for held_pid in pids:
        train_pool = [r for r in train_examples if r["patient_id"] != held_pid]
        test_limbs = [r for r in train_examples if r["patient_id"] == held_pid]
        ex_pairs = [(r["desc"], r["label"]) for r in train_pool]
        for q in test_limbs:
            pred = clf.classify(ex_pairs, q["desc"])
            oof_preds[(q["patient_id"], q["side"])] = pred

    # Score
    from sklearn.metrics import accuracy_score, f1_score
    y_true = [r["label"] for r in train_examples]
    y_pred = [oof_preds[(r["patient_id"], r["side"])] for r in train_examples]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    s2 = (acc + f1) / 2
    per_class = {c: f1_score(y_true, y_pred, labels=[c], average="macro", zero_division=0) for c in CLASSES}
    print(f"\n[MedGemma LOPO] Acc={acc:.4f}  Macro-F1={f1:.4f}  S_2={s2:.4f}")
    print(f"  per-class F1: {per_class}")
    print(f"  prediction distribution: {dict(Counter(y_pred))}")

    # Now full-train predict on the 9 Track 2 test patients (using all 22 train as examples).
    print("\nPredicting on Track 2 test patients ...")
    test_pids = cfg.TRACK2_TEST_IDS
    ex_pairs = [(r["desc"], r["label"]) for r in train_examples]
    test_preds = {}
    for tpid in test_pids:
        for side in ("L", "R"):
            row = evgs_df[(evgs_df.patient_id == tpid) & (evgs_df.side == side)].iloc[0]
            items = [int(row[f"prob_{it}"] >= 0.5) for it in EVGS_ITEMS]
            total = sum(items)
            q_desc = f"patient_{tpid} {side}-limb: {format_limb(items, total)}"
            pred = clf.classify(ex_pairs, q_desc)
            test_preds[(tpid, side)] = pred
            print(f"  pid={tpid} {side}: {pred}")

    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([
        {"patient_id": p, "side": s, "subtype": v} for (p, s), v in test_preds.items()
    ])
    out.to_parquet(cfg.CACHE_DIR / "track2_medgemma_test_preds.parquet", index=False)

    summary = {
        "oof_acc": float(acc), "oof_macro_f1": float(f1), "oof_s2": float(s2),
        "per_class_f1": per_class,
        "n_train_limbs": len(train_examples),
        "model": MODEL_ID,
    }
    (cfg.CACHE_DIR / "track2_medgemma_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run_lopo()
