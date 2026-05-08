"""Compare multiple 3-lead wearable montages against the full 12-lead model.

Evaluated montages:
  - V1/V2/V3            : data-driven, top of the lead-importance ranking
  - V1/II/V5            : clinical 3-channel Holter (MV1 + II + MV5)
  - Per-class consensus : top-3 leads by per-class occlusion votes across the test set
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             recall_score)

from data_processing.data_generator import DataGenerator
from utils.utils import input_length, num_channels, num_classes, model_results_path
from explainability.lead_importance import (build_tuned_cnn_transformer,
                                            CHECKPOINT_PATH, LEAD_NAMES)

OUTPUT_DIR = os.path.join(model_results_path, "lead_importance")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MONTAGES = {
    "12-lead (full)": list(range(12)),
    "V1/V2/V3 (data-driven)": [LEAD_NAMES.index(l) for l in ["V1", "V2", "V3"]],
    "V1/II/V5 (clinical Holter)": [LEAD_NAMES.index(l) for l in ["V1", "II", "V5"]],
}


def mask_to_subset(signals, keep_indices):
    masked = np.zeros_like(signals)
    masked[:, :, keep_indices] = signals[:, :, keep_indices]
    return masked


def compute_consensus_top3():
    """Pick the 3 leads that appear most in per-class occlusion top-3s.

    Reuses the precomputed occlusion ranking from lead_importance_summary.txt
    if present; otherwise falls back to the V1/V2/V3 set.
    """
    summary_path = os.path.join(OUTPUT_DIR, "lead_importance_summary.txt")
    if not os.path.exists(summary_path):
        return [LEAD_NAMES.index(l) for l in ["V1", "V2", "V3"]], ["V1", "V2", "V3"]

    counts = {l: 0 for l in LEAD_NAMES}
    with open(summary_path) as f:
        for line in f:
            if "occlusion:" in line:
                seg = line.split("occlusion:")[-1]
                for token in seg.replace("[", "").replace("]", "").replace("'", "").split(","):
                    name = token.strip()
                    if name in counts:
                        counts[name] += 1
    top3_names = sorted(counts, key=counts.get, reverse=True)[:3]
    return [LEAD_NAMES.index(l) for l in top3_names], top3_names


def main():
    with open("data/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    label_encoder = LabelEncoder()
    label_encoder.fit(test_data["classes"])
    class_names = list(label_encoder.classes_)

    consensus_idx, consensus_names = compute_consensus_top3()
    MONTAGES[f"{'/'.join(consensus_names)} (per-class consensus)"] = consensus_idx

    model = build_tuned_cnn_transformer((input_length, num_channels), num_classes)
    model.load_weights(CHECKPOINT_PATH)
    print(f"Loaded {CHECKPOINT_PATH}\n")

    test_gen = DataGenerator(test_data, label_encoder, shuffle=False, augment=False)
    all_signals, all_labels = [], []
    for x, y in test_gen:
        all_signals.append(x); all_labels.append(y)
    all_signals = np.concatenate(all_signals, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    rows = []
    per_class_f1 = {}
    for name, idx in MONTAGES.items():
        signals = all_signals if len(idx) == 12 else mask_to_subset(all_signals, idx)
        probs = model.predict(signals, batch_size=32, verbose=0)
        pred = probs.argmax(axis=1)

        acc = accuracy_score(all_labels, pred)
        macro_f1 = f1_score(all_labels, pred, average="macro", zero_division=0)
        macro_recall = recall_score(all_labels, pred, average="macro", zero_division=0)
        f1_per = f1_score(all_labels, pred, average=None, labels=range(num_classes),
                          zero_division=0)
        per_class_f1[name] = f1_per
        rows.append((name, acc, macro_recall, macro_f1))

    lines = []
    lines.append("MULTI-MONTAGE 3-LEAD COMPARISON")
    lines.append(f"Model: {CHECKPOINT_PATH}")
    lines.append("=" * 78)
    lines.append(f"\n{'Montage':<40} {'Acc':>8} {'Macro Rec':>10} {'Macro F1':>10}")
    lines.append("-" * 78)
    for name, acc, rec, f1 in rows:
        lines.append(f"{name:<40} {acc:>8.4f} {rec:>10.4f} {f1:>10.4f}")

    lines.append("\nPer-class F1 by montage:")
    header = f"{'Class':<8}" + "".join(f"{m[:18]:>20}" for m in MONTAGES)
    lines.append(header)
    lines.append("-" * len(header))
    for ci, c in enumerate(class_names):
        row = f"{c:<8}" + "".join(f"{per_class_f1[m][ci]:>20.3f}" for m in MONTAGES)
        lines.append(row)

    lines.append("\nDelta vs 12-lead (per-class F1):")
    base = per_class_f1["12-lead (full)"]
    other_montages = [m for m in MONTAGES if m != "12-lead (full)"]
    header = f"{'Class':<8}" + "".join(f"{m[:18]:>20}" for m in other_montages)
    lines.append(header)
    lines.append("-" * len(header))
    for ci, c in enumerate(class_names):
        row = f"{c:<8}" + "".join(
            f"{per_class_f1[m][ci] - base[ci]:>+20.3f}" for m in other_montages
        )
        lines.append(row)

    text = "\n".join(lines)
    out_path = os.path.join(OUTPUT_DIR, "montage_comparison.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
