"""Evaluate the lead-masked tuned CNN-Transformer with only V1/V2/V3 active.

Compares full 12-lead inference against a 3-lead wearable scenario where
leads other than V1, V2, V3 are zeroed at the input. The model was trained
with random lead masking (1/3/12 active leads), so this is exactly the
inference distribution it was prepared for -- no retraining needed.

Outputs:
  - 3lead_comparison.txt        accuracy + per-class precision/recall/F1
  - 3lead_confusion_matrix.png  confusion matrix for the 3-lead scenario
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

from data_processing.data_generator import DataGenerator
from utils.utils import input_length, num_channels, num_classes, model_results_path
from explainability.lead_importance import (build_tuned_cnn_transformer,
                                            CHECKPOINT_PATH, LEAD_NAMES)

WEARABLE_LEADS = ["V1", "V2", "V3"]
KEEP_INDICES = [LEAD_NAMES.index(l) for l in WEARABLE_LEADS]

OUTPUT_DIR = os.path.join(model_results_path, "lead_importance")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def mask_to_subset(signals, keep_indices):
    """Zero every lead not in keep_indices. signals: (N, time, leads)."""
    masked = np.zeros_like(signals)
    masked[:, :, keep_indices] = signals[:, :, keep_indices]
    return masked


def per_class_report(y_true, y_pred, class_names):
    rep = classification_report(y_true, y_pred, target_names=class_names,
                                digits=3, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return acc, rep


def plot_cm(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    with open("data/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    label_encoder = LabelEncoder()
    label_encoder.fit(test_data["classes"])
    class_names = list(label_encoder.classes_)

    model = build_tuned_cnn_transformer((input_length, num_channels), num_classes)
    model.load_weights(CHECKPOINT_PATH)
    print(f"Loaded {CHECKPOINT_PATH}")

    test_gen = DataGenerator(test_data, label_encoder, shuffle=False, augment=False)
    all_signals, all_labels = [], []
    for x, y in test_gen:
        all_signals.append(x); all_labels.append(y)
    all_signals = np.concatenate(all_signals, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Evaluating on {len(all_signals)} test samples.")

    # 12-lead baseline
    probs_12 = model.predict(all_signals, batch_size=32, verbose=0)
    pred_12 = probs_12.argmax(axis=1)
    acc_12, rep_12 = per_class_report(all_labels, pred_12, class_names)

    # 3-lead (V1/V2/V3) — others zeroed
    masked_signals = mask_to_subset(all_signals, KEEP_INDICES)
    probs_3 = model.predict(masked_signals, batch_size=32, verbose=0)
    pred_3 = probs_3.argmax(axis=1)
    acc_3, rep_3 = per_class_report(all_labels, pred_3, class_names)

    cm_3 = confusion_matrix(all_labels, pred_3, labels=range(num_classes))
    plot_cm(cm_3, class_names,
            f"3-lead (V1/V2/V3) confusion matrix — acc {acc_3:.3f}",
            os.path.join(OUTPUT_DIR, "3lead_confusion_matrix.png"))

    lines = []
    lines.append("WEARABLE 3-LEAD COMPARISON")
    lines.append(f"Model: {CHECKPOINT_PATH}")
    lines.append(f"Wearable leads: {WEARABLE_LEADS}  (indices {KEEP_INDICES})")
    lines.append("=" * 60)
    lines.append(f"\nFull 12-lead accuracy: {acc_12:.4f}")
    lines.append(f"3-lead V1/V2/V3 accuracy: {acc_3:.4f}")
    lines.append(f"Absolute drop: {(acc_12 - acc_3):.4f} "
                 f"({(acc_12 - acc_3) / max(acc_12, 1e-9) * 100:.2f}% relative)")
    lines.append("\n--- 12-LEAD CLASSIFICATION REPORT ---")
    lines.append(rep_12)
    lines.append("\n--- 3-LEAD (V1/V2/V3) CLASSIFICATION REPORT ---")
    lines.append(rep_3)

    # Per-class delta in F1
    from sklearn.metrics import f1_score
    f1_12 = f1_score(all_labels, pred_12, average=None, labels=range(num_classes), zero_division=0)
    f1_3 = f1_score(all_labels, pred_3, average=None, labels=range(num_classes), zero_division=0)
    lines.append("\n--- PER-CLASS F1 DELTA (12-lead -> 3-lead) ---")
    for i, c in enumerate(class_names):
        lines.append(f"  {c:6s}  12L={f1_12[i]:.3f}  3L={f1_3[i]:.3f}  "
                     f"Δ={f1_3[i]-f1_12[i]:+.3f}")

    text = "\n".join(lines)
    out_path = os.path.join(OUTPUT_DIR, "3lead_comparison.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {os.path.join(OUTPUT_DIR, '3lead_confusion_matrix.png')}")


if __name__ == "__main__":
    main()
