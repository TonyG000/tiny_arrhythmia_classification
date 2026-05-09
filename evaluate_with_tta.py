"""
evaluate_with_tta.py

Test-time augmentation evaluation for the CNN-Transformer model. Runs N forward
passes per sample with random lead masks (mirroring training-time augmentation),
averages the softmax outputs, and produces:
  - PR curves per class (with average precision)
  - Classification report and confusion matrix
  - Comparison metrics vs. single-pass inference

Usage:
    .venv/bin/python3 evaluate_with_tta.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
)

from data_processing.data_generator import DataGenerator
from utils.utils import num_classes, model_results_path
from utils.model_selector import get_model
from models.model.cnn_attention_transformer import TransformerEncoderBlock
from models.focal_loss import SparseCategoricalFocalLoss
from config import MODEL_NAME, LEARNING_RATE

# --- Settings ---
N_TTA_PASSES = 7  # 1 clean 12-lead + 6 random-mask passes
LEAD_OPTIONS = [12, 12, 3, 3, 1, 1]  # masks for the 6 augmented passes

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=f"models/checkpoints/{MODEL_NAME}_best.h5",
                    help="Path to model weights .h5 file")
parser.add_argument("--tag", default="TTA",
                    help="Suffix for output filenames (e.g. 'TTA_leadmasked_tuned')")
args = parser.parse_args()
CHECKPOINT_PATH = args.checkpoint
OUT_TAG = args.tag

# --- Load test data ---
with open("data/test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

label_encoder = LabelEncoder()
label_encoder.fit(test_data["classes"])
class_names = label_encoder.classes_

# Generator with augment=False: we'll apply masks manually
gen = DataGenerator(test_data, label_encoder, shuffle=False, augment=False)

# --- Load model ---
# Use load_model so architecture comes from the .h5 file (works for both
# CNNTransformerModel_best.h5 and tuned variants with different hyperparams).
try:
    model = tf.keras.models.load_model(
        CHECKPOINT_PATH,
        custom_objects={
            "TransformerEncoderBlock": TransformerEncoderBlock,
            "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
            "sparse_categorical_focal_loss_1": SparseCategoricalFocalLoss(gamma=2.0),
        },
        compile=False,
    )
except Exception as e:
    print(f"[load_model failed: {e}] falling back to load_weights")
    model = get_model(MODEL_NAME, gen[0][0].shape[1:], num_classes)
    model.load_weights(CHECKPOINT_PATH)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

def apply_lead_mask(x, k):
    """Keep k random leads per sample in batch, zero out the rest."""
    if k >= x.shape[-1]:
        return x
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        kept = np.random.choice(x.shape[-1], size=k, replace=False)
        out[i, :, kept] = x[i, :, kept]
    return out

# --- Run TTA ---
print(f"[TTA] {N_TTA_PASSES} passes per sample (1 clean + {len(LEAD_OPTIONS)} masked).")
all_probs = []
y_true = []

for batch_idx, (x, y) in enumerate(gen):
    if x.shape[0] == 0:
        break
    y_true.extend(y)

    # Pass 1: clean 12-lead
    probs = model.predict(x, verbose=0)

    # Passes 2..N: random lead masks
    for k in LEAD_OPTIONS:
        x_aug = apply_lead_mask(x, k)
        probs += model.predict(x_aug, verbose=0)

    probs /= N_TTA_PASSES
    all_probs.append(probs)
    if batch_idx % 5 == 0:
        print(f"  batch {batch_idx + 1}/{len(gen)}")

y_true = np.array(y_true, dtype=int)
y_prob_tta = np.concatenate(all_probs, axis=0)
y_pred_tta = np.argmax(y_prob_tta, axis=1)

# --- Single-pass baseline for comparison ---
print("[single-pass] Computing baseline...")
y_prob_single = model.predict(gen, verbose=0)
y_pred_single = np.argmax(y_prob_single, axis=1)

# --- Metrics ---
os.makedirs(model_results_path, exist_ok=True)

def per_class_ap(y_true, y_prob):
    onehot = np.eye(num_classes)[y_true]
    return [average_precision_score(onehot[:, i], y_prob[:, i]) for i in range(num_classes)]

ap_single = per_class_ap(y_true, y_prob_single)
ap_tta = per_class_ap(y_true, y_prob_tta)

print("\n=== Average Precision (per class) ===")
print(f"{'class':<8} {'single':>8} {'TTA':>8} {'delta':>8}")
for i, name in enumerate(class_names):
    delta = ap_tta[i] - ap_single[i]
    print(f"{name:<8} {ap_single[i]:>8.4f} {ap_tta[i]:>8.4f} {delta:>+8.4f}")
print(f"{'mAP':<8} {np.mean(ap_single):>8.4f} {np.mean(ap_tta):>8.4f} "
      f"{np.mean(ap_tta) - np.mean(ap_single):>+8.4f}")

print("\n=== Classification Report (TTA) ===")
report_tta = classification_report(y_true, y_pred_tta, target_names=class_names, digits=4)
print(report_tta)

# --- PR curves plot ---
plt.figure(figsize=(12, 8))
for i, name in enumerate(class_names):
    onehot_i = (y_true == i).astype(int)
    p, r, _ = precision_recall_curve(onehot_i, y_prob_tta[:, i])
    plt.plot(r, p, label=f"{name} (AP={ap_tta[i]:.3f})")
plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.title(f"PR Curves — {MODEL_NAME} (TTA, N={N_TTA_PASSES})", fontsize=14)
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
pr_path = os.path.join(model_results_path, f"{MODEL_NAME}_{OUT_TAG}_pr_curves.png")
plt.savefig(pr_path, dpi=300)
plt.close()
print(f"\n[saved] PR curves -> {pr_path}")

# --- Confusion matrix (TTA) ---
cm = confusion_matrix(y_true, y_pred_tta)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"{MODEL_NAME} TTA Confusion Matrix (Normalized)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_path = os.path.join(model_results_path, f"{MODEL_NAME}_{OUT_TAG}_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"[saved] Confusion matrix -> {cm_path}")

# --- Save report + AP table ---
report_path = os.path.join(model_results_path, f"{MODEL_NAME}_{OUT_TAG}_report.txt")
with open(report_path, "w") as f:
    f.write(f"Model: {MODEL_NAME}\nCheckpoint: {CHECKPOINT_PATH}\n")
    f.write(f"TTA passes: {N_TTA_PASSES}\n\n")
    f.write("=== Average Precision ===\n")
    f.write(f"{'class':<8} {'single':>8} {'TTA':>8} {'delta':>8}\n")
    for i, name in enumerate(class_names):
        f.write(f"{name:<8} {ap_single[i]:>8.4f} {ap_tta[i]:>8.4f} "
                f"{ap_tta[i] - ap_single[i]:>+8.4f}\n")
    f.write(f"{'mAP':<8} {np.mean(ap_single):>8.4f} {np.mean(ap_tta):>8.4f} "
            f"{np.mean(ap_tta) - np.mean(ap_single):>+8.4f}\n\n")
    f.write("=== Classification Report (TTA) ===\n")
    f.write(report_tta)
print(f"[saved] Report -> {report_path}")
