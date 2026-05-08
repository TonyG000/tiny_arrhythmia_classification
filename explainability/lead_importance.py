"""Per-lead importance analysis for picking the top-k leads for wearables.

Two complementary attribution methods, run on every test sample, aggregated
per arrhythmia class:

  1. Input-gradient saliency  -- sum(|grad of P(true_class) wrt input| * |input|)
                                 over time, per lead. Same backward-pass idea
                                 as Grad-CAM but attributed to the input layer
                                 (where the 12 leads still exist as channels).
  2. Lead occlusion           -- zero each lead, measure drop in P(true_class).
                                 Direct "what if this lead were missing?" test.

Outputs:
  - lead_importance_saliency.png    (num_classes x 12 heatmap)
  - lead_importance_occlusion.png   (num_classes x 12 heatmap)
  - lead_importance_overall.png     (mean importance per lead, both methods)
  - lead_importance_summary.txt     (top-3 leads per class + overall top-3)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import layers, models, regularizers

from data_processing.data_generator import DataGenerator
from models.model.cnn_attention_transformer import TransformerEncoderBlock
from utils.utils import input_length, num_channels, num_classes, model_results_path
from config import LEARNING_RATE

# Use the tuned + lead-masked checkpoint (best trial: num_heads=8, ff_dim=512).
MODEL_NAME = "CNNTransformerModel (tuned, lead-masked)"
CHECKPOINT_PATH = "models/checkpoints/CNNTransformerModel_lead_masked_tuned_best.h5"
TUNED_NUM_HEADS = 8
TUNED_FF_DIM = 512

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

OUTPUT_DIR = os.path.join(model_results_path, "lead_importance")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_tuned_cnn_transformer(input_shape, n_classes,
                                 num_heads=TUNED_NUM_HEADS, ff_dim=TUNED_FF_DIM):
    """Mirror models/model/cnn_attention_transformer.py but with tuned heads/ff_dim."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 15, strides=2, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.Conv1D(32, 15, strides=1, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(64, 15, strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.Conv1D(64, 15, strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, 15, strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.Conv1D(128, 15, strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2, strides=2, padding='same')(x)
    x = layers.Dropout(0.3)(x)

    attention = layers.Conv1D(128, 1, strides=1, activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    x = layers.MaxPooling1D(2, strides=2, padding='same', name='gradcam_target')(x)
    x = layers.Dropout(0.4)(x)

    x = TransformerEncoderBlock(embed_dim=128, num_heads=num_heads,
                                ff_dim=ff_dim, dropout=0.1)(x)
    x = TransformerEncoderBlock(embed_dim=128, num_heads=num_heads,
                                ff_dim=ff_dim, dropout=0.1)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_Transformer_Tuned")


def compute_saliency(model, signal, class_idx):
    """|grad(P(class)) wrt input| * |input|, summed over time -> (num_leads,)."""
    x = tf.convert_to_tensor(signal[np.newaxis], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        probs = model(x, training=False)
        score = probs[:, class_idx]
    grad = tape.gradient(score, x)[0].numpy()           # (time, leads)
    contribution = np.abs(grad) * np.abs(signal)        # input * gradient
    return contribution.sum(axis=0)                     # (leads,)


def compute_occlusion(model, signal, class_idx, baseline_prob):
    """Drop in P(class) when each lead is zeroed -> (num_leads,)."""
    n_leads = signal.shape[-1]
    occluded = np.tile(signal[np.newaxis], (n_leads, 1, 1))   # (leads, time, leads)
    for i in range(n_leads):
        occluded[i, :, i] = 0.0
    probs = model(tf.convert_to_tensor(occluded, dtype=tf.float32),
                  training=False).numpy()
    drops = baseline_prob - probs[:, class_idx]
    return np.maximum(drops, 0.0)                       # negative drops -> 0


def normalize_rows(matrix):
    row_max = matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    return matrix / row_max


def plot_heatmap(matrix, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(LEAD_NAMES)))
    ax.set_xticklabels(LEAD_NAMES)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("ECG Lead")
    ax.set_ylabel("Arrhythmia Class")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if matrix[i, j] < 0.6 else "black",
                    fontsize=7)
    plt.colorbar(im, ax=ax, label="Normalized importance (per row)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_overall(saliency_overall, occlusion_overall, save_path):
    x = np.arange(len(LEAD_NAMES))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, saliency_overall, width, label="Saliency", color="#4C72B0")
    ax.bar(x + width / 2, occlusion_overall, width, label="Occlusion", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(LEAD_NAMES)
    ax.set_ylabel("Mean importance across all classes (normalized)")
    ax.set_title("Overall lead importance — mean across arrhythmia classes")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_summary(saliency, occlusion, class_names, save_path):
    sal_overall = saliency.mean(axis=0)
    occ_overall = occlusion.mean(axis=0)
    combined = (sal_overall / sal_overall.max()) + (occ_overall / occ_overall.max())

    lines = []
    lines.append("LEAD IMPORTANCE SUMMARY")
    lines.append(f"Model: {MODEL_NAME}")
    lines.append("=" * 60)
    lines.append("\nTop-3 leads per class (by saliency / occlusion):")
    for i, cname in enumerate(class_names):
        sal_top = np.argsort(saliency[i])[::-1][:3]
        occ_top = np.argsort(occlusion[i])[::-1][:3]
        lines.append(
            f"  {cname:6s}  saliency: {[LEAD_NAMES[j] for j in sal_top]}    "
            f"occlusion: {[LEAD_NAMES[j] for j in occ_top]}"
        )

    lines.append("\nOverall lead ranking (combined saliency + occlusion):")
    overall_rank = np.argsort(combined)[::-1]
    for r, idx in enumerate(overall_rank):
        lines.append(f"  {r+1:2d}. {LEAD_NAMES[idx]:4s}  "
                     f"saliency={sal_overall[idx]:.4f}  "
                     f"occlusion={occ_overall[idx]:.4f}")

    top3 = [LEAD_NAMES[i] for i in overall_rank[:3]]
    lines.append(f"\nRecommended top-3 leads for wearable deployment: {top3}")

    text = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    with open("data/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    label_encoder = LabelEncoder()
    label_encoder.fit(test_data["classes"])
    class_names = list(label_encoder.classes_)

    model = build_tuned_cnn_transformer((input_length, num_channels), num_classes)
    model.load_weights(CHECKPOINT_PATH)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Loaded weights from {CHECKPOINT_PATH}")

    test_gen = DataGenerator(test_data, label_encoder, shuffle=False, augment=False)

    all_signals, all_labels = [], []
    for x_batch, y_batch in test_gen:
        all_signals.append(x_batch)
        all_labels.append(y_batch)
    all_signals = np.concatenate(all_signals, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Loaded {len(all_signals)} test samples across {len(class_names)} classes.")

    saliency_sum = np.zeros((num_classes, num_channels), dtype=np.float64)
    occlusion_sum = np.zeros((num_classes, num_channels), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)

    for i, (signal, label) in enumerate(zip(all_signals, all_labels)):
        baseline_probs = model(tf.convert_to_tensor(signal[np.newaxis],
                                                   dtype=tf.float32),
                               training=False).numpy()[0]
        baseline = baseline_probs[label]

        sal = compute_saliency(model, signal, label)
        occ = compute_occlusion(model, signal, label, baseline)

        saliency_sum[label] += sal
        occlusion_sum[label] += occ
        counts[label] += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(all_signals):
            print(f"  processed {i+1}/{len(all_signals)}")

    counts_safe = np.where(counts == 0, 1, counts)[:, None]
    saliency_mean = saliency_sum / counts_safe
    occlusion_mean = occlusion_sum / counts_safe

    saliency_norm = normalize_rows(saliency_mean)
    occlusion_norm = normalize_rows(occlusion_mean)

    plot_heatmap(saliency_norm, class_names,
                 f"Lead importance (input-gradient saliency) — {MODEL_NAME}",
                 os.path.join(OUTPUT_DIR, "lead_importance_saliency.png"))
    plot_heatmap(occlusion_norm, class_names,
                 f"Lead importance (occlusion) — {MODEL_NAME}",
                 os.path.join(OUTPUT_DIR, "lead_importance_occlusion.png"))

    sal_overall = saliency_norm.mean(axis=0)
    occ_overall = occlusion_norm.mean(axis=0)
    plot_overall(sal_overall, occ_overall,
                 os.path.join(OUTPUT_DIR, "lead_importance_overall.png"))

    write_summary(saliency_norm, occlusion_norm, class_names,
                  os.path.join(OUTPUT_DIR, "lead_importance_summary.txt"))

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
