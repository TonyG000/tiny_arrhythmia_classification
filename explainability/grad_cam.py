import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder

from data_processing.data_generator import DataGenerator
from utils.model_selector import get_model
from utils.utils import input_length, num_channels, num_classes, arrhythmia_classes_cpsc, model_results_path
from config import MODEL_NAME, LEARNING_RATE, CHECKPOINT_PATH, SAMPLING_RATE

GRADCAM_LAYER = "gradcam_target"
OUTPUT_DIR = os.path.join(model_results_path, "gradcam")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_FOR_PLOT = 1


def compute_gradcam(grad_model, signal, class_idx):
    signal_tensor = tf.cast(signal[np.newaxis], tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(signal_tensor, training=False)
        class_score = predictions[:, class_idx]
    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions.numpy()[0]


def plot_gradcam(signal, heatmap, true_label, pred_label, pred_prob, class_idx, save_path):
    time = np.arange(signal.shape[0]) / SAMPLING_RATE
    ecg = signal[:, LEAD_FOR_PLOT]

    heatmap_upsampled = np.interp(
        np.linspace(0, len(heatmap) - 1, len(ecg)),
        np.arange(len(heatmap)),
        heatmap
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5),
                                    gridspec_kw={"height_ratios": [4, 1]},
                                    sharex=True)

    ax1.plot(time, ecg, color="black", linewidth=0.7, label=f"Lead {LEAD_NAMES[LEAD_FOR_PLOT]}")
    ax1.set_ylabel("Amplitude (mV)")
    ax1.set_title(
        f"Grad-CAM  |  True: {true_label}  |  Predicted: {pred_label}  ({pred_prob:.1%})",
        fontsize=11
    )
    ax1.legend(loc="upper right", fontsize=8)

    cmap = plt.get_cmap("RdYlGn_r")
    for i in range(len(time) - 1):
        ax2.axvspan(time[i], time[i + 1], color=cmap(heatmap_upsampled[i]), alpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, orientation="horizontal", pad=0.4, fraction=0.8,
                 label="Importance (red = high, green = low)")
    ax2.set_yticks([])
    ax2.set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


with open("data/test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

label_encoder = LabelEncoder()
label_encoder.fit(test_data["classes"])

model = get_model(MODEL_NAME, (input_length, num_channels), num_classes)
model.load_weights(CHECKPOINT_PATH)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(GRADCAM_LAYER).output, model.output]
)

test_gen = DataGenerator(test_data, label_encoder, shuffle=False)

all_signals, all_labels = [], []
for x_batch, y_batch in test_gen:
    all_signals.append(x_batch)
    all_labels.append(y_batch)

all_signals = np.concatenate(all_signals, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

for class_idx, class_name in enumerate(label_encoder.classes_):
    class_mask = all_labels == class_idx
    class_signals = all_signals[class_mask]

    if len(class_signals) == 0:
        print(f"No samples found for class {class_name}, skipping.")
        continue

    signal = class_signals[0]
    heatmap, probs = compute_gradcam(grad_model, signal, class_idx)
    pred_idx = np.argmax(probs)
    pred_label = label_encoder.classes_[pred_idx]
    pred_prob = probs[pred_idx]

    save_path = os.path.join(OUTPUT_DIR, f"gradcam_{class_name}.png")
    plot_gradcam(signal, heatmap, class_name, pred_label, pred_prob, class_idx, save_path)
    print(f"Saved: {save_path}  (pred={pred_label}, conf={pred_prob:.1%})")

print(f"\nAll Grad-CAM plots saved to: {OUTPUT_DIR}")
