"""Comparative analysis across all trained model variants.

Loads every checkpoint with its matching architecture and reports:
  - Accuracy, macro-recall, macro-F1
  - Per-class F1 (one column per class)
  - Per-class recall (one column per class)

A clean single-table summary for presentation.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             classification_report)

from data_processing.data_generator import DataGenerator
from utils.model_selector import get_model
from utils.utils import (input_length, num_channels, num_classes,
                         arrhythmia_classes_cpsc, model_results_path)
from explainability.lead_importance import build_tuned_cnn_transformer

OUTPUT_DIR = os.path.join(model_results_path, "comparative")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_DIR = "models/checkpoints"

# Each entry: display_name -> (builder, checkpoint_path)
# Builders take (input_shape, num_classes) and return a compiled keras Model.
def build_default_cnn_transformer(input_shape, n_classes):
    return get_model("CNNTransformerModel", input_shape, n_classes)

def build_baseline_cnn_bilstm(input_shape, n_classes):
    return get_model("EnhancedCNNModel", input_shape, n_classes)

def build_ensemble(input_shape, n_classes):
    return get_model("EnsembleModel", input_shape, n_classes)

def build_cnn_transformer_tuned_small(input_shape, n_classes):
    # CNNTransformerModel_tuned_best.h5: num_heads=4, ff_dim=128 (best non-masked trial)
    return build_tuned_cnn_transformer(input_shape, n_classes,
                                       num_heads=4, ff_dim=128)

CONFIGS = [
    ("CNN-BiLSTM (baseline)", build_baseline_cnn_bilstm,
     f"{CKPT_DIR}/EnhancedCNNModel_best.h5"),
    ("CNN-Transformer (original)", build_default_cnn_transformer,
     f"{CKPT_DIR}/CNNTransformerModel_best_original.h5"),
    ("CNN-Transformer (tuned)", build_cnn_transformer_tuned_small,
     f"{CKPT_DIR}/CNNTransformerModel_tuned_best.h5"),
    ("CNN-Transformer (tuned + lead-masked)", build_tuned_cnn_transformer,
     f"{CKPT_DIR}/CNNTransformerModel_lead_masked_tuned_best.h5"),
    ("CNN-BiLSTM-Transformer Ensemble", build_ensemble,
     f"{CKPT_DIR}/EnsembleModel_best.h5"),
]


def load_test_set():
    with open("data/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    label_encoder = LabelEncoder()
    label_encoder.fit(test_data["classes"])
    class_names = list(label_encoder.classes_)

    gen = DataGenerator(test_data, label_encoder, shuffle=False, augment=False)
    xs, ys = [], []
    for x, y in gen:
        xs.append(x); ys.append(y)
    return np.concatenate(xs), np.concatenate(ys), class_names


def evaluate(name, builder, ckpt, x, y, class_names):
    print(f"\n>>> {name}")
    if not os.path.exists(ckpt):
        print(f"   missing checkpoint: {ckpt}")
        return None
    try:
        model = builder((input_length, num_channels), num_classes)
        model.load_weights(ckpt)
    except Exception as e:
        print(f"   FAILED to load: {type(e).__name__}: {str(e)[:200]}")
        return None
    probs = model.predict(x, batch_size=32, verbose=0)
    pred = probs.argmax(axis=1)
    return {
        "name": name,
        "ckpt": ckpt,
        "acc": accuracy_score(y, pred),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "f1_per": f1_score(y, pred, average=None, labels=range(num_classes), zero_division=0),
        "recall_per": recall_score(y, pred, average=None, labels=range(num_classes), zero_division=0),
        "report": classification_report(y, pred, target_names=class_names,
                                        digits=3, zero_division=0),
    }


def render_results(results, class_names, save_path):
    lines = []
    lines.append("COMPARATIVE ANALYSIS — all model variants on the same test set (n=" +
                 str(len(class_names)) + " classes)")
    lines.append("=" * 110)

    # Headline table
    lines.append(f"\n{'Model':<42} {'Acc':>8} {'M-Rec':>8} {'M-F1':>8}")
    lines.append("-" * 70)
    for r in results:
        lines.append(f"{r['name']:<42} {r['acc']:>8.4f} {r['macro_recall']:>8.4f} "
                     f"{r['macro_f1']:>8.4f}")

    # Per-class F1 table
    lines.append("\nPer-class F1:")
    header = f"{'Model':<42}" + "".join(f"{c:>7}" for c in class_names)
    lines.append(header); lines.append("-" * len(header))
    for r in results:
        lines.append(f"{r['name']:<42}" +
                     "".join(f"{v:>7.3f}" for v in r["f1_per"]))

    # Per-class recall table
    lines.append("\nPer-class Recall:")
    lines.append(header); lines.append("-" * len(header))
    for r in results:
        lines.append(f"{r['name']:<42}" +
                     "".join(f"{v:>7.3f}" for v in r["recall_per"]))

    # Best-per-class summary
    lines.append("\nBest model per class (by F1):")
    arr = np.array([r["f1_per"] for r in results])           # (M, C)
    for ci, c in enumerate(class_names):
        best = int(np.argmax(arr[:, ci]))
        lines.append(f"  {c:6s}  -> {results[best]['name']}  (F1={arr[best, ci]:.3f})")

    text = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    x, y, class_names = load_test_set()
    print(f"Loaded {len(x)} test samples across {len(class_names)} classes.")

    results = []
    for name, builder, ckpt in CONFIGS:
        r = evaluate(name, builder, ckpt, x, y, class_names)
        if r is not None:
            results.append(r)

    if not results:
        print("No models loaded successfully.")
        return

    out_path = os.path.join(OUTPUT_DIR, "comparative_summary.txt")
    render_results(results, class_names, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
