"""
demo_classify.py

Single-file CLI demo: takes one CPSC-format .mat ECG file and predicts the
arrhythmia class with confidence. Optionally runs TTA via lead masking.

Usage:
    .venv/bin/python3 demo_classify.py /path/to/A0001.mat
    .venv/bin/python3 demo_classify.py /path/to/A0001.mat --tta
    .venv/bin/python3 demo_classify.py /path/to/A0001.mat --top 3
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from data_processing.cpsc_signal_loader import RawSignalLoader, SignalProcessor
from utils.utils import (
    arrhythmia_classes_cpsc, input_length, num_channels, num_classes,
)
from utils.model_selector import get_model
from models.model.cnn_attention_transformer import TransformerEncoderBlock
from models.focal_loss import SparseCategoricalFocalLoss
from config import MODEL_NAME, LEARNING_RATE, SAMPLING_RATE, TARGET_LENGTH

CLASS_DESCRIPTIONS = {
    "SNR":  "Sinus Rhythm (Normal)",
    "AF":   "Atrial Fibrillation",
    "IAVB": "First-Degree AV Block",
    "LBBB": "Left Bundle Branch Block",
    "RBBB": "Right Bundle Branch Block",
    "PAC":  "Premature Atrial Contraction",
    "PVC":  "Premature Ventricular Contraction",
    "STD":  "ST-segment Depression",
    "STE":  "ST-segment Elevation",
}

def random_lead_mask(signal, k):
    """Keep k random leads; zero out the rest. signal shape (T, L)."""
    if k >= signal.shape[-1]:
        return signal
    out = np.zeros_like(signal)
    kept = np.random.choice(signal.shape[-1], size=k, replace=False)
    out[:, kept] = signal[:, kept]
    return out

def main():
    parser = argparse.ArgumentParser(description="ECG arrhythmia classification demo")
    parser.add_argument("mat_path", help="Path to CPSC .mat file (12-lead ECG)")
    parser.add_argument("--tta", action="store_true",
                        help="Apply test-time augmentation (7 passes)")
    parser.add_argument("--top", type=int, default=3,
                        help="Show top-N predictions (default 3)")
    parser.add_argument("--checkpoint", default=f"models/checkpoints/{MODEL_NAME}_best.h5",
                        help="Path to model checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.mat_path):
        raise FileNotFoundError(args.mat_path)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    # --- Load + preprocess signal ---
    print(f"[load] {args.mat_path}")
    raw_signals, _ = RawSignalLoader(file_path=args.mat_path, lead_mode="multi").load_signal()
    processor = SignalProcessor(
        sampling_rate=SAMPLING_RATE, target_length=TARGET_LENGTH, lead_mode="multi"
    )
    signal = processor.process_signal(raw_signals).astype(np.float32)
    print(f"[shape] {signal.shape} (time, leads)")

    # --- Load model ---
    # Use load_model so architecture (incl. tuned hyperparams) comes from the
    # .h5 file. Falls back to get_model+load_weights for weights-only files.
    print(f"[model] {MODEL_NAME} <- {args.checkpoint}")
    try:
        model = tf.keras.models.load_model(
            args.checkpoint,
            custom_objects={
                "TransformerEncoderBlock": TransformerEncoderBlock,
                "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
                "sparse_categorical_focal_loss_1": SparseCategoricalFocalLoss(gamma=2.0),
            },
            compile=False,
        )
    except Exception as e:
        print(f"[load_model failed: {e}] falling back to load_weights")
        model = get_model(MODEL_NAME, (input_length, num_channels), num_classes)
        model.load_weights(args.checkpoint)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # --- Class label order matches LabelEncoder fit during training ---
    le = LabelEncoder()
    le.fit(arrhythmia_classes_cpsc)
    class_names = le.classes_

    # --- Predict ---
    x = np.expand_dims(signal, axis=0)
    if args.tta:
        lead_options = [12, 12, 3, 3, 1, 1]  # 6 augmented passes + 1 clean = 7
        probs = model.predict(x, verbose=0)[0]
        for k in lead_options:
            x_aug = np.expand_dims(random_lead_mask(signal, k), axis=0)
            probs += model.predict(x_aug, verbose=0)[0]
        probs /= (1 + len(lead_options))
        method = f"TTA ({1 + len(lead_options)} passes)"
    else:
        probs = model.predict(x, verbose=0)[0]
        method = "single pass"

    # --- Output ---
    top_idx = np.argsort(probs)[::-1][: args.top]
    print(f"\n=== Prediction ({method}) ===")
    print(f"{'rank':<5}{'class':<8}{'description':<35}{'confidence':>12}")
    for rank, i in enumerate(top_idx, start=1):
        name = class_names[i]
        desc = CLASS_DESCRIPTIONS.get(name, "")
        print(f"{rank:<5}{name:<8}{desc:<35}{probs[i]:>11.2%}")

    print(f"\nPredicted: {class_names[top_idx[0]]} "
          f"({CLASS_DESCRIPTIONS.get(class_names[top_idx[0]], '')}) "
          f"with {probs[top_idx[0]]:.1%} confidence.")

if __name__ == "__main__":
    main()
