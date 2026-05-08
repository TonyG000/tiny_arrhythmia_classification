"""Render the comparative analysis as presentation-ready plots."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.utils import model_results_path

OUTPUT_DIR = os.path.join(model_results_path, "comparative")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    "CNN-BiLSTM\n(baseline)",
    "CNN-Transformer\n(original)",
    "CNN-Transformer\n(tuned)",
    "CNN-Transformer\n(tuned + lead-masked)",
    "CNN-BiLSTM-\nTransformer Ensemble",
]

# rows: models, cols: [Acc, Macro Recall, Macro F1]
HEADLINE = np.array([
    [0.7446, 0.7252, 0.7085],
    [0.7935, 0.7958, 0.7611],
    [0.7745, 0.7644, 0.7416],
    [0.7962, 0.7973, 0.7721],
    [0.7867, 0.7812, 0.7570],
])
METRIC_NAMES = ["Accuracy", "Macro Recall", "Macro F1"]

CLASS_NAMES = ["AF", "IAVB", "LBBB", "PAC", "PVC", "RBBB", "SNR", "STD", "STE"]
PER_CLASS_F1 = np.array([
    [0.769, 0.797, 0.690, 0.543, 0.789, 0.830, 0.688, 0.729, 0.542],  # baseline
    [0.856, 0.853, 0.741, 0.667, 0.780, 0.822, 0.791, 0.825, 0.515],  # original
    [0.833, 0.818, 0.792, 0.604, 0.771, 0.814, 0.767, 0.783, 0.492],  # tuned
    [0.839, 0.843, 0.737, 0.682, 0.781, 0.805, 0.818, 0.820, 0.625],  # tuned+masked
    [0.838, 0.834, 0.750, 0.638, 0.797, 0.826, 0.767, 0.775, 0.588],  # ensemble
])

BEST_IDX = 3  # tuned + lead-masked


def plot_headline_bars():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(MODELS))
    w = 0.27
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, metric in enumerate(METRIC_NAMES):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, HEADLINE[:, i], w, label=metric, color=colors[i])
        for j, b in enumerate(bars):
            label_color = "black"
            weight = "bold" if j == BEST_IDX else "normal"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{HEADLINE[j, i]:.3f}", ha="center", va="bottom",
                    fontsize=8, color=label_color, fontweight=weight)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylim(0.65, 0.85)
    ax.set_ylabel("Score")
    ax.set_title("Comparative analysis — all model variants on full 12-lead test set (n=736)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Highlight best model
    ax.axvspan(BEST_IDX - 0.5, BEST_IDX + 0.5, alpha=0.08, color="gold", zorder=0)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "comparative_headline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_per_class_heatmap():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(PER_CLASS_F1, cmap="RdYlGn", vmin=0.4, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([m.replace("\n", " ") for m in MODELS])
    ax.set_xlabel("Arrhythmia class"); ax.set_title("Per-class F1 across model variants", fontsize=11)

    # Annotate each cell; bold the best per column
    best_per_class = PER_CLASS_F1.argmax(axis=0)
    for i in range(PER_CLASS_F1.shape[0]):
        for j in range(PER_CLASS_F1.shape[1]):
            v = PER_CLASS_F1[i, j]
            is_best = (i == best_per_class[j])
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8.5, color="black",
                    fontweight="bold" if is_best else "normal")
    plt.colorbar(im, ax=ax, label="F1 score", fraction=0.025)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "comparative_per_class_f1.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_per_class_bars():
    """Best-per-class bar showing which model wins each class."""
    best_per_class = PER_CLASS_F1.argmax(axis=0)
    best_scores = PER_CLASS_F1.max(axis=0)
    bilstm_scores = PER_CLASS_F1[0]  # baseline reference

    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bar_colors = [palette[m] for m in best_per_class]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(CLASS_NAMES))
    w = 0.4
    ax.bar(x - w/2, bilstm_scores, w, label="Baseline (CNN-BiLSTM)",
           color="lightgray", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, best_scores, w, color=bar_colors, edgecolor="black", linewidth=0.5)

    for i, (s, b) in enumerate(zip(best_scores, bilstm_scores)):
        ax.text(i + w/2, s + 0.005, f"{s:.2f}", ha="center", fontsize=8)
        ax.text(i - w/2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8, color="dimgray")

    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class winner vs baseline — bars colored by which model won",
                 fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    handles = [mpatches.Patch(color="lightgray", label="Baseline (CNN-BiLSTM)")]
    used_indices = sorted(set(best_per_class))
    for idx in used_indices:
        handles.append(mpatches.Patch(color=palette[idx],
                                      label=f"Winner: {MODELS[idx].replace(chr(10), ' ')}"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "comparative_per_class_winner.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_headline_bars()
    plot_per_class_heatmap()
    plot_per_class_bars()
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
