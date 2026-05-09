"""Generate a lead masking visualization for the website."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SAMPLE_FILE = "/home/g6/cpsc_2018_data/g1/A0001.mat"
SECONDS = 5
FS = 250
N = SECONDS * FS

mat = loadmat(SAMPLE_FILE)
sig = mat["val"].astype(np.float32)
sig = sig[:, :N]
sig = sig / (np.max(np.abs(sig), axis=1, keepdims=True) + 1e-6)

rng = np.random.default_rng(42)
masked_leads = rng.choice(12, size=4, replace=False)
masked = sig.copy()
masked[masked_leads, :] = 0

fig, axes = plt.subplots(12, 2, figsize=(14, 12), sharex=True)
fig.suptitle("Lead Masking Augmentation: full 12-lead input (left) vs randomly masked input (right)",
             fontsize=14, fontweight="bold")
t = np.arange(N) / FS
for i in range(12):
    axes[i, 0].plot(t, sig[i], color="#1f77b4", linewidth=0.8)
    axes[i, 0].set_ylabel(LEAD_NAMES[i], rotation=0, ha="right", va="center", fontweight="bold")
    axes[i, 0].set_yticks([])
    axes[i, 0].set_ylim(-1.2, 1.2)

    is_masked = i in masked_leads
    color = "#d62728" if is_masked else "#1f77b4"
    axes[i, 1].plot(t, masked[i], color=color, linewidth=0.8)
    axes[i, 1].set_yticks([])
    axes[i, 1].set_ylim(-1.2, 1.2)
    if is_masked:
        axes[i, 1].axhspan(-1.2, 1.2, color="#d62728", alpha=0.08)
        axes[i, 1].text(0.5, 0, "MASKED", transform=axes[i, 1].transAxes,
                        ha="center", va="center", color="#d62728",
                        fontsize=10, fontweight="bold")

axes[0, 0].set_title("Original 12-lead ECG", fontsize=12)
axes[0, 1].set_title(f"After lead masking ({len(masked_leads)} of 12 leads zeroed)", fontsize=12)
axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = "/home/g6/tiny_arrhythmia_classification/sherifmost DeepLearning master Project-website-template/team-specific/resources/images/lead_masking.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"Saved to {out}")
