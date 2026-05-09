"""Run a single forward pass (no TTA) on the test set with the tuned + lead-masked
CNN-Transformer and save clean confusion matrix + PR curves for the website."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder, label_binarize

from data_processing.data_generator import DataGenerator
from utils.utils import num_classes
from models.model.cnn_attention_transformer import TransformerEncoderBlock
from models.focal_loss import SparseCategoricalFocalLoss

CHECKPOINT = "models/checkpoints/CNNTransformerModel_lead_masked_tuned_best.h5"
OUT_DIR = "sherifmost DeepLearning master Project-website-template/team-specific/resources/images"

with open("data/test_data.pkl", "rb") as f:
    test_data = pickle.load(f)
label_encoder = LabelEncoder().fit(test_data["classes"])
class_names = label_encoder.classes_
test_gen = DataGenerator(test_data, label_encoder, shuffle=False, augment=False)

model = tf.keras.models.load_model(
    CHECKPOINT,
    custom_objects={
        "TransformerEncoderBlock": TransformerEncoderBlock,
        "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
        "sparse_categorical_focal_loss_1": SparseCategoricalFocalLoss(gamma=2.0),
    },
    compile=False,
)

probs = model.predict(test_gen, verbose=1)
y_true = np.concatenate([np.argmax(y, axis=1) if y.ndim == 2 else y for _, y in test_gen])
y_pred = np.argmax(probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(9, 7))
sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
            xticklabels=class_names, yticklabels=class_names, cbar_kws={"label": "Row-normalized rate"})
plt.title("Final Model Confusion Matrix (CNN-Transformer, tuned + lead-masked)")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=130, bbox_inches="tight")
plt.close()

y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
plt.figure(figsize=(9, 7))
for i, name in enumerate(class_names):
    p, r, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
    ap = average_precision_score(y_true_bin[:, i], probs[:, i])
    plt.plot(r, p, label=f"{name} (AP={ap:.3f})")
plt.title("Final Model Precision-Recall Curves (CNN-Transformer, tuned + lead-masked)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curves.png"), dpi=130, bbox_inches="tight")
plt.close()

acc = (y_true == y_pred).mean()
print(f"Single-pass accuracy (no TTA): {acc:.4f}")
print("Saved confusion_matrix.png and pr_curves.png")
