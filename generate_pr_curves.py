# generate_pr_curves.py
import os
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, average_precision_score
from data_processing.data_generator import DataGenerator
from utils.utils import input_length, num_channels, num_classes, model_results_path
from models.model.cnn_attention_transformer import TransformerEncoderBlock
from models.focal_loss import SparseCategoricalFocalLoss
from utils.model_selector import get_model

# --- Load test data ---
with open('data/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

le = LabelEncoder()
le.fit(test_data['classes'])
gen = DataGenerator(test_data, le)
class_names = le.classes_

# --- Load model ---
model = tf.keras.models.load_model(
    'models/checkpoints/CNNTransformerModel_lead_masked_tuned_best.h5',
    custom_objects={
        'TransformerEncoderBlock': TransformerEncoderBlock,
        'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss,
        'sparse_categorical_focal_loss_1': SparseCategoricalFocalLoss(gamma=2.0)
    }
)
# --- Get predictions ---
y_true = np.concatenate([y for _, y in gen], axis=0)
y_prob = model.predict(gen)

# One-hot encode y_true for per-class curves
y_true_onehot = np.eye(num_classes)[y_true.astype(int)]

# --- Plot ---
os.makedirs(model_results_path, exist_ok=True)

plt.figure(figsize=(12, 8))
for i, class_name in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_prob[:, i])
    ap = average_precision_score(y_true_onehot[:, i], y_prob[:, i])
    plt.plot(recall, precision, label=f'{class_name} (AP={ap:.2f})')

plt.xlabel('Recall', fontsize=13)
plt.ylabel('Precision', fontsize=13)
plt.title('Precision-Recall Curves — CNN-Transformer Lead Masked Tuned', fontsize=14)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_path = os.path.join(model_results_path, 'CNN_Transformer_lead_masked_tuned_pr_curves.png')
plt.savefig(out_path, dpi=300)
plt.close()
print(f'Saved to: {out_path}')