# train_custom_log_cpsc.py 
import os
# Set GPU visibility (if using GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import mlflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_recall_curve

from config import (
    MODEL_NAME, ENABLE_MLFLOW, MLFLOW_URI, EXPERIMENT_NAME, RUN_NAME,
    EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, CHECKPOINT_PATH, CSV_FILENAME,
    LOSS_TYPE, FOCAL_GAMMA
)
from models.focal_loss import SparseCategoricalFocalLoss
from utils.utils import (
    arrhythmia_classes_cpsc, samples_limit, input_length, 
    num_channels, num_classes
)
from utils.model_selector import get_model
from data_processing.data_generator import DataGenerator
from data_processing.data_loading import DataLoader
from data_processing.data_reduction import DataReducer
from data_processing.data_splitter import DataSplitter

class MetricsPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, log_frequency=10):
        super().__init__()
        self.val_data = val_data
        self.log_frequency = log_frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_frequency == 0:
            val_labels, val_preds = [], []
            for x_batch, y_batch in self.val_data:
                if len(x_batch) == 0:
                    print("Empty batch encountered. Skipping...")
                    break
                predictions = self.model.predict(x_batch)
                val_labels.extend(y_batch)
                val_preds.extend(np.argmax(predictions, axis=1))

            if len(val_labels) == 0:
                return

            val_labels = np.array(val_labels)
            val_preds = np.array(val_preds)

            self.plot_confusion_matrix(val_labels, val_preds, epoch)
            self.plot_precision_recall(val_labels, val_preds, epoch)

    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - Epoch {epoch}")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        path = f"confusion_matrix_epoch_{epoch}.png"
        plt.savefig(path)
        plt.close()
        if ENABLE_MLFLOW:
            mlflow.log_artifact(path)

    def plot_precision_recall(self, y_true, y_pred, epoch):
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        y_pred_bin = label_binarize(y_pred, classes=np.arange(num_classes))
        plt.figure(figsize=(8, 6))
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
            plt.plot(recall, precision, lw=2, label=f'Class {i}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - Epoch {epoch}")
        plt.legend(loc="best")
        path = f"precision_recall_epoch_{epoch}.png"
        plt.savefig(path)
        plt.close()
        if ENABLE_MLFLOW:
            mlflow.log_artifact(path)

# Prepare directories
os.makedirs("data", exist_ok=True)
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Load data
data_loader = DataLoader(CSV_FILENAME)
data = data_loader.load_data()

# Reduce and split
data = DataReducer(data, arrhythmia_classes_cpsc, max_samples=samples_limit).reduce_data()
train_data, val_data, test_data = DataSplitter(data).split()

# Save test data
with open("data/test_data.pkl", "wb") as f:
    pickle.dump(test_data, f)

# Label encoding
label_encoder = LabelEncoder()
label_encoder.fit(data["classes"])

# Data generators
train_gen = DataGenerator(train_data, label_encoder)
val_gen = DataGenerator(val_data, label_encoder)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_data['classes']), y=train_data['classes'])
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Model initialization via selector
model = get_model(MODEL_NAME, (input_length, num_channels), num_classes)

if LOSS_TYPE == "focal":
    loss_fn = SparseCategoricalFocalLoss(gamma=FOCAL_GAMMA, class_weight=list(class_weights))
else:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn,
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor="val_loss", mode="min"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    MetricsPlotCallback(val_data=val_gen, log_frequency=10)
]

# MLflow training
if ENABLE_MLFLOW:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment = mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("input_length", input_length)
        mlflow.log_param("num_channels", num_channels)
        for fname in ['train_class_counts.csv', 'val_class_counts.csv', 'test_class_counts.csv']:
            if os.path.exists(fname):
                mlflow.log_artifact(fname)

        fit_kwargs = {"class_weight": class_weight_dict} if LOSS_TYPE == "weighted_ce" else {}
        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, **fit_kwargs)
        model.save(os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}.h5"))
        mlflow.log_artifact(os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}.h5"))
else:
    fit_kwargs = {"class_weight": class_weight_dict} if LOSS_TYPE == "weighted_ce" else {}
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, **fit_kwargs)
    model.save(os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}.h5"))
