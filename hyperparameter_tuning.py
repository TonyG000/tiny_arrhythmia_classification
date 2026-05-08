# hyperparameter_tuning.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pickle
import numpy as np
import keras_tuner
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from config import (
    MODEL_NAME, EPOCHS, CSV_FILENAME, LOSS_TYPE, FOCAL_GAMMA
)
from models.focal_loss import SparseCategoricalFocalLoss
# from models.model.cnn_bilstm_transformer_ensemble import EnsembleCNNBiLSTMTransformerModel
from models.model.cnn_attention_transformer import CNNTransformerModel, TransformerEncoderBlock
from utils.utils import (
    arrhythmia_classes_cpsc, samples_limit, input_length,
    num_channels, num_classes
)
from data_processing.data_generator import DataGenerator
from data_processing.data_loading import DataLoader
from data_processing.data_reduction import DataReducer
from data_processing.data_splitter import DataSplitter

# --- Config for tuning ---
# Set this to the model you want to tune after comparing results:
#   "EnsembleModel"       if ensemble outperforms CNN-Transformer
#   "CNNTransformerModel" if CNN-Transformer is better
TUNER_MODEL = "CNNTransformerModel"
TUNING_EPOCHS = 30
MAX_TRIALS = 30
TUNING_DIR = "models/hp_tuning"

# --- Prepare directories ---
os.makedirs(TUNING_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("models/final", exist_ok=True)

# --- Data loading (identical to train_custom_log_cpsc.py) ---
data_loader = DataLoader(CSV_FILENAME)
data = data_loader.load_data()

data = DataReducer(data, arrhythmia_classes_cpsc, max_samples=samples_limit).reduce_data()
train_data, val_data, test_data = DataSplitter(data).split()

with open("data/test_data.pkl", "wb") as f:
    pickle.dump(test_data, f)

label_encoder = LabelEncoder()
label_encoder.fit(data["classes"])

train_gen = DataGenerator(train_data, label_encoder, augment=True)
val_gen = DataGenerator(val_data, label_encoder, augment=False)

# --- Class weights (computed at module level so build_tunable_model can close over them) ---
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_data['classes']),
    y=train_data['classes']
)

# --- F1 Macro callback ---
class F1MacroCallback(tf.keras.callbacks.Callback):
    """Computes macro-F1 on the validation generator and writes to logs."""

    def __init__(self, val_generator):
        super().__init__()
        self.val_generator = val_generator

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for x_batch, y_batch in self.val_generator:
            preds = self.model.predict(x_batch, verbose=0)
            y_true.extend(y_batch)
            y_pred.extend(np.argmax(preds, axis=1))
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if logs is not None:
            logs['val_f1_macro'] = f1


# --- Tunable model builder ---
def build_tunable_model(hp):
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=5e-3, sampling='log')
    gamma = hp.Choice('focal_gamma', [1.0, 2.0, 3.0, 4.0])
    t_heads = hp.Choice('transformer_heads', [2, 4, 8])
    t_ff = hp.Choice('transformer_ff_dim', [128, 256, 512])

    # Build model with tunable transformer params
    inputs = tf.keras.layers.Input(shape=(input_length, num_channels))

    x = inputs
    # CNN backbone (fixed — identical to original)
    for filters, dropout in [(32, 0.2), (64, 0.3), (128, 0.3)]:
        x = tf.keras.layers.Conv1D(filters, 15, strides=2, padding='same',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv1D(filters, 15, strides=1, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling1D(2, strides=2, padding='same')(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    # Attention gate (fixed)
    attention = tf.keras.layers.Conv1D(128, 1, activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, attention])
    x = tf.keras.layers.MaxPooling1D(2, strides=2, padding='same', name='gradcam_target')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Tunable Transformer blocks
    x = TransformerEncoderBlock(embed_dim=128, num_heads=t_heads, ff_dim=t_ff, dropout=0.1)(x)
    x = TransformerEncoderBlock(embed_dim=128, num_heads=t_heads, ff_dim=t_ff, dropout=0.1)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Classifier (fixed)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    loss_fn = SparseCategoricalFocalLoss(gamma=gamma, class_weight=list(class_weights))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0),
        loss=loss_fn,
        metrics=['accuracy']
    )
    return model


# --- Tuner ---
tuner = keras_tuner.RandomSearch(
    hypermodel=build_tunable_model,
    objective=keras_tuner.Objective('val_f1_macro', direction='max'),
    max_trials=MAX_TRIALS,
    executions_per_trial=1,
    directory=TUNING_DIR,
    project_name=f"{TUNER_MODEL}_lead_masked",
    overwrite=False,
)

tuner.search_space_summary()

# --- Search ---
search_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, verbose=1
    ),
    F1MacroCallback(val_gen),
]

print(f"\nStarting random search: {MAX_TRIALS} trials × up to {TUNING_EPOCHS} epochs each")
tuner.search(
    train_gen,
    validation_data=val_gen,
    epochs=TUNING_EPOCHS,
    callbacks=search_callbacks,
)

# --- Report results ---
print("\n=== Top 5 trials ===")
best_hps_list = tuner.get_best_hyperparameters(num_trials=5)
for i, hps in enumerate(best_hps_list):
    print(f"\nTrial #{i+1}:")
    for key, val in hps.values.items():
        print(f"  {key}: {val}")

best_hps = best_hps_list[0]
print("\n=== Best hyperparameters ===")
for key, val in best_hps.values.items():
    print(f"  {key}: {val}")

# --- Retrain best config for full EPOCHS ---
print(f"\nRetraining best config for {EPOCHS} epochs...")
best_model = tuner.hypermodel.build(best_hps)

tuned_checkpoint = f"models/checkpoints/{TUNER_MODEL}_lead_masked_tuned_best.h5"
final_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        tuned_checkpoint, save_best_only=True, monitor='val_loss', mode='min'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    ),
    F1MacroCallback(val_gen),
]

best_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=final_callbacks,
)

tuned_final = f"models/final/{TUNER_MODEL}_lead_masked_tuned.h5"
best_model.save(tuned_final)
print(f"\nTuned model saved to: {tuned_final}")
print(f"Best checkpoint saved to: {tuned_checkpoint}")
print("\nRun test_cpsc.py with the tuned checkpoint to get final metrics.")
