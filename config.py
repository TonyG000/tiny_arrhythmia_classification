# config.py

# Model Selection
# MODEL_NAME = "EnsembleModel"        # Feature-level fusion: CNN-BiLSTM + CNN-Transformer
MODEL_NAME = "CNNTransformerModel"
# MODEL_NAME = "EnhancedCNNModel"

# # MLflow
# ENABLE_MLFLOW = True
# MLFLOW_URI = "http://192.168.95.103:5005"
# EXPERIMENT_NAME = "Lightweight_CNN_Attentation_BiLSTM"
# RUN_NAME = "CPSC_Data_Lightweight_CNN_Attentation_BiLSTM"
# MLflow
ENABLE_MLFLOW = False
MLFLOW_URI = ""
EXPERIMENT_NAME = "Lightweight_CNN_Attentation_BiLSTM"
RUN_NAME = "CPSC_Data_Lightweight_CNN_Attentation_BiLSTM"
 

# Base path for the dataset
# DATASET_PATH = "../../../../common/Project_Arrhythmia/datasets/"
DATASET_PATH = "/home/g6/cpsc_2018_data/"
# CSV_PATH = "../../../../common/Project_Arrhythmia/datasets/csv_files/"
CSV_PATH = "/home/g6/tiny_arrhythmia_classification/csv_files/"
# Define paths for saving models, results, and plots
BASE_MODEL_DIR = "models"
CSV_FILENAME = "cpsc_ecg_data_3.csv"


# Training Settings
EPOCHS = 100
LEARNING_RATE = 0.001
LOSS_TYPE = "focal"   # "weighted_ce" or "focal"
FOCAL_GAMMA = 2.0
MODEL_SAVE_PATH = "models/final"
CHECKPOINT_PATH = "models/checkpoints/{}_best.h5".format(MODEL_NAME)
BATCH_SIZE  = 32


# General settings
SAMPLING_RATE = 250
TARGET_LENGTH = 15000

# Mode: "multi" for 12-lead ECG, "single" for specific lead
LEAD_MODE = "multi"  # or "single"

# Use only if LEAD_MODE is "single"
DESIRED_LEAD = 1  # Lead number (1 to 12)

# Image or ECG signal Dump
IMAGE_DUMP = 0

# Random Lead Masking (training augmentation for wearable generalization)
# Every training sample is randomly assigned 1, 3, or 12 active leads.
LEAD_MASKING = True

# Rare-class augmentation (applied only on training, only to rare-class samples)
# Goal: improve PR curves on under-represented classes (LBBB, STE).
RARE_CLASS_AUG = True
RARE_CLASSES = ["LBBB", "STE"]
# Oversample rare classes up to the median count of the remaining train classes.
# Each duplicated sample is regenerated with noise + time shift each epoch so
# duplicates are not identical.
RARE_OVERSAMPLE_TO_MEDIAN = True
RARE_AUG_NOISE_STD = 0.02      # fraction of signal std added as Gaussian noise
RARE_AUG_MAX_SHIFT_FRAC = 0.05  # max circular shift as fraction of signal length




