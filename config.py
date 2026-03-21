# config.py

# Model Selection
MODEL_NAME = "CNNTransformerModel"  # Options: EnhancedCNNModel, ResNet1DModel

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




