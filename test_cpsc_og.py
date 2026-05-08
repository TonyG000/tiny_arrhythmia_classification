# test_cpsc.py
import os
# Set GPU visibility (if using GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing.data_generator import DataGenerator
from utils.model_selector import get_model  # Import the get_model function
from utils.utils import input_length, num_channels, num_classes, model_results_path
import mlflow
from config import ENABLE_MLFLOW, MLFLOW_URI, EXPERIMENT_NAME, RUN_NAME, MODEL_NAME,LEARNING_RATE,IMAGE_DUMP # Import MODEL_NAME

# MLflow setup
if ENABLE_MLFLOW:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print("Experiment does not exist! Creating a new experiment.")
        experiment = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        print(f"Logging to existing Experiment: {experiment.experiment_id}")

    # Enable MLflow autologging for TensorFlow
    mlflow.tensorflow.autolog()

# Load the saved test data
test_data_path = "data/test_data.pkl"

with open(test_data_path, "rb") as f:
    test_data = pickle.load(f)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(test_data["classes"])

# Initialize data generator for testing
test_generator = DataGenerator(test_data, label_encoder)

# Load the model architecture using the selector
input_shape = (input_length, num_channels)
model = get_model(MODEL_NAME, input_shape, num_classes)

# Determine the model's class name for filenames and titles
model_class_name = model.name  # TensorFlow models usually have a 'name' attribute

# Load the saved weights (from the checkpoint saved during training)
checkpoint_path = f"models/checkpoints/{MODEL_NAME}_best.h5" # Assuming your checkpoint path uses MODEL_NAME
#checkpoint_path  = "../../../../common/Project_Arrhythmia/110642159774068834/3ff31e2c1f854e8288d05e393ad985ef/artifacts/EnhancedCNNModel.h5"

model.load_weights(checkpoint_path)

# Compile the model (ensure it is compiled before testing)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

if ENABLE_MLFLOW:
    with mlflow.start_run(run_name=RUN_NAME + "_test") as run:
        # Evaluate the model on the test data
        test_loss, test_acc = model.evaluate(test_generator)
        # Log test results
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_acc}")

        # Initialize true and predicted labels
        y_true = []
        y_pred = []
        misclassified_samples = []

        # Directory to save misclassified samples
        misclassified_dir = os.path.join(model_results_path, f"misclassified_samples_{MODEL_NAME}_60sec") # Using MODEL_NAME in directory
        os.makedirs(misclassified_dir, exist_ok=True)

        # Get the true labels and predictions
        for batch_index, (batch_data, batch_labels) in enumerate(test_generator):
            # Ensure the batch is not empty
            if batch_data.shape[0] == 0:
                break

            # Append the true labels to y_true
            y_true.extend(batch_labels)

            # Predict for the batch
            batch_predictions = model.predict(batch_data, batch_size=batch_data.shape[0])  # Use dynamic batch size
            batch_pred_classes = np.argmax(batch_predictions, axis=1)
            y_pred.extend(batch_pred_classes)

            # Identify misclassified samples
            for i in range(len(batch_labels)):
                if batch_labels[i] != batch_pred_classes[i]:
                    true_label = label_encoder.inverse_transform([batch_labels[i]])[0]
                    predicted_label = label_encoder.inverse_transform([batch_pred_classes[i]])[0]
                    misclassified_samples.append({
                        'Input_data': batch_data[i],
                        'True_label': true_label,
                        'Predicted_label': predicted_label
                    })

                    if(IMAGE_DUMP == 1):
                        # Define the filename
                        filename = f"sample_{batch_index * test_generator.batch_size + i}_true_{true_label}_pred_{predicted_label}.png"
                        filepath = os.path.join(misclassified_dir, filename)

                        # Define lead names (standard 12-lead ECG names)
                        lead_names = [
                            'I', 'II', 'III',
                            'aVR', 'aVL', 'aVF',
                            'V1', 'V2', 'V3',
                            'V4', 'V5', 'V6'
                        ]

                        # Generate time values for the x-axis (in seconds)
                        time = np.arange(0, batch_data[i].shape[0]) / 250.0  # 250 Hz sampling rate

                        # Create a figure with 12 subplots (one for each lead)
                        plt.figure(figsize=(15, 20))  # Adjust figure size for better visualization

                        # Plot each lead in a separate subplot
                        for channel_index in range(12):
                            plt.subplot(12, 1, channel_index + 1)  # Create subplots in a 12x1 grid
                            signal = batch_data[i][:, channel_index]  # Extract the signal for the current lead
                            plt.plot(time, signal, color='b', linewidth=1)  # Plot the signal
                            plt.ylabel(lead_names[channel_index], fontsize=10)  # Add lead name as y-label
                            plt.grid(True)  # Add grid for better readability
                            if channel_index == 11:  # Add x-label only for the last subplot
                                plt.xlabel('Time (seconds)', fontsize=12)

                        # Add a main title for the entire plot
                        plt.suptitle(f'True: {true_label}, Predicted: {predicted_label}', fontsize=14, y=1.02)

                        # Save the plot
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
                        plt.close()

                        print(f"Saved misclassified ECG plot: {filepath}")


        # Convert y_true and y_pred to numpy arrays
        y_true = np.array(y_true)
        y_pred_classes = np.array(y_pred)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)

        # Print confusion matrix
        print("Confusion Matrix:")
        print(cm)

        # Classification Report
        report = classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)
        print(report)
        # Plot and save the heatmap for the raw confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f"{model_class_name} Confusion Matrix (Raw)")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.tight_layout()
        plt.savefig(f"{model_results_path}/{model_class_name}_confusion_matrix_raw.png")
        plt.close()
        if ENABLE_MLFLOW:
            mlflow.log_artifact(f"{model_results_path}/{model_class_name}_confusion_matrix_raw.png")

        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot and save the heatmap for the normalized confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f"{model_class_name} Confusion Matrix (Normalized)")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.tight_layout()
        plt.savefig(f"{model_results_path}/{model_class_name}_confusion_matrix_normalized.png")
        plt.close()
        if ENABLE_MLFLOW:
            mlflow.log_artifact(f"{model_results_path}/{model_class_name}_confusion_matrix_normalized.png")


        # Save classification report
        classification_report_path = os.path.join(model_results_path, "classification_report.txt")
        with open(classification_report_path, "w") as f:
            f.write(report)
        if ENABLE_MLFLOW:
            mlflow.log_artifact(classification_report_path)

        # Save misclassified samples data
        misclassified_samples_path = os.path.join(model_results_path, f"misclassified_samples_{MODEL_NAME}.pkl")
        with open(misclassified_samples_path, "wb") as f:
            pickle.dump(misclassified_samples, f)
        if ENABLE_MLFLOW:
            mlflow.log_artifact(misclassified_samples_path)

         # Predict probabilities
        y_pred_prob = model.predict(test_generator, batch_size=32)  # Adjust batch_size as needed

        # Compute AUC scores
        auc_macro = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')
        mlflow.log_metric("AUC_macro", auc_macro)

        # Prepare to save AUCs in a file
        auc_log_path = os.path.join(model_results_path, "auc_scores.txt")
        with open(auc_log_path, "w") as f:
            f.write(f"AUC (Macro-average): {auc_macro:.4f}\n")

            for class_idx in range(num_classes):
                y_true_class = (y_true == class_idx).astype(int)
                y_pred_class = y_pred_prob[:, class_idx]
                class_auc = roc_auc_score(y_true_class, y_pred_class)
                mlflow.log_metric(f"AUC_class_{label_encoder.classes_[class_idx]}", class_auc)
                f.write(f"AUC for class {label_encoder.classes_[class_idx]}: {class_auc:.4f}\n")

        # Log AUC file to MLflow
        if ENABLE_MLFLOW:
            mlflow.log_artifact(auc_log_path)

else:
    print("MLflow is disabled. Skipping MLflow logging for test results.")
    # Evaluate the model on the test data (without MLflow)
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

    # Basic evaluation without MLflow logging for plots
    y_true = np.concatenate([y for x, y in test_generator], axis=0)
    y_pred = np.argmax(model.predict(test_generator), axis=1)
    # You can still generate and save plots locally if needed
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"{model.name} Test Confusion Matrix (Local)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(f"{model_results_path}/{model.name}_test_confusion_matrix_local.png")
    plt.close()