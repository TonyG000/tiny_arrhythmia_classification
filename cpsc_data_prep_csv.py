import os
import csv

# Define the SNOMED to label mapping
snomed_to_label = {
    '164889003': 'AF', 
    '270492004': 'IAVB',
    '164909002': 'LBBB', 
    '59118001': 'RBBB',
    '426783006': 'SNR',  
    '429622005': 'STD',
    '164931005': 'STE',
    '164884008': 'PVC',
    '284470004': 'PAC',
}

# Base directory
# base_dir = "../../../common/Project_Arrhythmia/datasets/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/"

base_dir = "/home/g6/cpsc_2018_data/"

groups = [f"g{i}" for i in range(1, 8)]  # g1 to g7
output_csv = "cpsc_ecg_data_3.csv"
# output_csv = "/root/tiny_arrhythmia_classification/csv_files/cpsc_ecg_data_3.csv"
output_csv = "/home/g6/tiny_arrhythmia_classification/csv_files/cpsc_ecg_data_3.csv"
# List to store extracted data
data = []

# Loop through directories
for group in groups:
    group_path = os.path.join(base_dir, group)
    
    for file in os.listdir(group_path):
        if file.endswith(".hea"):
            hea_path = os.path.join(group_path, file)
            mat_path = hea_path.replace(".hea", ".mat")
            
            # Read the .hea file
            with open(hea_path, "r") as f:
                lines = f.readlines()
                
            # Extract Dx line
            dx_line = next((line for line in lines if line.startswith("# Dx:")), None)
            unique_dx_codes = set()
            if dx_line:
                dx_codes = dx_line.strip().split(": ")[1].split(",")
                # Add DX codes to the unique set
                unique_dx_codes.update(dx_codes)
                labels = [snomed_to_label[code] for code in dx_codes if code in snomed_to_label]
                dx_codes.append(dx_codes)
                if labels:
                    data.append([mat_path, " ".join(labels)])
            # Print unique DX codes
            #print("Unique DX Codes:", unique_dx_codes)

# Write to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_path", "classes"])
    writer.writerows(data)

print(f"CSV file saved: {output_csv}")

import pandas as pd

# df = pd.read_csv("cpsc_ecg_data_3.csv")

# # Exploding multi-labels
# df["classes"] = df["classes"].str.split(" ")  # Convert to list
# df = df.explode("classes").reset_index(drop=True)  # Explode multi-labels

# df.to_csv("cpsc_ecg_data_3.csv")
df = pd.read_csv(output_csv)

# Exploding multi-labels
df["classes"] = df["classes"].str.split(" ")  # Convert to list
df = df.explode("classes").reset_index(drop=True)  # Explode multi-labels

df.to_csv(output_csv, index=False)