import pandas as pd
import chardet
import os

# Path to your dataset
file_path = " "  # Adjust the path as necessary

# Detect encoding
with open(file_path, "rb") as f:
    encoding = chardet.detect(f.read())['encoding']

# Load data
data = pd.read_csv(file_path, encoding=encoding)

# Keep only decoded messages that are not UNDECODABLE
decoded_data = data[data['decoded'] != "UNDECODABLE"]

# Count T and R labels
label_counts = decoded_data['label'].value_counts()

print(f"Counts of T and R in decoded column (excluding UNDECODABLE):\n{label_counts}")