"""
data_split.py

This script splits the MIAS breast cancer dataset into training, validation, and test subsets.
It also organizes the corresponding images into separate folders for each split.

Steps performed:
1. Reads the metadata CSV.
2. Filters entries with missing class labels.
3. Generates full image filenames.
4. Performs stratified splitting.
5. Saves split metadata as CSV files.
6. Copies images into corresponding split directories.
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === Step 1: Define paths ===
csv_path = "/home/senanur/deep_learning_homework/dataset/mias_info.csv"  # Path to the metadata CSV file
images_dir = "/home/senanur/deep_learning_homework/dataset/MIAS/"        # Directory containing all PNG images
output_dir = "dataset_split/"                                             # Output directory for split data

# === Step 2: Load and clean metadata ===
# Load the dataset metadata
df = pd.read_csv(csv_path)

# Remove rows where the 'CLASS' column is empty (missing label)
df = df[df['CLASS'].notna()]

# Create the 'filename' column by converting REFNUM to lowercase and appending '.png'
df['filename'] = df['REFNUM'].str.lower() + ".png"

# === Step 3: Stratified split into train, validation, and test sets ===
# First split: 80% train, 20% temp (which will be split again)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['CLASS'], random_state=42)

# Second split: split temp into 50% val and 50% test → resulting in 10% val, 10% test
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['CLASS'], random_state=42)

# Organize the splits in a dictionary for easier handling
splits = {'train': train_df, 'val': val_df, 'test': test_df}

# === Step 4: Save split metadata as CSVs ===
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save each DataFrame to a CSV file
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

# === Step 5: Copy image files into corresponding folders ===
# Loop over each split (train/val/test) and copy files into subfolders
for split_name, split_df in splits.items():
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)  # Create split directory if it doesn't exist

    # Copy each image to the correct split folder
    for _, row in split_df.iterrows():
        src_path = os.path.join(images_dir, row['filename'])
        dst_path = os.path.join(split_dir, row['filename'])

        # Only copy if file exists
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"[Warning] Image not found: {src_path}")
