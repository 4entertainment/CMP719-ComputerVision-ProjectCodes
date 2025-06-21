"""
data_split.py

Splits the MIAS dataset into training, validation, and test sets
and saves them along with the corresponding images into subfolders.
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === Configuration ===
csv_path = "/home/senanur/deep_learning_homework/dataset/mias_info.csv"  # CSV metadata path
images_dir = "/home/senanur/deep_learning_homework/dataset/MIAS/"        # Directory containing PNG images
output_dir = "dataset_split/"                                             # Output directory for splits

# === Read metadata ===
df = pd.read_csv(csv_path)

# Filter out rows with missing CLASS labels
df = df[df['CLASS'].notna()]

# Generate filename column based on REFNUM field
df['filename'] = df['REFNUM'].str.lower() + ".png"

# === Perform stratified split ===
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['CLASS'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['CLASS'], random_state=42)

splits = {'train': train_df, 'val': val_df, 'test': test_df}

# Save CSVs for each split
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

# === Copy image files into split folders ===
for split_name, split_df in splits.items():
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for _, row in split_df.iterrows():
        src_path = os.path.join(images_dir, row['filename'])
        dst_path = os.path.join(split_dir, row['filename'])

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"[Warning] Image not found: {src_path}")
