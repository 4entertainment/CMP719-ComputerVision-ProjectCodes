"""
dataset_download.py

This script downloads the MIAS breast cancer dataset using the KaggleHub library.
Make sure you have KaggleHub configured and authenticated before running this script.
"""

import kagglehub

# Download the latest version of the dataset from Kaggle
dataset_path = kagglehub.dataset_download("orvile/mias-dataset")

print("Dataset successfully downloaded to:", dataset_path)
