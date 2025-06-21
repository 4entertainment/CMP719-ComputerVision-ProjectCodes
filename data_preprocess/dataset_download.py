"""
dataset_download.py

This script downloads the MIAS breast cancer dataset from KaggleHub.
Ensure you have KaggleHub configured properly before running.
"""

import kagglehub

# Download the latest version of the MIAS dataset
dataset_path = kagglehub.dataset_download("orvile/mias-dataset")

print("Dataset successfully downloaded to:", dataset_path)
