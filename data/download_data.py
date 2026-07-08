# download_data.py
# Downloading the Heart Disease UCI dataset
# Reference: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
# Initially had issues with the URL - had to use the processed.cleveland.data file
# TODO: Add fallback URLs in case UCI is down

import os
import requests
import pandas as pd


def download_dataset(save_path="data/heart.csv"):
    """
    Downloads the Cleveland Heart Disease dataset from UCI repo
    and saves it as a cleaned CSV file.
    """
    # UCI dataset URL - using the processed Cleveland data
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/heart-disease/processed.cleveland.data"
    )

    # Column names based on UCI dataset documentation
    column_names = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]

    print("[INFO] Downloading dataset from UCI repository...")
    print(f"[INFO] URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        print(f"[INFO] Download successful! Status: {response.status_code}")
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        print("[ERROR] Please check your internet connection or try again later")
        raise

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save raw data first (for debugging if needed)
    raw_path = "data/raw_heart.data"
    with open(raw_path, "wb") as file:
        file.write(response.content)
    print(f"[INFO] Raw data saved to {raw_path}")

    # Load the data - UCI uses '?' for missing values
    try:
        df = pd.read_csv(raw_path, names=column_names, na_values="?")
        print(f"[INFO] Dataset loaded successfully | Shape: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to parse CSV: {e}")
        raise

    # Binary classification: target > 0 means disease present
    # The UCI dataset has values 0-4, we're converting to binary
    print("[INFO] Converting target to binary classification...")
    df["target"] = (df["target"] > 0).astype(int)
    
    # Check the distribution
    print(f"[INFO] Target distribution:")
    print(df["target"].value_counts())

    # Save as clean CSV
    df.to_csv(save_path, index=False)
    print(f"[INFO] Clean dataset saved at {save_path}")
    
    # Clean up raw file (optional - keeping for now for debugging)
    # os.remove(raw_path)
    
    return df


if __name__ == "__main__":
    print("="*60)
    print("HEART DISEASE DATASET DOWNLOAD")
    print("="*60)
    
    try:
        df = download_dataset()
        print("\n[SUCCESS] Dataset download and processing complete!")
        print(f"[INFO] You can now run: python src/train.py")
    except Exception as e:
        print(f"\n[FAILED] Could not download dataset: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Try accessing the URL directly in browser")
        print("3. Check if UCI repository is accessible")