# preprocess.py
# Data preprocessing pipeline for heart disease dataset
# Initially tried mean imputation but switched to median - seemed to work better for skewed features
# Still need to experiment with different encoding strategies for categorical variables

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(file_path="data/heart.csv"):
    """Load the CSV dataset into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"[INFO] Loaded data | Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        print("Run download_data.py first!")
        raise


def handle_missing_values(df):
    """
    Fill missing values with median for numeric columns.
    The UCI dataset has a few '?' values in 'ca' and 'thal'.
    Tried mean first but median gave better results for this dataset
    """
    df = df.copy()

    missing_before = df.isnull().sum().sum()
    print(f"[INFO] Missing values before cleaning: {missing_before}")

    # Handle the '?' values that might still be in the data
    # This was a tricky part - had to convert them to NaN first
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  -> Filled '{col}' with median = {median_val}")

    print(f"[INFO] Missing values after cleaning: {df.isnull().sum().sum()}")
    return df


def encode_features(df):
    """
    Encode categorical features using one-hot encoding.
    Categorical cols: cp, restecg, slope, thal
    Note: drop_first=True to avoid multicollinearity
    Could try label encoding for ordinal features if needed later
    """
    df = df.copy()

    categorical_cols = ["cp", "restecg", "slope", "thal"]
    
    # One-hot encoding - this increases dimensionality but works well for tree models
    # For logistic regression, might want to try different approach
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print(f"[INFO] Shape after encoding: {df.shape}")
    print(f"[INFO] New columns from encoding: {len(categorical_cols)} -> {df.shape[1] - len([c for c in df.columns if c not in categorical_cols])} columns")
    return df


def split_features_target(df, target_col="target"):
    """Split DataFrame into features (X) and target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"[INFO] Features shape: {X.shape} | Target shape: {y.shape}")
    return X, y


def scale_features(X_train, X_test, scaler_path="models/scaler.pkl"):
    """
    Fit StandardScaler on training data and transform both sets.
    Save the fitted scaler for inference later.
    Note: StandardScaler assumes Gaussian distribution - might not be ideal for all features
    MinMaxScaler could be alternative for non-normal distributions
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[INFO] Scaler saved to {scaler_path}")
    print(f"[DEBUG] Train scaled - mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    return X_train_scaled, X_test_scaled, scaler


def full_preprocessing_pipeline(file_path="data/heart.csv", test_size=0.2, random_state=42):
    """
    End-to-end preprocessing pipeline.
    Returns train/test splits with scaled features.
    """
    print("="*60)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*60)
    
    df_raw = load_data(file_path)
    df_clean = handle_missing_values(df_raw)
    df_encoded = encode_features(df_clean)

    X, y = split_features_target(df_encoded)

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[INFO] Train size: {X_train.shape} | Test size: {X_test.shape}")
    print(f"[INFO] Class distribution in train: {np.bincount(y_train)}")
    print(f"[INFO] Class distribution in test: {np.bincount(y_test)}")

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # Save column names for inference reproducibility
    # This is crucial - need to ensure same columns during prediction
    feature_cols = list(X.columns)
    cols_path = "models/feature_columns.pkl"
    with open(cols_path, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"[INFO] Feature columns saved to {cols_path}")
    print(f"[DEBUG] Total features: {len(feature_cols)}")

    print("="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)

    return X_train_sc, X_test_sc, y_train, y_test, feature_cols


if __name__ == "__main__":
    # Test the pipeline
    full_preprocessing_pipeline()