# predict.py
# Inference utility - loads model + scaler and predicts on new data
# Had some issues with feature alignment during testing - fixed by reindexing
# TODO: Add input validation and error handling for edge cases
# TODO: Add batch prediction support for multiple patients

import pickle
import numpy as np
import pandas as pd


def load_artifacts(model_path="models/best_model.pkl",
                   scaler_path="models/scaler.pkl",
                   cols_path="models/feature_columns.pkl"):
    """
    Load the saved model, scaler, and feature columns.
    This was tricky - needed to ensure all three files are loaded in correct order
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"[INFO] Loaded model from {model_path}")

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"[INFO] Loaded scaler from {scaler_path}")

        with open(cols_path, "rb") as f:
            feature_cols = pickle.load(f)
        print(f"[INFO] Loaded feature columns from {cols_path}")
        print(f"[DEBUG] Number of features: {len(feature_cols)}")

        return model, scaler, feature_cols
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load artifacts: {e}")
        print("Make sure to run train.py first!")
        raise


def preprocess_input(input_dict, feature_cols, scaler):
    """
    Takes raw JSON input dict, aligns to expected features,
    applies encoding and scaling.
    Note: Need to handle cases where categorical values might be different from training
    """
    # Convert input to DataFrame
    df_input = pd.DataFrame([input_dict])

    # One-hot encode same categorical cols as training
    # This must match exactly what was done during training
    categorical_cols = ["cp", "restecg", "slope", "thal"]
    df_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

    # Align to training columns - fill missing with 0
    # This is crucial - if a categorical value wasn't seen during training,
    # it will create a new column that needs to be handled
    df_aligned = df_input.reindex(columns=feature_cols, fill_value=0)

    # Debug: check if any columns are missing
    missing_cols = set(feature_cols) - set(df_aligned.columns)
    if missing_cols:
        print(f"[WARNING] Missing columns after alignment: {missing_cols}")

    # Apply scaling
    scaled = scaler.transform(df_aligned)
    return scaled


def run_prediction(input_dict):
    """
    Main prediction function called by the API.
    Returns prediction label and confidence score.
    """
    print(f"[INFO] Running prediction for input: {input_dict}")

    model, scaler, feature_cols = load_artifacts()

    processed = preprocess_input(input_dict, feature_cols, scaler)

    prediction = int(model.predict(processed)[0])
    confidence = round(float(model.predict_proba(processed)[0][prediction]), 4)

    label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    result = {
        "prediction": prediction,
        "label": label,
        "confidence": confidence
    }

    print(f"[INFO] Prediction result: {result}")
    return result


if __name__ == "__main__":
    # Sample test input - using values from the dataset
    print("="*60)
    print("TESTING PREDICTION FUNCTION")
    print("="*60)
    
    sample_input = {
        "age": 54, "sex": 1, "cp": 2, "trestbps": 130,
        "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
        "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 2
    }
    
    try:
        output = run_prediction(sample_input)
        print(f"\n[PREDICTION] {output}")
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        print("Make sure you've trained the model first!")