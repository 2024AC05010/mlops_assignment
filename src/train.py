# train.py
# Model training script with MLflow tracking
# Initially tried SVM and XGBoost but they took too long to train
# Sticking with LR and RF for now - good baseline models
# TODO: Add hyperparameter tuning with Optuna or GridSearch when time permits

import mlflow
import mlflow.sklearn
import pickle
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import full_preprocessing_pipeline


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained model and returns a dict of metrics.
    Focusing on ROC-AUC as primary metric since it's binary classification
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"\n[RESULTS] {model_name}:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))

    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, model_name="Model", save_dir="reports/"):
    """Save confusion matrix plot as artifact for MLflow."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    save_path = os.path.join(save_dir, f"cm_{model_name.replace(' ', '_')}.png")
    fig.savefig(save_path)
    plt.close(fig)

    print(f"[INFO] Confusion matrix saved: {save_path}")
    return save_path


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train Logistic Regression with MLflow tracking.
    Tried different C values but 1.0 seemed to work best for this dataset
    Could experiment with different solvers (liblinear, saga) for comparison
    """
    exp_name = "Heart_Disease_Classifier"
    mlflow.set_experiment(exp_name)

    # These parameters worked well after some experimentation
    params = {
        "C": 1.0,
        "max_iter": 200,
        "solver": "lbfgs",
        "random_state": 42
    }

    with mlflow.start_run(run_name="LogisticRegression_Run"):
        print(f"[INFO] Training Logistic Regression with params: {params}")
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Cross-validation to check for overfitting
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_mean = round(np.mean(cv_scores), 4)
        cv_std = round(np.std(cv_scores), 4)
        print(f"[INFO] LR CV Accuracy: {cv_mean} (+/- {cv_std})")

        metrics, y_pred = evaluate_model(model, X_test, y_test, "Logistic Regression")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metric("cv_mean_accuracy", cv_mean)
        mlflow.log_metric("cv_std_accuracy", cv_std)
        for metric_name, metric_val in metrics.items():
            mlflow.log_metric(metric_name, metric_val)

        # Log confusion matrix plot
        cm_path = plot_confusion_matrix(y_test, y_pred, "Logistic Regression")
        mlflow.log_artifact(cm_path)

        # Log model
        mlflow.sklearn.log_model(model, "logistic_regression_model")

        print("[INFO] Logistic Regression run logged to MLflow.")

    return model, metrics


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train Random Forest with MLflow tracking.
    RF tends to perform better on this dataset due to non-linear relationships
    Still need to tune hyperparameters properly - current values are just starting point
    """
    exp_name = "Heart_Disease_Classifier"
    mlflow.set_experiment(exp_name)

    # These are conservative values to prevent overfitting
    # Might need to increase n_estimators for better performance
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 5,
        "random_state": 42
    }

    with mlflow.start_run(run_name="RandomForest_Run"):
        print(f"[INFO] Training Random Forest with params: {params}")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_mean = round(np.mean(cv_scores), 4)
        cv_std = round(np.std(cv_scores), 4)
        print(f"[INFO] RF CV Accuracy: {cv_mean} (+/- {cv_std})")

        metrics, y_pred = evaluate_model(model, X_test, y_test, "Random Forest")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metric("cv_mean_accuracy", cv_mean)
        mlflow.log_metric("cv_std_accuracy", cv_std)
        for metric_name, metric_val in metrics.items():
            mlflow.log_metric(metric_name, metric_val)

        # Log feature importance plot
        feature_importance_path = plot_feature_importance(model, "Random Forest")
        mlflow.log_artifact(feature_importance_path)

        # Log confusion matrix
        cm_path = plot_confusion_matrix(y_test, y_pred, "Random Forest")
        mlflow.log_artifact(cm_path)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        print("[INFO] Random Forest run logged to MLflow.")

    return model, metrics


def plot_feature_importance(model, model_name="RF", save_dir="reports/"):
    """
    Plot and save feature importance for tree-based models.
    This helps understand which features are driving predictions
    """
    os.makedirs(save_dir, exist_ok=True)

    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(importances)), sorted(importances, reverse=True))
    ax.set_title(f"Feature Importances - {model_name}")
    ax.set_xlabel("Importance Score")

    save_path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(save_path)
    plt.close(fig)

    return save_path


def save_best_model(lr_model, rf_model, lr_metrics, rf_metrics):
    """
    Compare both models and save the better one.
    Using ROC-AUC as the comparison metric since it's threshold-independent
    """
    os.makedirs("models", exist_ok=True)

    best_model = None
    best_name = ""

    print("\n[INFO] Comparing models based on ROC-AUC...")
    print(f"  Logistic Regression ROC-AUC: {lr_metrics['roc_auc']}")
    print(f"  Random Forest ROC-AUC: {rf_metrics['roc_auc']}")

    if lr_metrics["roc_auc"] >= rf_metrics["roc_auc"]:
        best_model = lr_model
        best_name = "LogisticRegression"
        print("[INFO] Selecting Logistic Regression as best model")
    else:
        best_model = rf_model
        best_name = "RandomForest"
        print("[INFO] Selecting Random Forest as best model")

    model_path = "models/best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"[INFO] Best model: {best_name} saved to {model_path}")
    return best_model, best_name


if __name__ == "__main__":
    print("=" * 60)
    print("STARTING MODEL TRAINING")
    print("=" * 60)

    # Load preprocessed data
    X_train, X_test, y_train, y_test, feature_cols = full_preprocessing_pipeline()

    # Train both models
    lr_model, lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)

    # Select and save best model
    best_model, best_name = save_best_model(lr_model, rf_model, lr_metrics, rf_metrics)

    print("=" * 60)
    print(f"[DONE] Training complete. Best model: {best_name}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the model: python src/predict.py")
    print("2. Start API: python app/app.py")
