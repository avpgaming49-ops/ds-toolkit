# Model metrics and evaluation helpers

# Regression and Classification evaluation metrics
import numpy as np
import pandas as pd
from typing import Union, Optional

ArrayLike = Union[pd.Series, np.ndarray, list]


def evaluate_regression(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_features: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate MAE, MSE, RMSE, R2, and (optionally) Adjusted R2 for regression models.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Shape mismatch: y_true has length {len(y_t)}, but y_pred has length {len(y_p)}."
        )

    if len(y_t) == 0:
        raise ValueError("Input arrays cannot be empty.")

    # Drop pairs where either true or predicted is NaN
    valid_mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    if not np.all(valid_mask):
        y_t = y_t[valid_mask]
        y_p = y_p[valid_mask]
        if len(y_t) == 0:
            raise ValueError("All values contain NaNs; cannot calculate regression metrics.")

    mae = np.mean(np.abs(y_t - y_p))
    mse = np.mean((y_t - y_p) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)

    metrics = {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }

    if n_features is not None:
        n = len(y_t)
        if n > n_features + 1:
            adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
            metrics["Adjusted_R2"] = round(float(adj_r2), 4)
        else:
            metrics["Adjusted_R2"] = None

    return pd.DataFrame([metrics], index=["Regression Metrics"])


def evaluate_classification(y_true: ArrayLike, y_pred: ArrayLike) -> pd.DataFrame:
    """
    Calculate Accuracy, Precision, Recall, F1-Score, and Confusion Matrix counts
    for binary classification.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Shape mismatch: y_true has length {len(y_t)}, but y_pred has length {len(y_p)}."
        )

    if len(y_t) == 0:
        raise ValueError("Input arrays cannot be empty.")

    # Drop pairs where either is NaN
    valid_mask = ~pd.isna(y_t) & ~pd.isna(y_p)
    if not np.all(valid_mask):
        y_t = y_t[valid_mask]
        y_p = y_p[valid_mask]
        if len(y_t) == 0:
            raise ValueError("All values contain NaNs; cannot calculate classification metrics.")

    # Convert predictions to integers (handles booleans or thresholded floats safely)
    try:
        y_t = y_t.astype(int)
        y_p = y_p.astype(int)
    except (ValueError, TypeError):
        raise TypeError("Classification targets and predictions must be convertible to integers (0 and 1).")

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    total = len(y_t)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1_Score": round(float(f1), 4),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }

    return pd.DataFrame([metrics], index=["Classification Metrics"])