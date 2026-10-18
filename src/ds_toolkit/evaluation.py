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
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_t - y_p))
    mse = np.mean((y_t - y_p) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    metrics = {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }

    if n_features is not None and len(y_t) > n_features + 1:
        n = len(y_t)
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
        metrics["Adjusted_R2"] = round(adj_r2, 4)

    return pd.DataFrame([metrics], index=["Regression Metrics"])


def evaluate_classification(y_true: ArrayLike, y_pred: ArrayLike) -> pd.DataFrame:
    """
    Calculate Accuracy, Precision, Recall, F1-Score, and Confusion Matrix counts
    for binary classification (0 and 1).
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

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
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }

    return pd.DataFrame([metrics], index=["Classification Metrics"])