# Statistics, correlations, outliers

# Dataset overview, numeric/categorical summaries, and outlier detection
import pandas as pd
import numpy as np
from typing import Dict, Any


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive column-by-column health report of the DataFrame
    (dtype, missing count/%, unique values, and sample value).
    """
    overview = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null_count": df.notna().sum(),
        "missing_count": df.isna().sum(),
        "missing_pct": ((df.isna().sum() / len(df)) * 100).round(2),
        "unique_values": df.nunique(),
        "sample_value": [df[col].dropna().iloc[0] if not df[col].dropna().empty else None for col in df.columns]
    })
    return overview


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended descriptive statistics for numeric columns, including 
    skewness, kurtosis, and Interquartile Range (IQR).
    """
    num_df = df.select_dtypes(include=["number"])
    if num_df.empty:
        return pd.DataFrame()

    stats = num_df.describe().T
    stats["skewness"] = num_df.skew().round(3)
    stats["kurtosis"] = num_df.kurt().round(3)
    stats["iqr"] = (stats["75%"] - stats["25%"]).round(3)
    return stats.round(3)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary of categorical/object columns showing unique classes,
    most frequent value (top), and its percentage share.
    """
    cat_df = df.select_dtypes(include=["object", "category"])
    if cat_df.empty:
        return pd.DataFrame()

    records = []
    for col in cat_df.columns:
        non_null = cat_df[col].dropna()
        top_val = non_null.mode().iloc[0] if not non_null.empty else None
        top_freq = (non_null == top_val).sum() if top_val is not None else 0
        
        records.append({
            "column": col,
            "unique_classes": cat_df[col].nunique(),
            "top_value": top_val,
            "top_freq": top_freq,
            "top_pct": round((top_freq / len(df)) * 100, 2) if len(df) > 0 else 0.0
        })

    return pd.DataFrame(records).set_index("column")


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in numeric columns using the 1.5 * IQR rule.
    Returns a summary table of outlier counts, percentages, and bounds.
    """
    num_df = df.select_dtypes(include=["number"])
    records = []

    for col in num_df.columns:
        series = num_df[col].dropna()
        if series.empty:
            continue
            
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        records.append({
            "column": col,
            "outlier_count": len(outliers),
            "outlier_pct": round((len(outliers) / len(df)) * 100, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        })

    return pd.DataFrame(records).set_index("column")