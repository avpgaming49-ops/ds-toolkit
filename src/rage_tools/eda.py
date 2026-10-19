# Statistics, correlations, outliers

# Dataset overview, numeric/categorical summaries, and outlier detection
import pandas as pd
import numpy as np
from typing import Dict, Any


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive column-by-column health report of the DataFrame
    (dtype, non-null count, missing count/%, unique values, and sample value).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if df.empty or len(df) == 0:
        return pd.DataFrame(
            columns=[
                "dtype",
                "non_null_count",
                "missing_count",
                "missing_pct",
                "unique_values",
                "sample_value",
            ]
        )

    row_count = len(df)
    missing_counts = df.isna().sum()

    overview = pd.DataFrame({
        "dtype": df.dtypes.astype(str),  # Converts dtypes to clean strings (prevents JSON crashes)
        "non_null_count": df.notna().sum(),
        "missing_count": missing_counts,
        "missing_pct": ((missing_counts / row_count) * 100).round(2),
        "unique_values": df.nunique(),
        "sample_value": [
            df[col].dropna().iloc[0] if not df[col].dropna().empty else None 
            for col in df.columns
        ]
    })
    
    overview.index.name = "column_name"
    return overview


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended descriptive statistics for numeric columns, including
    skewness, kurtosis, and Interquartile Range (IQR).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    num_df = df.select_dtypes(include=["number"])
    if num_df.empty or len(num_df.columns) == 0:
        return pd.DataFrame()

    # Handle duplicate column names if present to prevent alignment crashes
    if not num_df.columns.is_unique:
        num_df = num_df.copy()
        num_df.columns = pd.io.formats.format.clean_column_indexes(num_df) # or deduplicate safely

    stats = num_df.describe().T
    stats["skewness"] = num_df.skew().round(3)
    stats["kurtosis"] = num_df.kurt().round(3)
    stats["iqr"] = (stats["75%"] - stats["25%"]).round(3)
    
    stats.index.name = "column_name"
    return stats.round(3)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary of categorical/object/string columns showing unique classes,
    most frequent value (top), and its percentage share.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    cat_df = df.select_dtypes(include=["object", "category", "string"])
    if cat_df.empty or len(cat_df.columns) == 0:
        return pd.DataFrame()

    records = []
    seen_cols = set()
    
    for col in cat_df.columns:
        # Handle duplicate column names gracefully
        base_col = col
        counter = 1
        while col in seen_cols:
            col = f"{base_col}_{counter}"
            counter += 1
        seen_cols.add(col)

        non_null = cat_df[base_col].dropna()
        top_val = non_null.mode().iloc[0] if not non_null.empty else None
        top_freq = (non_null == top_val).sum() if top_val is not None else 0
        row_count = len(df)

        records.append({
            "column": col,
            "unique_classes": cat_df[base_col].nunique(),
            "top_value": top_val,
            "top_freq": top_freq,
            "top_pct": round((top_freq / row_count) * 100, 2) if row_count > 0 else 0.0
        })

    summary_df = pd.DataFrame(records)
    if not summary_df.empty and "column" in summary_df.columns:
        summary_df.set_index("column", inplace=True)
        summary_df.index.name = "column_name"

    return summary_df


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in numeric columns using the 1.5 * IQR rule.
    Returns a summary table of outlier counts, percentages, and bounds.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    num_df = df.select_dtypes(include=["number"])
    if num_df.empty or len(num_df.columns) == 0 or len(df) == 0:
        return pd.DataFrame(columns=[
            "column", "outlier_count", "outlier_pct", "lower_bound", "upper_bound"
        ]).set_index("column")

    records = []
    total_rows = len(df)

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
        outlier_count = len(outliers)
        outlier_pct = round((outlier_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        records.append({
            "column": col,
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        })

    if not records:
        return pd.DataFrame(columns=[
            "column", "outlier_count", "outlier_pct", "lower_bound", "upper_bound"
        ]).set_index("column")

    summary_df = pd.DataFrame(records)
    summary_df.set_index("column", inplace=True)
    summary_df.index.name = "column_name"
    return summary_df