# Missing values, duplicates, column names
import pandas as pd
from typing import Optional, List, Union


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame column names to snake_case."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def remove_duplicates(
    df: pd.DataFrame, 
    subset: Optional[List[str]] = None, 
    keep: str = "first"
) -> pd.DataFrame:
    """Drop duplicate rows and reset the index."""
    df = df.copy()
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    return df


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of columns containing missing values."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct.round(2),
        "dtype": df.dtypes
    })
    return summary[summary["missing_count"] > 0].sort_values(
        by="missing_pct", ascending=False
    )


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    fill_value: Optional[Union[str, int, float]] = None
) -> pd.DataFrame:
    """
    Fill missing values automatically across numeric and categorical columns.
    numeric_strategy: 'mean', 'median', or 'zero'
    categorical_strategy: 'mode' or 'constant'
    """
    df = df.copy()
    
    # 1. Handle Numeric Columns
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isna().sum() > 0:
            if numeric_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif numeric_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif numeric_strategy == "zero":
                df[col] = df[col].fillna(0)

    # 2. Handle Categorical / Object Columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            if categorical_strategy == "mode" and not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            elif categorical_strategy == "constant":
                df[col] = df[col].fillna(fill_value if fill_value is not None else "Unknown")

    return df