# Outlier capping, feature scaling, categorical encoding, train-test split, and auto-pipeline
import re
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from ds_toolkit.data_cleaning import (
    clean_column_names,
    remove_duplicates,
    handle_missing_values,
)
from ds_toolkit.utils import timer, reduce_memory_usage


def cap_outliers_iqr(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Cap (clip) extreme outliers in numeric columns to the lower and upper IQR bounds
    instead of dropping rows.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def scale_numeric_features(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    method: str = "standard"
) -> pd.DataFrame:
    """
    Scale numeric features using 'standard' (Z-score) or 'minmax' (0 to 1) scaling.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    for col in columns:
        if method == "standard":
            std = df[col].std()
            if std != 0 and not np.isnan(std):
                df[col] = (df[col] - df[col].mean()) / std
        elif method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            raise ValueError("method must be 'standard' or 'minmax'")

    return df


def encode_categorical(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    drop_first: bool = True
) -> pd.DataFrame:
    """
    One-hot encode categorical columns and return numeric (0/1) columns.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not columns:
        return df

    return pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)


def split_features_target(
    df: pd.DataFrame, 
    target_col: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Shuffle and split DataFrame into X_train, X_test, y_train, y_test.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * (1 - test_size))

    train_df = df_shuffled.iloc[:split_idx]
    test_df = df_shuffled.iloc[split_idx:]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test


@timer
def auto_preprocess(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    cap_outliers: bool = True,
    scale_method: Optional[str] = "standard",
    encode_cats: bool = True,
    optimize_memory: bool = True,
) -> pd.DataFrame:
    """
    Master 1-click pipeline:
    1. Cleans column names
    2. Removes duplicates
    3. Imputes missing values
    4. Caps IQR outliers (excluding target_col)
    5. Scales numeric features (excluding target_col)
    6. One-hot encodes categorical features (excluding target_col)
    7. Optimizes memory usage
    """
    df = clean_column_names(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)

    # Clean target_col name the exact same way clean_column_names does
    target_clean = None
    if target_col:
        target_clean = re.sub(r"\s+", "_", target_col.strip().lower())
        target_clean = re.sub(r"[^a-z0-9_]", "", target_clean)

    num_cols = [
        c for c in df.select_dtypes(include=["number"]).columns 
        if c != target_clean
    ]
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns 
        if c != target_clean
    ]

    if cap_outliers and num_cols:
        df = cap_outliers_iqr(df, columns=num_cols)

    if scale_method and num_cols:
        df = scale_numeric_features(df, columns=num_cols, method=scale_method)

    if encode_cats and cat_cols:
        df = encode_categorical(df, columns=cat_cols, drop_first=True)

    if optimize_memory:
        df = reduce_memory_usage(df, verbose=True)

    return df