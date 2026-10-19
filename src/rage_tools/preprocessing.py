# Outlier capping, feature scaling, categorical encoding, train-test split, and auto-pipeline
import re
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union

from rage_tools.data_cleaning import (
    clean_column_names,
    remove_duplicates,
    handle_missing_values,
)
from rage_tools.utils import timer, reduce_memory_usage


def cap_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Cap (clip) extreme outliers in numeric columns to the lower and upper IQR bounds
    instead of dropping rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    df = df.copy()

    # Handle if user passes a single string column name instead of a list
    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        # Validate that requested columns exist and are numeric
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue  # Skip non-numeric columns gracefully

        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        # If IQR is 0 (all values identical), skip clipping for this column
        if iqr == 0:
            continue

        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def scale_numeric_features(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    method: str = "standard"
) -> pd.DataFrame:
    """
    Scale numeric features using 'standard' (Z-score) or 'minmax' (0 to 1) scaling.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    df = df.copy()

    # Handle if user passes a single string column name instead of a list
    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    method = method.lower().strip()
    if method not in ("standard", "minmax"):
        raise ValueError("method must be 'standard' or 'minmax'.")

    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue  # Skip non-numeric columns gracefully

        series = df[col].dropna()
        if series.empty:
            continue

        if method == "standard":
            std = df[col].std()
            mean = df[col].mean()
            if std != 0 and not pd.isna(std):
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0.0  # Fallback if standard deviation is zero

        elif method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val and not pd.isna(max_val):
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0  # Fallback if all values are identical

    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    drop_first: bool = True
) -> pd.DataFrame:
    """
    One-hot encode categorical columns and return numeric (0/1) columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    df = df.copy()

    # Handle if user passes a single string column name instead of a list
    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        columns = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    else:
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

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
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    if not (0.0 < test_size < 1.0):
        raise ValueError("Parameter 'test_size' must be strictly between 0.0 and 1.0.")

    if len(df) < 2:
        raise ValueError("DataFrame must contain at least 2 rows to perform a train-test split.")

    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * (1 - test_size))

    # Ensure train and test sets both have at least 1 row
    split_idx = max(1, min(split_idx, len(df_shuffled) - 1))

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
    optimize_memory: bool = True
) -> pd.DataFrame:
    """
    Master 1-click pipeline:
    1. Cleans column names
    2. Removes duplicates
    3. Imutes missing values
    4. Caps IQR outliers (excluding target_col)
    5. Scales numeric features (excluding target_col)
    6. One-hot encodes categorical features (excluding target_col)
    7. Optimizes memory usage
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    df = df.copy()

    if target_col is not None and target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # 1. Standardize column names first
    df = clean_column_names(df)

    # Resolve target_col name after column cleaning
    target_clean = None
    if target_col is not None:
        # Pass target_col through a temporary Series/Index or re-run the same cleaning logic
        temp_series = pd.Index([target_col])
        temp_cleaned = (
            temp_series.astype(str)
            .str.strip()
            .str.replace(r"([a-z0-9])([A-Z])", r"\1_\2", regex=True)
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )
        target_clean = temp_cleaned[0]
        if target_clean not in df.columns:
            raise ValueError(f"Cleaned target column '{target_clean}' not found in DataFrame columns.")

    # 2. Clean data steps
    df = remove_duplicates(df)
    df = handle_missing_values(df)

    # 3. Identify feature columns (excluding target)
    num_cols = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c != target_clean
    ]
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category", "string"]).columns
        if c != target_clean
    ]

    # 4. Outlier Capping
    if cap_outliers and num_cols:
        df = cap_outliers_iqr(df, columns=num_cols)

    # 5. Feature Scaling
    if scale_method and num_cols:
        df = scale_numeric_features(df, columns=num_cols, method=scale_method)

    # 6. Categorical Encoding
    if encode_cats and cat_cols:
        df = encode_categorical(df, columns=cat_cols, drop_first=True)

    # 7. Memory Optimization
    if optimize_memory:
        df = reduce_memory_usage(df, verbose=False)  # Set verbose=False to keep logs clean

    return df