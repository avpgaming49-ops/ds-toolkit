# Missing values, duplicates, column names
import pandas as pd
from typing import Optional, List, Union


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame column names to snake_case."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    df = df.copy()
    df.columns = (
        df.columns.astype(str)  # Fixes crash on integer/non-string columns
        .str.strip()
        # Optional: Convert CamelCase (e.g. "firstName" -> "first_Name") before lowering
        .str.replace(r"([a-z0-9])([A-Z])", r"\1_\2", regex=True)
        .str.lower()
        # Replace spaces, hyphens, slashes, and symbols with underscores
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        # Collapse multiple underscores into one and strip leading/trailing '_'
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[Union[str, List[str]]] = None,
    keep: Union[str, bool] = "first"
) -> pd.DataFrame:
    """Drop duplicate rows and reset the index."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if df.empty:
        return df.copy()

    # 1. Validate 'keep' parameter
    if isinstance(keep, str) and keep.lower() == "false":
        keep = False
    if keep not in ("first", "last", False):
        raise ValueError("Parameter 'keep' must be 'first', 'last', or False.")

    # 2. Validate 'subset' parameter
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        if len(subset) == 0:
             subset = None  # Prevents ValueError on empty list subset=[]
        else:
            missing_cols = [col for col in subset if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    df = df.copy()

    # 3. Drop duplicates (with fallback if cells contain unhashable lists/dicts)
    try:
        df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    except TypeError:
        # Fallback if a column contains lists or dicts inside cells
        temp_cols = subset if subset is not None else df.columns
        mask = df[temp_cols].astype(str).duplicated(keep=keep)
        df = df[~mask].reset_index(drop=True)

    return df


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of columns containing missing values."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if df.empty or len(df) == 0:
        return pd.DataFrame(columns=["missing_count", "missing_pct", "dtype"])

    missing_count = df.isna().sum()
    missing_pct = ((missing_count / len(df)) * 100).round(2)

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "dtype": df.dtypes.astype(str)  # String prevents JSON/UI serialization crashes
    })

    # Filter only columns with missing values and sort by count first for exact tie-breaking
    summary = summary[summary["missing_count"] > 0].sort_values(
        by=["missing_count", "missing_pct"], ascending=False
    )
    summary.index.name = "column_name"
    return summary


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    fill_value: Optional[Union[str, int, float]] = None
) -> pd.DataFrame:
    """
    Fill missing values automatically across numeric and categorical columns.
    numeric_strategy: 'mean', 'median', 'zero', or 'constant'
    categorical_strategy: 'mode' or 'constant'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if df.empty:
        return df.copy()

    numeric_strategy = numeric_strategy.lower().strip()
    categorical_strategy = categorical_strategy.lower().strip()

    if numeric_strategy not in ("median", "mean", "zero", "constant"):
        raise ValueError(
            "numeric_strategy must be 'median', 'mean', 'zero', or 'constant'."
        )
    if categorical_strategy not in ("mode", "constant"):
        raise ValueError(
            "categorical_strategy must be 'mode' or 'constant'."
        )

    df = df.copy()

    # 1. Handle Numeric Columns
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isna().sum() > 0:
            if numeric_strategy == "median":
                val = df[col].median()
            elif numeric_strategy == "mean":
                val = df[col].mean()
            elif numeric_strategy == "zero":
                val = 0
            elif numeric_strategy == "constant":
                val = fill_value if isinstance(fill_value, (int, float)) else 0

            # Fallback to 0 if the entire column was 100% NaN
            if pd.isna(val):
                val = 0

            # Prevent crash on Nullable Integer columns (Int64) when filling with a float
            if pd.api.types.is_integer_dtype(df[col].dtype) and isinstance(val, float) and not val.is_integer():
                df[col] = df[col].astype(float)

            df[col] = df[col].fillna(val)

    # 2. Handle Categorical / Object / String Columns
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    default_cat_fill = str(fill_value) if fill_value is not None else "Unknown"

    for col in cat_cols:
        if df[col].isna().sum() > 0:
            if categorical_strategy == "mode":
                mode_series = df[col].mode(dropna=True)
                val = mode_series.iloc[0] if not mode_series.empty else default_cat_fill
            elif categorical_strategy == "constant":
                val = default_cat_fill

            # Prevent TypeError crash if column is 'category' dtype and val isn't in categories
            if isinstance(df[col].dtype, pd.CategoricalDtype) and val not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([val])

            df[col] = df[col].fillna(val)

    return df