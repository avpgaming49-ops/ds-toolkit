# General reusable functions

# Memory optimization, execution timer decorator, and reproducibility helpers
import time
import functools
import numpy as np
import pandas as pd
from typing import Callable, Any


def timer(func: Callable) -> Callable:
    """Decorator to measure and print the execution time of any function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            print(f"[TIMER] '{func.__name__}' executed in {elapsed:.4f}s")
            
    return wrapper


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns (int64/float64) to the smallest safe subtype
    to save RAM on large datasets.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    df = df.copy()
    start_mem = df.memory_usage().sum() / (1024 ** 2)

    for col in df.select_dtypes(include=["number"]).columns:
        col_type = df[col].dtype
        
        # Skip datetime or boolean columns if captured by number selector
        if pd.api.types.is_bool_dtype(col_type):
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        # Check if integer
        if pd.api.types.is_integer_dtype(col_type):
            # Skip if column contains NaN (numpy int types cannot hold NaNs)
            if df[col].isna().any():
                continue

            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        # Check if float
        elif pd.api.types.is_float_dtype(col_type):
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / (1024 ** 2)
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0.0
        print(f"Memory reduced from {start_mem:.4f} MB to {end_mem:.4f} MB ({reduction:.1f}% saved)")

    return df