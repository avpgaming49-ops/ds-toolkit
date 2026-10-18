import pandas as pd
import numpy as np
from ds_toolkit.eda import (
    dataset_overview,
    numeric_summary,
    categorical_summary,
    detect_outliers_iqr,
)

def run_eda_tests():
    df = pd.DataFrame({
        "city": ["Lucknow", "Delhi", "Lucknow", "Mumbai", "Lucknow", None],
        "age": [20, 22, 21, 23, 19, 95],  # 95 is an outlier!
        "salary": [30000, 35000, 32000, 40000, 28000, 31000]
    })

    print("--- 1. DATASET OVERVIEW ---")
    print(dataset_overview(df))

    print("\n--- 2. NUMERIC SUMMARY (with Skew & IQR) ---")
    print(numeric_summary(df))

    print("\n--- 3. CATEGORICAL SUMMARY ---")
    print(categorical_summary(df))

    print("\n--- 4. OUTLIER DETECTION (IQR) ---")
    print(detect_outliers_iqr(df))

if __name__ == "__main__":
    run_eda_tests()