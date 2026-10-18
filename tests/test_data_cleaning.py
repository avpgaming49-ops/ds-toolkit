import pandas as pd
import numpy as np
from ds_toolkit.data_cleaning import (
    clean_column_names,
    remove_duplicates,
    missing_value_summary,
    handle_missing_values,
)

def run_tests():
    # Create a messy dummy dataset
    raw_data = {
        "Customer Name ": ["Aman", "Aman", "Priya", None],
        "Age (Years)": [21, 21, np.nan, 25],
        "Annual Income$": [50000, 50000, 60000, 55000]
    }
    df = pd.DataFrame(raw_data)

    # 1. Test clean_column_names
    df_clean = clean_column_names(df)
    print("1. Cleaned Columns:", df_clean.columns.tolist())

    # 2. Test remove_duplicates
    df_dedup = remove_duplicates(df_clean)
    print(f"2. Rows after deduplication: {len(df_dedup)} (Expected: 3)")

    # 3. Test missing_value_summary
    print("\n3. Missing Value Summary:")
    print(missing_value_summary(df_dedup))

    # 4. Test handle_missing_values
    df_filled = handle_missing_values(df_dedup, numeric_strategy="median", categorical_strategy="mode")
    print("\n4. Final Cleaned DataFrame:")
    print(df_filled)

if __name__ == "__main__":
    run_tests()