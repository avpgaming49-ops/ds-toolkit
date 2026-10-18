import pandas as pd
import numpy as np
from rage-tools import auto_preprocess
from rage-tools.preprocessing import (
    cap_outliers_iqr,
    scale_numeric_features,
    encode_categorical,
    split_features_target,
)

def run_preprocessing_tests():
    df = pd.DataFrame({
        "city": ["Lucknow", "Delhi", "Lucknow", "Mumbai", "Delhi", "Mumbai"],
        "age": [20, 22, 21, 23, 19, 120],  # 120 is an extreme outlier
        "salary": [30000, 40000, 35000, 50000, 25000, 60000],
        "purchased": [0, 1, 0, 1, 0, 1]
    })

    # 1. Cap outliers in age
    df_capped = cap_outliers_iqr(df, columns=["age"])
    print("1. Age after capping outlier (120 -> capped):", df_capped["age"].tolist())

    # 2. Scale numeric columns (excluding target 'purchased')
    df_scaled = scale_numeric_features(df_capped, columns=["age", "salary"], method="minmax")
    print("\n2. MinMax Scaled Age & Salary:")
    print(df_scaled[["age", "salary"]].round(2))

    # 3. One-Hot Encode 'city'
    df_encoded = encode_categorical(df_scaled, columns=["city"], drop_first=True)
    print("\n3. Encoded DataFrame Columns:", df_encoded.columns.tolist())

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = split_features_target(df_encoded, target_col="purchased", test_size=0.33)
    print(f"\n4. Split Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 5. Master 1-Click Auto Preprocess
    print("\n--- 5. MASTER 1-CLICK AUTO PREPROCESS ---")
    raw_messy_df = pd.DataFrame({
        "User City ": ["Lucknow", "Lucknow", "Delhi", "Mumbai", None, "Delhi"],
        "Age (Years)": [20, 20, 22, 150, np.nan, 25],
        "Monthly Salary$": [30000, 30000, 45000, 50000, 35000, 60000],
        "Target": [0, 0, 1, 1, 0, 1]
    })
    clean_ml_df = auto_preprocess(raw_messy_df, target_col="Target")
    print(clean_ml_df)

if __name__ == "__main__":
    run_preprocessing_tests()