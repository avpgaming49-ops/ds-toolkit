import pandas as pd
import numpy as np
from rage-tools.evaluation import evaluate_regression, evaluate_classification
from rage-tools.utils import timer, reduce_memory_usage

@timer
def run_eval_and_utils_tests():
    # 1. Test Regression Metrics
    y_true_reg = [100, 150, 200, 250, 300]
    y_pred_reg = [110, 145, 195, 260, 290]
    print("--- 1. REGRESSION METRICS ---")
    print(evaluate_regression(y_true_reg, y_pred_reg, n_features=2))

    # 2. Test Classification Metrics
    y_true_clf = [1, 0, 1, 1, 0, 1, 0, 0]
    y_pred_clf = [1, 0, 1, 0, 0, 1, 1, 0]
    print("\n--- 2. CLASSIFICATION METRICS ---")
    print(evaluate_classification(y_true_clf, y_pred_clf))

    # 3. Test Memory Reduction
    print("\n--- 3. MEMORY REDUCTION ---")
    big_df = pd.DataFrame({
        "small_int": np.random.randint(0, 50, size=10000, dtype=np.int64),
        "precise_float": np.random.randn(10000).astype(np.float64)
    })
    optimized_df = reduce_memory_usage(big_df)
    print("Old dtypes:", big_df.dtypes.to_dict())
    print("New dtypes:", optimized_df.dtypes.to_dict())

if __name__ == "__main__":
    run_eval_and_utils_tests()