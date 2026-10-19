# 🛠️ Rage Tools

A fast, modular Python library for automating daily Data Science & Machine Learning workflows (Data Cleaning, EDA, Preprocessing, Evaluation, and Utilities).

## 🚀 Installation

Install the package directly from PyPI:

```bash
pip install rage-tools


💡 Quick Start

1. 1-Click Auto Preprocessing Pipeline
Clean your dataset, handle missing values, cap outliers, scale numeric features, one-hot encode categorical features, and reduce memory usage with a single command:

'''python
import pandas as pd
from rage_tools import auto_preprocess

df = pd.read_csv("your_dataset.csv")'''

# Clean and preprocess everything while excluding your target column
processed_df = auto_preprocess(df, target_col="target", scale_method="standard")

2. Regression Evaluation
Evaluate your machine learning model instantly with metrics and a clean DataFrame summary:

'''python
from rage_tools import evaluate_regression

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

metrics_df = evaluate_regression(y_true, y_pred, n_features=2)
print(metrics_df)'''

📦 Core Modules
data_cleaning: Standardize column names, remove duplicates, and handle missing values.

eda: Generate comprehensive dataset overviews, numeric summaries, categorical stats, and IQR outlier detection.

preprocessing: Cap outliers, scale features, encode categoricals, and split features/targets.

evaluation: Compute regression and classification metrics cleanly.

utils: High-precision execution timers and RAM-saving downcasting tools.

📄 License
Distributed under the MIT License. See LICENSE for more information.