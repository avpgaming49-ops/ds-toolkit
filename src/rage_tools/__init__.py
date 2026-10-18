from .data_cleaning import (
    clean_column_names,
    remove_duplicates,
    missing_value_summary,
    handle_missing_values,
)
from .eda import (
    dataset_overview,
    numeric_summary,
    categorical_summary,
    detect_outliers_iqr,
)
from .preprocessing import (
    cap_outliers_iqr,
    scale_numeric_features,
    encode_categorical,
    split_features_target,
    auto_preprocess,  # <-- Added here!
)
from .evaluation import (
    evaluate_regression,
    evaluate_classification,
)
from .utils import (
    timer,
    reduce_memory_usage,
)

__all__ = [
    "clean_column_names",
    "remove_duplicates",
    "missing_value_summary",
    "handle_missing_values",
    "dataset_overview",
    "numeric_summary",
    "categorical_summary",
    "detect_outliers_iqr",
    "cap_outliers_iqr",
    "scale_numeric_features",
    "encode_categorical",
    "split_features_target",
    "auto_preprocess",  # <-- And added here!
    "evaluate_regression",
    "evaluate_classification",
    "timer",
    "reduce_memory_usage",
]

from .data_cleaning import (
    clean_column_names,
    remove_duplicates,
    missing_value_summary,
    handle_missing_values,
)
from .eda import (
    dataset_overview,
    numeric_summary,
    categorical_summary,
    detect_outliers_iqr,
)
from .preprocessing import (
    cap_outliers_iqr,
    scale_numeric_features,
    encode_categorical,
    split_features_target,
    auto_preprocess,
)
from .evaluation import (
    evaluate_regression,
    evaluate_classification,
)
from .utils import (
    timer,
    reduce_memory_usage,
)