import pandas as pd

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID":      ["1", "2", "3", "4"],
        "gender":          ["Female", "Male", "Female", "Male"],
        "SeniorCitizen":   [0, 1, 0, 0],
        "Partner":         ["Yes", "No", "Yes", "No"],
        "Dependents":      ["No", "No", "Yes", "Yes"],
        "tenure":          [1, 24, 60, 12],
        "PhoneService":    ["Yes", "Yes", "No", "Yes"],
        "Contract":        ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "PaperlessBilling":["Yes", "No", "No", "Yes"],
        "MonthlyCharges":  [29.85, 56.95, 53.85, 70.0],
        "TotalCharges":    [29.85, 1366.8, 3230.1, 840.0],
        "Churn":           ["No", "Yes", "No", "Yes"],
    })


def test_no_object_dtype_columns_remain():
    """The model can't consume strings, so feature engineering must remove them."""
    df = preprocess_data(_sample_df())
    out = build_features(df)
    assert out.select_dtypes(include=["object"]).empty


def test_target_column_is_preserved():
    df = preprocess_data(_sample_df())
    out = build_features(df)
    assert "Churn" in out.columns


def test_row_count_is_unchanged():
    df = preprocess_data(_sample_df())
    out = build_features(df)
    assert len(out) == len(df)
