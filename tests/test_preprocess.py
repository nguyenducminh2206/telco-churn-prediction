import pandas as pd

from src.data.preprocess import preprocess_data


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID":     ["1", "2", "3"],
        "gender":         ["Female", "Male", "Female"],
        "SeniorCitizen":  [0, 1, 0],
        "tenure":         [1, 24, 60],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        # one bad row in TotalCharges to test coercion
        "TotalCharges":   ["29.85", " ", "108.15"],
        "Churn":          ["No", "Yes", "No"],
    })


def test_drops_customer_id_column():
    out = preprocess_data(_sample_df())
    assert "customerID" not in out.columns


def test_churn_is_encoded_as_binary_integers():
    out = preprocess_data(_sample_df())
    assert set(out["Churn"].unique()) == {0, 1}


def test_totalcharges_is_numeric_after_coercion():
    out = preprocess_data(_sample_df())
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    # the " " value should have been coerced and then filled with 0
    assert out["TotalCharges"].isna().sum() == 0
