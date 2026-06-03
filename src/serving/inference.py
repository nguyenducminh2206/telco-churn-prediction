"""
Load the trained model + feature schema and predict churn for a single customer.

The transformations here must mirror those used at training time
(see src/data/preprocess.py and src/features/build_features.py).
"""

import os
import json
import pandas as pd
import mlflow.sklearn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def _find_latest_model_dir() -> str:
    """Pick the most recently modified MLflow model folder."""
    candidates = list(MLRUNS_DIR.glob("*/*/artifacts/model"))
    if not candidates:
        raise FileNotFoundError(
            f"No model found under {MLRUNS_DIR}. "
            "Run `python scripts/run_pipeline.py --input ...` first."
        )
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


# allow overriding via env var (useful for Docker / production)
MODEL_DIR = os.environ.get("MODEL_DIR") or _find_latest_model_dir()
print(f"Loading model from {MODEL_DIR}")

model = mlflow.sklearn.load_model(MODEL_DIR)

with open(ARTIFACTS_DIR / "feature_columns.json") as f:
    FEATURE_COLS = json.load(f)
print(f"Loaded {len(FEATURE_COLS)} feature columns")


# fixed mappings — must match training. If training changes, update here.
BINARY_MAP = {
    "gender":           {"Female": 0, "Male": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as training, then align columns."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # numeric coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = (
            pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        )

    # binary encoding
    for col, mapping in BINARY_MAP.items():
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().map(mapping).fillna(0).astype(int)
            )

    # one-hot encode anything still categorical
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    # align with training schema: missing dummies -> 0, extra cols dropped
    return df.reindex(columns=FEATURE_COLS, fill_value=0)


def predict_churn(customer: dict, threshold: float = 0.35) -> dict:
    """
    Predict churn for a single customer.

    Args:
        customer: dict of raw feature values (same keys as the raw CSV).
        threshold: probability cutoff for the "churn" label (default 0.35,
                   matching the run_pipeline.py default).

    Returns:
        {"probability": float, "label": "Likely to churn" | "Not likely to churn"}
    """
    df = pd.DataFrame([customer])
    df_enc = _transform(df)

    proba = float(model.predict_proba(df_enc)[0, 1])
    label = "Likely to churn" if proba >= threshold else "Not likely to churn"

    return {"probability": round(proba, 4), "label": label}


if __name__ == "__main__":
    # smoke test
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
    print(predict_churn(sample))
