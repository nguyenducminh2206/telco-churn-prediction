import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str="Churn") -> pd.DataFrame:
    """
    Cleaning telco churn data
    - trim column names
    - drop ID col
    - coerce TotalCharge
    - map target churn to binary
    - Nan handling
    """
    # clean header
    df.columns = df.columns.str.strip()

    # drop ids
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # binary encode the target if it's Yes/No
    if target_col in df.columns and df[target_col].dtype == 'object':
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # coerce the TotalCharge to float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # handling SeniorCitizen
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)
    
    # Nan handling
    # - numeric: fill with 0
    # - others: leave for encoders to handle
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df