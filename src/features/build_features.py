import pandas as pd

def map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply binary encoding for 2 values categorical columns
    """
    # gets unique values and remove Nan
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # Yes/No mapping
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
    
    # gender mapping
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")
    
    # generic mapping
    # for any other 2-value categorical cols, use stable aphabetical ordering
    if len(vals) == 2:
        # sort values
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]:1}
        return s.astype(str).map(mapping).astype("Int64")
    
    return s


def build_features(df: pd.DataFrame, target_col: str="Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data
    """
    df = df.copy()

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    sep = "-" * 60
    print(sep)
    print("Feature engineering pipeline")
    print(sep)
    print(f"  input shape       : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  numerical cols    : {len(num_cols)}")
    print(f"  categorical cols  : {len(obj_cols)}  (binary={len(binary_cols)}, multi={len(multi_cols)})")
    if binary_cols:
        print(f"    binary          : {binary_cols}")
    if multi_cols:
        print(f"    multi-category  : {multi_cols}")
    print(sep)

    print("[1/3] binary encoding")
    if binary_cols:
        for c in binary_cols:
            original_dtype = df[c].dtype
            df[c] = map_binary_series(df[c].astype(str))
            print(f"      {c:<18} {str(original_dtype):<8} -> Int64 (0/1)")
    else:
        print("      (no binary columns)")

    print("[2/3] boolean -> int")
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"      converted {len(bool_cols)} cols: {bool_cols}")
    else:
        print("      (no boolean columns)")

    print("[3/3] one-hot encoding (drop_first=True)")
    if multi_cols:
        before_cols = df.shape[1]
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        dummies_added = df.shape[1] - (before_cols - len(multi_cols))
        print(f"      {len(multi_cols)} cols -> {dummies_added} dummy cols")
    else:
        print("      (no multi-category columns)")

    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)

    print(sep)
    print(f"Done: {df.shape[1]} final features ({df.shape[0]} rows)")
    print(sep)
    return df