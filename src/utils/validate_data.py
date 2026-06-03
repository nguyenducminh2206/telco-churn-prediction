import great_expectations as ge
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    sep = "-" * 60
    print(sep)
    print("Data validation (Great Expectations)")
    print(sep)
    print(f"  input shape       : {df.shape[0]} rows x {df.shape[1]} cols")
    print(sep)

    ge_df = ge.dataset.PandasDataset(df)

    print("[1/5] schema and required columns")
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")
    ge_df.expect_column_to_exist("gender")
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")

    print("[2/5] business logic (allowed category values)")
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )
    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"],
    )

    print("[3/5] numeric ranges (lower bounds)")
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)

    print("[4/5] statistical / plausibility bounds")
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

    print("[5/5] cross-column consistency")
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95,
    )

    results = ge_df.validate()

    failed_expectations = [
        r["expectation_config"]["expectation_type"]
        for r in results["results"]
        if not r["success"]
    ]
    total_checks = len(results["results"])
    failed_checks = len(failed_expectations)
    passed_checks = total_checks - failed_checks

    print(sep)
    status = "PASSED" if results["success"] else "FAILED"
    print(f"Result: {status}  ({passed_checks}/{total_checks} checks passed, {failed_checks} failed)")
    if failed_expectations:
        print("Failed expectations:")
        for name in failed_expectations:
            print(f"  - {name}")
    print(sep)

    return results["success"], failed_expectations