# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python project with a checked-in virtualenv at `.venv/`. Activate with `source .venv/bin/activate` before running anything.
- Dependencies are pinned in `requirements.txt` (install with `pip install -r requirements.txt`). Notable pins: `scikit-learn==1.5.2`, `xgboost==3.0.3`, `lightgbm==4.6.0`, `mlflow==2.14.1`, `optuna==4.4.0`, `great_expectations==1.5.8`, `pandas==2.1.4`, `numpy==1.26.4`.
- No `pyproject.toml`, `setup.py`, lint config, or CI workflows yet — `.github/workflows/`, `tests/`, `docker/`, `great_expectation/`, `mlruns/`, and `artifacts/` exist but are empty placeholders.

## Common commands

- Launch notebooks: `jupyter lab` (or `jupyter notebook`) from repo root. Notebooks live in `notebooks/` and reference data with relative paths like `../data/processed/telco_features.csv`, so run Jupyter from repo root for those paths to resolve.
- Run a single test (once tests exist): `pytest tests/path/to/test_file.py::test_name`. `pytest` is in `requirements.txt` but no tests are written yet.
- Start MLflow UI against the local store: `mlflow ui --backend-store-uri ./mlruns`.

## Architecture

This is an ML pipeline for the IBM Telco Customer Churn dataset (binary `Churn` target). The codebase is **notebook-driven** — `src/` contains a small library of pure functions that the notebooks import; orchestration, training, and tracking all happen inside the notebooks.

### Data flow

1. `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` — original Kaggle/IBM dataset.
2. `notebooks/eda.ipynb` → `notebooks/preprocessing.ipynb` → writes `data/processed/telco_features.csv`. Notebooks pass dataframes between each other via IPython `%store` (e.g. `%store -r df`), so they must be run in order within the same kernel session/profile.
3. `notebooks/modeling.ipynb` reads the processed CSV, splits train/test (stratified, `random_state=42`, `test_size=0.2`), and compares RandomForest / LightGBM / XGBoost.
4. `notebooks/tuning_and_tracking.ipynb` runs Optuna on XGBoost and (per the filename) wires results into MLflow.

### `src/` modules used by the notebooks

- `src/data/load_data.py` — `load_data(path)` thin CSV loader.
- `src/data/preprocess.py` — `preprocess_data(df, target_col="Churn")`: strips column whitespace, drops `customerID` variants, coerces `TotalCharges` to numeric, maps `Churn` Yes/No → 1/0, fills numeric NaNs with 0 (categorical NaNs left for downstream encoders).
- `src/features/build_features.py` — `build_features(df, target_col="Churn")` + `map_binary_series(s)`. Splits object columns by cardinality: 2-value columns get binary-encoded (`Yes`/`No` → 1/0, `Male`/`Female` → 1/0, otherwise alphabetical), >2-value columns are intended for one-hot encoding. **Note**: the binary encoder returns `Int64` (nullable); the function as currently written ends mid-logic after handling boolean columns and does not yet apply one-hot to `multi_cols` or return — treat it as in-progress when editing.
- `src/models/` and `src/utils/` — directories exist but are empty.

### Conventions worth preserving

- Target column name is `Churn` throughout; positive class = 1 (churned). Models use `class_weight='balanced'` (sklearn/LightGBM) or `scale_pos_weight = neg/pos` (XGBoost) for the ~3:1 class imbalance.
- Decision threshold is tuned manually; notebooks use `THRESHOLD = 0.3` (not 0.5) when converting probabilities to labels — primary metric of interest is recall on the churn class.
- `random_state=42` everywhere for reproducibility.
