import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# fix path for local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules - Core pipeline components
from src.data.load_data import load_data                    
from src.data.preprocess import preprocess_data       
from src.features.build_features import build_features    
from src.utils.validate_data import validate_telco_data   


def main(args):
    # configure MLflow to use local file-based tracking (not a tracking server)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"  # Local file-based tracking
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    # start MLflow run
    with mlflow.start_run():
        # log hyperparams and configurations
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        bar = "=" * 60

        # === STAGE 1: DATA LOADING AND VALIDATION ===
        print()
        print(bar)
        print("STAGE 1/7: DATA LOADING AND VALIDATION")
        print(bar)
        df = load_data(args.input)
        print(f"  loaded {df.shape[0]} rows x {df.shape[1]} cols from {args.input}")

        # validates data
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectation.json")
            raise ValueError(f"Data quality check failed. Issue: {failed}")
        else:
            print("  validation passed (logged to MLflow)")

        # === STAGE 2: DATA PREPROCESSING ===
        print()
        print(bar)
        print("STAGE 2/7: DATA PREPROCESSING")
        print(bar)
        df = preprocess_data(df)

        processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"  preprocessed -> {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"  saved        -> {processed_path}")

        # === STAGE 3: FEATURE ENGINEERING ===
        print()
        print(bar)
        print("STAGE 3/7: FEATURE ENGINEERING")
        print(bar)
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        ## apply transformation 
        df_enc = build_features(df, target_col=target)  # Binary encoding + one-hot encoding
        
        # convert boolean columns to integers 
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"  feature engineering completed: {df_enc.shape[1]} features")

        # save feature metadata for serving consistancy
        # ensure serving pipeline uses exact same features in exact same order
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # extract feature cols
        feature_cols = list(df_enc.drop(columns=[target]).columns)

        # save locally
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        # log to MLflow for production serving
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        # save preprocessing artifacts for serving pipeline
        # these artifacts ensure training and serving use identical transformation
        preprocessing_artifact = {
            "feature_columns": feature_cols,  # Exact feature order
            "target": target                  # Target column name
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"  saved {len(feature_cols)} feature columns for serving consistency")

        # === STAGE 4: TRAIN/TEST SPLIT ===
        print()
        print(bar)
        print("STAGE 4/7: TRAIN/TEST SPLIT")
        print(bar)
        X = df_enc.drop(columns=[target])  # Feature matrix
        y = df_enc[target]                 # Target vector
        
        # stratified split to maintain class distribution in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,    
            stratify=y,                  # maintain class balance
            random_state=42              
        )
        print(f"  train rows         : {X_train.shape[0]}")
        print(f"  test rows          : {X_test.shape[0]}")

        # handle class imbalance
        # calculate scale_pos_weight to handle imbalanced dataset
        # this tells XGBoost to give more weight to the minority class (churners)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"  scale_pos_weight   : {scale_pos_weight:.2f}  (neg/pos ratio)")

        # === STAGE 5: MODEL TRAINING WITH OPTIMIZED HYPERPARAMS ===
        print()
        print(bar)
        print("STAGE 5/7: MODEL TRAINING")
        print(bar)
        
        model = XGBClassifier(
            # tree structure parameters
            n_estimators=301,        
            learning_rate=0.034,     
            max_depth=7,            
            
            # regularization parameters
            subsample=0.95,         # sample ratio of training instances
            colsample_bytree=0.98,  # sample ratio of features for each tree
            
            # performance parameters
            n_jobs=-1,              # use all CPU cores
            random_state=42,        # reproducible results
            eval_metric="logloss",  # evaluation metric
            
            # handle class imbalance
            scale_pos_weight=scale_pos_weight  
        )

        # train model and track with time
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)  # track training performance
        print(f"  model      : XGBClassifier")
        print(f"  train time : {train_time:.2f}s")

        # === STAGE 6: MODEL EVALUATION ===
        print()
        print(bar)
        print("STAGE 6/7: MODEL EVALUATION")
        print(bar)
        
        # generate predictions and track inference time
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]  # Get probability of churn (class 1)
        
        # apply classification threshold (default: 0.35, optimized for churn detection)
        # Lower threshold = more sensitive to churn (higher recall, lower precision)
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)  # track inference performance

        # log evaluation metrics to MLflow
        # these metrics are essential for model comparison and monitoring
        precision = precision_score(y_test, y_pred)    # predicted churners, how many actually churned?
        recall = recall_score(y_test, y_pred)          # actual churners, how many did we catch?
        f1 = f1_score(y_test, y_pred)                  # harmonic mean of precision and recall
        roc_auc = roc_auc_score(y_test, proba)         # area under ROC curve (threshold-independent)
        
        # log all metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"  precision  : {precision:.4f}")
        print(f"  recall     : {recall:.4f}")
        print(f"  f1         : {f1:.4f}")
        print(f"  roc_auc    : {roc_auc:.4f}")
        print(f"  threshold  : {args.threshold}")

        # === STAGE 7: MODEL SERIALIZATION AND LOGGING ===
        print()
        print(bar)
        print("STAGE 7/7: MODEL SERIALIZATION AND LOGGING")
        print(bar)
        # ESSENTIAL: Log model in MLflow's standard format for serving
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"  # This creates a 'model/' folder in MLflow run artifacts
        )
        run_id = mlflow.active_run().info.run_id
        print(f"  model logged to MLflow run {run_id}")

        # === FINAL PERFORMANCE SUMMARY ===
        print()
        print(bar)
        print("PERFORMANCE SUMMARY")
        print(bar)
        print(f"  training time   : {train_time:.2f}s")
        print(f"  inference time  : {pred_time:.4f}s")
        print(f"  throughput      : {len(X_test)/pred_time:.0f} samples/sec")
        print(bar)
        print("Detailed classification report")
        print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)

r"""
# Use this below to run the pipeline:

python scripts/run_pipeline.py \
    --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --target Churn
"""