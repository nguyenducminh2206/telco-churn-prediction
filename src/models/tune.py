import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def tune_model(X, y):
    """
    Tunes an XGBoost model using Optuna.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("-" * 40)
    print("Hyperparameter tuning complete")
    print(f"  trials       : {len(study.trials)}")
    print(f"  best recall  : {study.best_value:.4f}")
    print("  best params  :")
    for name, val in study.best_params.items():
        if isinstance(val, float):
            print(f"    {name:<18} {val:.4f}")
        else:
            print(f"    {name:<18} {val}")
    print("-" * 40)
    return study.best_params