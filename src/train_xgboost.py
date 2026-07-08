"""
XGBoost trainer with Optuna hyperparameter optimisation and MLflow experiment tracking.

Pipeline
--------
1. Build feature set via feature_engineering.py
2. Run 50-trial Optuna study (3-fold TimeSeriesSplit) to minimise MAE
3. Retrain on full training set with best hyperparameters
4. Evaluate on hold-out test set → log MAE / RMSE / MAPE to MLflow
5. Compute SHAP summary (500-sample subset) and log as artifact
6. Persist model + SHAP values to models/

Usage
-----
    python src/train_xgboost.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost

# ---------------------------------------------------------------------------
# MLflow – absolute path avoids Windows URL-encoding issues.
# MLFLOW_ALLOW_FILE_STORE is required by MLflow 3.x to use the file store.
# ---------------------------------------------------------------------------
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
_MLRUNS_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns"))
mlflow.set_tracking_uri(f"file:///{_MLRUNS_PATH}")

import optuna
import xgboost as xgb
import shap
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.feature_engineering import (
    build_feature_set,
    get_train_test,
    get_tscv,
    FEATURE_COLS,
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MLFLOW_EXPERIMENT = "pjm_forecasting"
N_OPTUNA_TRIALS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _objective(trial: optuna.Trial, X_train, y_train) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }

    tscv = get_tscv(n_splits=3)
    fold_maes = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        fold_maes.append(mean_absolute_error(y_val, preds))

    return float(np.mean(fold_maes))


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("📊 Building feature set …")
    df = build_feature_set()
    X_train, y_train, X_test, y_test = get_train_test(df)
    print(f"   Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"   Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    with mlflow.start_run(run_name="xgboost_optuna") as run:

        # ------------------------------------------------------------------
        # Optuna search
        # ------------------------------------------------------------------
        print(f"🔍 Running Optuna hyperparameter search ({N_OPTUNA_TRIALS} trials) …")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            study_name="xgb_mae",
        )
        study.optimize(
            lambda trial: _objective(trial, X_train, y_train),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )

        best_params = study.best_params.copy()
        best_params.update({"random_state": 42, "n_jobs": -1, "tree_method": "hist", "verbosity": 0})
        print(f"✅ Best CV MAE : {study.best_value:.2f} MW")
        print(f"   Best params : {best_params}")
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_mae", study.best_value)

        # ------------------------------------------------------------------
        # Train final model on full training set
        # ------------------------------------------------------------------
        print("🚀 Training final XGBoost model on full training set …")
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)

        # ------------------------------------------------------------------
        # Evaluation on test set
        # ------------------------------------------------------------------
        preds = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, preds)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        test_mape = mape(y_test.values, preds)

        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mape", test_mape)
        print(f"📈 Test  MAE : {test_mae:.2f} MW")
        print(f"   Test RMSE : {test_rmse:.2f} MW")
        print(f"   Test MAPE : {test_mape:.2f} %")

        # ------------------------------------------------------------------
        # SHAP values
        # ------------------------------------------------------------------
        print("🔬 Computing SHAP values (500-row sample) …")
        sample = X_test.iloc[:500]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        fig, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, sample, feature_names=FEATURE_COLS, show=False)
        shap_img = os.path.join(MODEL_DIR, "shap_summary.png")
        plt.savefig(shap_img, bbox_inches="tight", dpi=150)
        plt.close()
        mlflow.log_artifact(shap_img)

        # Persist SHAP data for dashboard
        joblib.dump(
            {"shap_values": shap_values, "X_sample": sample},
            os.path.join(MODEL_DIR, "shap_values.joblib"),
        )

        # ------------------------------------------------------------------
        # Save model — canonical filename: xgboost_model.joblib
        # ------------------------------------------------------------------
        model_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
        joblib.dump(model, model_path)
        mlflow.xgboost.log_model(model, "xgboost_model")
        print(f"💾 Model saved → {model_path}")
        print(f"📝 MLflow run ID : {run.info.run_id}")

    return model, test_mae, test_rmse, test_mape


if __name__ == "__main__":
    train()
