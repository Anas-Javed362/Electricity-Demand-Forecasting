"""
Prophet trainer with weather regressors, US holiday effects, and MLflow tracking.

Pipeline
--------
1. Build feature set via feature_engineering.py
2. Prepare ds/y DataFrame (+ weather regressors: temp, rhum, wspd)
3. Fit Prophet with daily / weekly / yearly seasonality + US holidays
4. Evaluate on test split → log MAE / RMSE / MAPE to MLflow
5. Generate 72-hour future forecast and persist for dashboard use
6. Save model artifact to models/

Usage
-----
    python src/train_prophet.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import mlflow

# ---------------------------------------------------------------------------
# MLflow - absolute path avoids Windows URL-encoding issues.
# MLFLOW_ALLOW_FILE_STORE is required by MLflow 3.x to use the file store.
# ---------------------------------------------------------------------------
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
_MLRUNS_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "mlruns"))
mlflow.set_tracking_uri(f"file:///{_MLRUNS_PATH}")

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.feature_engineering import build_feature_set, TRAIN_TEST_SPLIT_DATE

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MLFLOW_EXPERIMENT = "pjm_forecasting"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train():
    from prophet import Prophet

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("📊 Building feature set …")
    df = build_feature_set()

    # Prepare Prophet-format DataFrame
    prophet_df = df[["load_mw", "temp", "rhum", "wspd"]].reset_index()
    prophet_df = prophet_df.rename(columns={"Datetime": "ds", "load_mw": "y"})

    train_df = prophet_df[prophet_df["ds"] < TRAIN_TEST_SPLIT_DATE].copy()
    test_df = prophet_df[prophet_df["ds"] >= TRAIN_TEST_SPLIT_DATE].copy()
    print(f"   Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

    params = {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "multiplicative",
        "daily_seasonality": True,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
    }

    with mlflow.start_run(run_name="prophet") as run:
        mlflow.log_params(params)

        print("🔮 Training Prophet model …")
        model = Prophet(
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            seasonality_mode=params["seasonality_mode"],
            daily_seasonality=params["daily_seasonality"],
            weekly_seasonality=params["weekly_seasonality"],
            yearly_seasonality=params["yearly_seasonality"],
        )
        model.add_country_holidays(country_name="US")
        model.add_regressor("temp")
        model.add_regressor("rhum")
        model.add_regressor("wspd")

        # Suppress verbose Stan output
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        model.fit(train_df)

        # ------------------------------------------------------------------
        # Evaluate on test set
        # ------------------------------------------------------------------
        future = test_df[["ds", "temp", "rhum", "wspd"]].copy()
        forecast = model.predict(future)

        y_true = test_df["y"].values
        y_pred = forecast["yhat"].values

        test_mae = float(mean_absolute_error(y_true, y_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        test_mape = mape(y_true, y_pred)

        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mape", test_mape)
        print(f"📈 Test  MAE : {test_mae:.2f} MW")
        print(f"   Test RMSE : {test_rmse:.2f} MW")
        print(f"   Test MAPE : {test_mape:.2f} %")

        # ------------------------------------------------------------------
        # 72-hour future forecast (weather regressors = test-set rolling mean)
        # ------------------------------------------------------------------
        last_date = prophet_df["ds"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1), periods=72, freq="h"
        )
        future_72 = pd.DataFrame({"ds": future_dates})
        future_72["temp"] = test_df["temp"].mean()
        future_72["rhum"] = test_df["rhum"].mean()
        future_72["wspd"] = test_df["wspd"].mean()
        forecast_72 = model.predict(future_72)

        # ------------------------------------------------------------------
        # Persist
        # ------------------------------------------------------------------
        model_path = os.path.join(MODEL_DIR, "prophet_model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        forecast_path = os.path.join(MODEL_DIR, "prophet_forecast.joblib")
        joblib.dump(
            {
                "forecast_test": forecast,
                "test_df": test_df,
                "forecast_72h": forecast_72,
            },
            forecast_path,
        )

        print(f"💾 Model saved → {model_path}")
        print(f"📝 MLflow run ID : {run.info.run_id}")

    return model, test_mae, test_rmse, test_mape


if __name__ == "__main__":
    train()
