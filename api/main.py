"""
FastAPI REST backend for PJM energy consumption forecasting.

Endpoints
---------
  GET  /health           Liveness probe
  GET  /docs             Swagger UI (auto-generated)
  POST /predict          Single-point energy forecast
  GET  /metrics          Latest model evaluation scores
  GET  /forecast/72h     72-hour Prophet forward forecast

Run
---
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
import os
import sys
from datetime import datetime, timezone

# Make project root importable when running from the forecasting/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = "models"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PJM Energy Forecasting API",
    description=(
        "Predict hourly electricity consumption for the PJM Eastern Interconnection "
        "using XGBoost (Optuna-tuned), LSTM, and Prophet models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lazy-loaded model cache
# ---------------------------------------------------------------------------

_xgb_model = None


def _get_xgb_model():
    global _xgb_model
    if _xgb_model is None:
        path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
        if not os.path.exists(path):
            raise RuntimeError(
                "XGBoost model not found. Run: python src/train_xgboost.py"
            )
        _xgb_model = joblib.load(path)
    return _xgb_model


def _build_feature_row(dt_obj: pd.Timestamp, temp: float, rhum: float, wspd: float) -> pd.DataFrame:
    import holidays as hd

    us_hols = hd.US()
    _LOAD_MEAN = 30_000.0
    row = {
        "hour": dt_obj.hour,
        "day_of_week": dt_obj.dayofweek,
        "month": dt_obj.month,
        "quarter": dt_obj.quarter,
        "day_of_year": dt_obj.dayofyear,
        "week_of_year": dt_obj.isocalendar()[1],
        "is_weekend": int(dt_obj.dayofweek >= 5),
        "is_holiday": int(dt_obj.date() in us_hols),
        "hour_sin": np.sin(2 * np.pi * dt_obj.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt_obj.hour / 24),
        "month_sin": np.sin(2 * np.pi * dt_obj.month / 12),
        "month_cos": np.cos(2 * np.pi * dt_obj.month / 12),
        "lag_1h": _LOAD_MEAN,
        "lag_24h": _LOAD_MEAN,
        "lag_168h": _LOAD_MEAN,
        "rolling_mean_24h": _LOAD_MEAN,
        "rolling_mean_168h": _LOAD_MEAN,
        "rolling_std_24h": 2_000.0,
        "temp": temp,
        "rhum": rhum,
        "wspd": wspd,
        "temp_sq": temp ** 2,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    datetime: str = Field(
        ...,
        example="2025-06-01 14:00",
        description="ISO datetime string (YYYY-MM-DD HH:MM)",
    )
    temp: float = Field(..., example=22.5, description="Temperature in °C")
    rhum: float = Field(..., example=65.0, description="Relative humidity in %")
    wspd: float = Field(..., example=12.0, description="Wind speed in km/h")


class PredictResponse(BaseModel):
    forecast_mw: float
    model: str
    input_datetime: str
    server_timestamp: str


class MetricsResponse(BaseModel):
    metrics: dict
    source: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Liveness probe."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Forecast"])
def predict(req: PredictRequest):
    """Return a single-point energy consumption forecast (MW)."""
    try:
        dt_obj = pd.Timestamp(req.datetime)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Cannot parse datetime: '{req.datetime}'")

    try:
        model = _get_xgb_model()
        X = _build_feature_row(dt_obj, req.temp, req.rhum, req.wspd)
        pred = float(model.predict(X)[0])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictResponse(
        forecast_mw=round(pred, 2),
        model="xgboost",
        input_datetime=req.datetime,
        server_timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics", tags=["Evaluation"])
def metrics():
    """Return the latest model evaluation scores from models/model_metrics.csv."""
    path = os.path.join(MODEL_DIR, "model_metrics.csv")
    if not os.path.exists(path):
        return {
            "metrics": {},
            "source": "not_found",
            "hint": "Run: python src/evaluate.py",
        }
    df = pd.read_csv(path, index_col=0)
    return {
        "metrics": df.round(2).to_dict(orient="index"),
        "source": path,
    }


@app.get("/forecast/72h", tags=["Forecast"])
def forecast_72h():
    """Return the pre-computed 72-hour Prophet forward forecast."""
    path = os.path.join(MODEL_DIR, "prophet_forecast.joblib")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Prophet forecast not found. Run: python src/train_prophet.py",
        )
    data = joblib.load(path)
    fc = data["forecast_72h"][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    fc["ds"] = fc["ds"].astype(str)
    return {"forecast": fc.to_dict(orient="records")}
