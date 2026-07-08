"""
api/main.py
===========
FastAPI REST backend for PJM Energy Demand Forecasting.

Feature Engineering
-------------------
All feature construction uses src/feature_engineering.py FEATURE_COLS (23 features)
as the single source of truth, exactly matching what the models were trained on.

Endpoints
---------
  GET  /health           Liveness probe
  GET  /docs             Swagger UI (auto-generated)
  POST /predict          Single-point energy forecast (XGBoost / RF / Linear)
  GET  /metrics          Latest model evaluation scores
  GET  /forecast/72h     72-hour recursive XGBoost forward forecast

Run
---
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Single source of truth for feature names — MUST match training exactly
# ---------------------------------------------------------------------------
from src.feature_engineering import FEATURE_COLS

MODEL_DIR = "models"

# Typical historical load / weather statistics used as defaults for
# lag/rolling features when no live context window is available.
_LOAD_MEAN    = 32_000.0
_LOAD_STD     =  6_500.0
_TEMP_DEFAULT =     10.0   # °C  (annual avg for PJM region)
_RHUM_DEFAULT =     60.0   # %
_WSPD_DEFAULT =      8.0   # km/h

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PJM Energy Forecasting API",
    description=(
        "Predict hourly electricity consumption for the PJM Eastern Interconnection "
        "using Random Forest, XGBoost, and LSTM models trained on 145k real hours "
        "(2002-2018). Feature engineering is synchronised with src/feature_engineering.py."
    ),
    version="2.1.0",
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

_models: dict = {}


def _get_model(name: str = "xgboost"):
    """Load and cache a model by name. Raises RuntimeError if file missing."""
    if name not in _models:
        # Canonical filenames — .joblib preferred; .pkl for RF/Linear (sklearn)
        file_map = {
            "xgboost": ["xgboost_model.joblib", "xgboost_model.pkl"],
            "rf":      ["rf_model.pkl"],
            "linear":  ["linear_model.pkl"],
        }
        if name not in file_map:
            raise RuntimeError(f"Unknown model: '{name}'. Choose from: {list(file_map)}")

        loaded = None
        for fname in file_map[name]:
            path = os.path.join(MODEL_DIR, fname)
            if os.path.exists(path):
                loaded = joblib.load(path)
                break

        if loaded is None:
            tried = ", ".join(os.path.join(MODEL_DIR, f) for f in file_map[name])
            raise RuntimeError(
                f"Model file not found. Tried: {tried}. "
                "Run: python src/train.py  or  python src/train_xgboost.py"
            )
        _models[name] = loaded
    return _models[name]


def _build_feature_row(dt: pd.Timestamp) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame with the EXACT column names and order
    defined in src/feature_engineering.py FEATURE_COLS (23 features).

    Calendar + cyclical features are derived from the timestamp.
    Lag and rolling features are approximated with the historical population mean
    since we have no live context window at single-point inference time.

    Use /forecast/72h for multi-step recursive inference with real lag propagation.
    """
    import holidays as hd
    us_hols = hd.US()

    hour = dt.hour
    month = dt.month

    row = {
        # ── Calendar ──────────────────────────────────────────────────
        "hour":         hour,
        "day_of_week":  dt.dayofweek,
        "month":        month,
        "quarter":      (month - 1) // 3 + 1,
        "day_of_year":  dt.dayofyear,
        "week_of_year": int(dt.isocalendar()[1]),
        "is_weekend":   int(dt.dayofweek >= 5),
        "is_holiday":   int(dt.date() in us_hols),

        # ── Cyclical encodings ─────────────────────────────────────────
        "hour_sin":  np.sin(2 * np.pi * hour / 24),
        "hour_cos":  np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),

        # ── Lag features (approximated with historical mean) ───────────
        "lag_1h":   _LOAD_MEAN,
        "lag_24h":  _LOAD_MEAN,
        "lag_168h": _LOAD_MEAN,

        # ── Rolling features (approximated) ───────────────────────────
        "rolling_mean_24h":  _LOAD_MEAN,
        "rolling_mean_168h": _LOAD_MEAN,
        "rolling_std_24h":   _LOAD_STD,

        # ── Weather features (approximated with annual averages) ───────
        "temp":    _TEMP_DEFAULT,
        "rhum":    _RHUM_DEFAULT,
        "wspd":    _WSPD_DEFAULT,
        "temp_sq": _TEMP_DEFAULT ** 2,
    }

    # Guarantee column order matches training — this is the safety net
    return pd.DataFrame([row])[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    datetime: str = Field(
        ...,
        example="2025-06-01 14:00",
        description="ISO datetime string (YYYY-MM-DD HH:MM)",
    )
    model: str = Field(
        default="xgboost",
        example="xgboost",
        description="Model to use: 'xgboost', 'rf', or 'linear'",
    )


class PredictResponse(BaseModel):
    forecast_mw: float
    model: str
    input_datetime: str
    server_timestamp: str
    feature_count: int
    note: str


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
        "version": "2.1.0",
        "feature_count": len(FEATURE_COLS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Forecast"])
def predict(req: PredictRequest):
    """
    Return a single-point energy consumption forecast (MW).

    Uses src/feature_engineering.py FEATURE_COLS (23 features) to build the
    feature row — exactly matching what the models were trained on.

    Lag/rolling features are approximated with the historical population mean
    (~32,000 MW). For accurate recursive multi-step forecasting with real
    lag propagation use GET /forecast/72h.
    """
    try:
        dt_obj = pd.Timestamp(req.datetime)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Cannot parse datetime: '{req.datetime}'")

    try:
        model = _get_model(req.model)
        X = _build_feature_row(dt_obj)
        pred = float(model.predict(X)[0])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictResponse(
        forecast_mw=round(pred, 2),
        model=req.model,
        input_datetime=req.datetime,
        server_timestamp=datetime.now(timezone.utc).isoformat(),
        feature_count=len(FEATURE_COLS),
        note=(
            "Lag/rolling features approximated with historical mean (~32,000 MW). "
            "Weather features approximated with regional annual averages. "
            "Use /forecast/72h for recursive forecast with real lag propagation."
        ),
    )


@app.get("/metrics", tags=["Evaluation"])
def metrics():
    """Return the latest model evaluation scores from models/metrics.json."""
    json_path = os.path.join(MODEL_DIR, "metrics.json")
    csv_path  = os.path.join(MODEL_DIR, "model_metrics.csv")

    if os.path.exists(json_path):
        import json
        with open(json_path) as f:
            data = json.load(f)
        return {"metrics": data, "source": json_path}

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        return {"metrics": df.round(2).to_dict(orient="index"), "source": csv_path}

    return {
        "metrics": {},
        "source": "not_found",
        "hint": "Run: python src/evaluate.py",
    }


@app.get("/forecast/72h", tags=["Forecast"])
def forecast_72h(model: str = "xgboost"):
    """
    Return a 72-hour recursive forward forecast starting from the last
    known timestamp in the training data.

    Uses src/predict.py which propagates lag features step-by-step
    without consuming future ground truth.

    Query param:
        model: 'xgboost' (default) | 'rf' | 'linear'
    """
    valid_models = {"xgboost", "rf", "linear"}
    if model not in valid_models:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid model '{model}'. Choose from: {list(valid_models)}",
        )

    try:
        from src.predict import predict as do_predict
        fc_series = do_predict(model, n_hours=72)
        records = [
            {"datetime": str(ts), "forecast_mw": round(float(v), 2)}
            for ts, v in fc_series.items()
        ]
        return {
            "model": model,
            "horizon_hours": 72,
            "feature_count": len(FEATURE_COLS),
            "forecast": records,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}")
