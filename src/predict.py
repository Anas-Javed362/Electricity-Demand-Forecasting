"""
src/predict.py
==============
Recursive multi-step forecasting from saved models.

Given a model name and recent history (at least 168 hourly rows),
produces a forecast for the next N hours by reconstructing lag/rolling
features step-by-step — no future ground truth is assumed.

Feature pipeline
----------------
Uses src/feature_engineering.py FEATURE_COLS (23 features) — the canonical
single source of truth. All feature names match what the models were trained on.

Usage
-----
    python src/predict.py --model xgboost --hours 24
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import holidays as hd

# ---------------------------------------------------------------------------
# Single source of truth for feature names
# ---------------------------------------------------------------------------
from src.feature_engineering import (
    build_feature_set,
    FEATURE_COLS,
    TARGET_COL,
    TRAIN_TEST_SPLIT_DATE,
)

MODELS_DIR = "models"
# Must match SEQUENCE_LEN in train_lstm.py
LSTM_LOOKBACK = 24  # 24-hour look-back (168 requires ~7.5GB RAM)

_US_HOLS = hd.US()

# Typical historical load statistics for weather/lag approximation at inference
_LOAD_MEAN = 32_000.0
_TEMP_DEFAULT = 10.0
_RHUM_DEFAULT = 60.0
_WSPD_DEFAULT = 8.0


def _calendar_row(dt: pd.Timestamp) -> dict:
    """Build the calendar + cyclical portion of a feature row for a given timestamp."""
    return {
        # Calendar
        "hour":         dt.hour,
        "day_of_week":  dt.dayofweek,
        "month":        dt.month,
        "quarter":      (dt.month - 1) // 3 + 1,
        "day_of_year":  dt.dayofyear,
        "week_of_year": int(dt.isocalendar()[1]),
        "is_weekend":   int(dt.dayofweek >= 5),
        "is_holiday":   int(dt.date() in _US_HOLS),
        # Cyclical encodings
        "hour_sin":  np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos":  np.cos(2 * np.pi * dt.hour / 24),
        "month_sin": np.sin(2 * np.pi * dt.month / 12),
        "month_cos": np.cos(2 * np.pi * dt.month / 12),
        # Weather defaults (approximated for forward forecast)
        "temp":    _TEMP_DEFAULT,
        "rhum":    _RHUM_DEFAULT,
        "wspd":    _WSPD_DEFAULT,
        "temp_sq": _TEMP_DEFAULT ** 2,
    }


def forecast_tabular(model, history: pd.Series, n_hours: int) -> pd.Series:
    """
    Recursive forecasting for tabular models (LR, RF, XGBoost).

    `history` must be a Series indexed by DatetimeIndex (hourly, sorted ascending).
    All 23 FEATURE_COLS are reconstructed step-by-step.
    """
    hist = history.copy()
    results = {}

    for step in range(n_hours):
        next_dt = hist.index[-1] + pd.Timedelta(hours=1)
        row = _calendar_row(next_dt)

        # Lag features (new names matching feature_engineering.py)
        row["lag_1h"]   = hist.iloc[-1]
        row["lag_24h"]  = hist.iloc[-24] if len(hist) >= 24  else hist.mean()
        row["lag_168h"] = hist.iloc[-168] if len(hist) >= 168 else hist.mean()

        # Rolling features (new names matching feature_engineering.py)
        recent_24  = hist.iloc[-24:]
        recent_168 = hist.iloc[-168:]
        row["rolling_mean_24h"]  = recent_24.mean()
        row["rolling_mean_168h"] = recent_168.mean()
        row["rolling_std_24h"]   = recent_24.std(ddof=0)

        # Build a single-row DataFrame guaranteed to have the right column order
        X = pd.DataFrame([row])[FEATURE_COLS]
        pred = float(model.predict(X)[0])
        results[next_dt] = pred
        hist = pd.concat([hist, pd.Series([pred], index=[next_dt])])

    return pd.Series(results)


def forecast_lstm(scaler_X, scaler_y, keras_model, history_df: pd.DataFrame,
                  n_hours: int) -> pd.Series:
    """
    Recursive LSTM forecasting.

    `history_df` must have FEATURE_COLS columns, indexed by DatetimeIndex.
    """
    # Keep last LOOKBACK rows as seed window
    seed_X = history_df[FEATURE_COLS].values[-LSTM_LOOKBACK:]
    seed_X_s = scaler_X.transform(seed_X)
    window = list(seed_X_s)

    # Also keep raw series for lag reconstruction
    raw_hist = history_df[TARGET_COL].copy()
    results = {}

    for step in range(n_hours):
        seq = np.array(window[-LSTM_LOOKBACK:]).reshape(1, LSTM_LOOKBACK, len(FEATURE_COLS))
        pred_s = keras_model.predict(seq, verbose=0)[0, 0]
        pred = float(scaler_y.inverse_transform([[pred_s]])[0, 0])

        next_dt = raw_hist.index[-1] + pd.Timedelta(hours=1)
        results[next_dt] = pred
        raw_hist = pd.concat([raw_hist, pd.Series([pred], index=[next_dt])])

        # Build next feature row
        row = _calendar_row(next_dt)
        row["lag_1h"]   = raw_hist.iloc[-1]
        row["lag_24h"]  = raw_hist.iloc[-24]  if len(raw_hist) >= 24  else raw_hist.mean()
        row["lag_168h"] = raw_hist.iloc[-168] if len(raw_hist) >= 168 else raw_hist.mean()
        recent_24  = raw_hist.iloc[-24:]
        recent_168 = raw_hist.iloc[-168:]
        row["rolling_mean_24h"]  = recent_24.mean()
        row["rolling_mean_168h"] = recent_168.mean()
        row["rolling_std_24h"]   = recent_24.std(ddof=0)

        row_arr = np.array([[row[c] for c in FEATURE_COLS]])
        row_s   = scaler_X.transform(row_arr)
        window.append(row_s[0])

    return pd.Series(results)


def predict(model_name: str, n_hours: int = 24) -> pd.Series:
    """
    Load saved model and return an n_hours forecast from the end of real data.

    Parameters
    ----------
    model_name : str
        One of 'linear', 'rf', 'xgboost', 'lstm'
    n_hours : int
        Forecast horizon in hours

    Returns
    -------
    pd.Series
        Forecast values indexed by future DatetimeIndex
    """
    # Load the full feature set to get the most recent data as context
    df = build_feature_set()

    if model_name == "linear":
        model = joblib.load(os.path.join(MODELS_DIR, "linear_model.pkl"))
        return forecast_tabular(model, df[TARGET_COL], n_hours)

    elif model_name == "rf":
        model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        return forecast_tabular(model, df[TARGET_COL], n_hours)

    elif model_name == "xgboost":
        # Canonical filename: xgboost_model.joblib
        model_path = os.path.join(MODELS_DIR, "xgboost_model.joblib")
        if not os.path.exists(model_path):
            # Backward-compat fallback for old .pkl files
            model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
        model = joblib.load(model_path)
        return forecast_tabular(model, df[TARGET_COL], n_hours)

    elif model_name == "lstm":
        # Canonical filenames: lstm_model.keras + lstm_scalers.joblib
        model_path = os.path.join(MODELS_DIR, "lstm_model.keras")
        scaler_path = os.path.join(MODELS_DIR, "lstm_scalers.joblib")
        # Backward-compat fallback
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, "lstm_model.h5")
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")

        scalers = joblib.load(scaler_path)
        import tensorflow as tf
        keras_model = tf.keras.models.load_model(model_path, compile=False)
        return forecast_lstm(scalers["scaler_X"], scalers["scaler_y"], keras_model, df, n_hours)

    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose: 'linear', 'rf', 'xgboost', 'lstm'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PJM recursive forecaster")
    parser.add_argument("--model", default="xgboost",
                        choices=["linear", "rf", "xgboost", "lstm"])
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    fc = predict(args.model, args.hours)
    print(f"\nForecast ({args.hours}h ahead) using {args.model}:")
    print(fc.round(1).to_string())
