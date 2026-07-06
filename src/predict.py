"""
src/predict.py
==============
Recursive multi-step forecasting from saved models.

Given a model name and recent history (at least 168 hourly rows),
produces a forecast for the next N hours by reconstructing lag/rolling
features step-by-step — no future ground truth is assumed.

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

from src.utils import load_and_clean
from src.features import FEATURE_COLS, TARGET_COL

MODELS_DIR = "models"
LSTM_LOOKBACK = 24

_US_HOLS = hd.US()


def _calendar_row(dt: pd.Timestamp) -> dict:
    return {
        "hour":       dt.hour,
        "day":        dt.day,
        "month":      dt.month,
        "dayofweek":  dt.dayofweek,
        "weekofyear": dt.isocalendar()[1],
        "is_weekend": int(dt.dayofweek >= 5),
        "is_holiday": int(dt.date() in _US_HOLS),
    }


def forecast_xgb_or_lr_or_rf(model, history: pd.Series, n_hours: int) -> pd.Series:
    """
    Recursive forecasting for tabular models (LR, RF, XGBoost).
    `history` must be a Series indexed by DatetimeIndex (hourly, sorted ascending).
    """
    hist = history.copy()
    results = {}

    for step in range(n_hours):
        next_dt = hist.index[-1] + pd.Timedelta(hours=1)
        row = _calendar_row(next_dt)

        row["lag_1"]   = hist.iloc[-1]
        row["lag_24"]  = hist.iloc[-24] if len(hist) >= 24  else hist.mean()
        row["lag_168"] = hist.iloc[-168] if len(hist) >= 168 else hist.mean()

        recent_24  = hist.iloc[-24:]
        recent_168 = hist.iloc[-168:]
        row["rolling_mean_24"]  = recent_24.mean()
        row["rolling_std_24"]   = recent_24.std(ddof=0)
        row["rolling_mean_168"] = recent_168.mean()
        row["rolling_std_168"]  = recent_168.std(ddof=0)

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
    from sklearn.preprocessing import MinMaxScaler

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
        row["lag_1"]   = raw_hist.iloc[-1]
        row["lag_24"]  = raw_hist.iloc[-24]  if len(raw_hist) >= 24  else raw_hist.mean()
        row["lag_168"] = raw_hist.iloc[-168] if len(raw_hist) >= 168 else raw_hist.mean()
        recent_24  = raw_hist.iloc[-24:]
        recent_168 = raw_hist.iloc[-168:]
        row["rolling_mean_24"]  = recent_24.mean()
        row["rolling_std_24"]   = recent_24.std(ddof=0)
        row["rolling_mean_168"] = recent_168.mean()
        row["rolling_std_168"]  = recent_168.std(ddof=0)

        row_arr = np.array([[row[c] for c in FEATURE_COLS]])
        row_s   = scaler_X.transform(row_arr)
        window.append(row_s[0])

    return pd.Series(results)


def predict(model_name: str, n_hours: int = 24) -> pd.Series:
    """
    Load saved model and return an n_hours forecast from the end of real data.
    """
    df = load_and_clean("data/PJME_hourly.csv")

    if model_name == "linear":
        model = joblib.load(os.path.join(MODELS_DIR, "linear_model.pkl"))
        return forecast_xgb_or_lr_or_rf(model, df[TARGET_COL], n_hours)

    elif model_name == "rf":
        model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        return forecast_xgb_or_lr_or_rf(model, df[TARGET_COL], n_hours)

    elif model_name == "xgboost":
        model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
        return forecast_xgb_or_lr_or_rf(model, df[TARGET_COL], n_hours)

    elif model_name == "lstm":
        scalers    = joblib.load(os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
        import tensorflow as tf
        keras_model = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "lstm_model.h5"), compile=False
        )
        from src.features import build_features
        df_feat = build_features(df)
        return forecast_lstm(scalers["scaler_X"], scalers["scaler_y"],
                              keras_model, df_feat, n_hours)
    else:
        raise ValueError(f"Unknown model '{model_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PJM recursive forecaster")
    parser.add_argument("--model",  default="xgboost",
                        choices=["linear", "rf", "xgboost", "lstm"])
    parser.add_argument("--hours",  type=int, default=24)
    args = parser.parse_args()

    fc = predict(args.model, args.hours)
    print(f"\nForecast ({args.hours}h ahead) using {args.model}:")
    print(fc.round(1).to_string())
