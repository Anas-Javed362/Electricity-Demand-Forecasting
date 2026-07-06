"""
Feature engineering pipeline for PJM energy consumption forecasting.

Produces a fully-featured DataFrame ready for model training with:
  - Calendar features (hour, day-of-week, month, holidays, cyclical encodings)
  - Lag features (1h, 24h, 168h) and rolling statistics
  - Weather features (temp, rhum, wspd) with quadratic cooling/heating proxy
  - TimeSeriesSplit helper for walk-forward validation
"""
import os
import pandas as pd
import numpy as np
import holidays
from sklearn.model_selection import TimeSeriesSplit

DATA_DIR = "data"
PJME_FILE = os.path.join(DATA_DIR, "PJME_hourly.csv")
WEATHER_FILE = os.path.join(DATA_DIR, "weather_hourly.csv")
TRAIN_TEST_SPLIT_DATE = "2017-01-01"

# Ordered feature list used consistently across all models
FEATURE_COLS = [
    "hour", "day_of_week", "month", "quarter", "day_of_year", "week_of_year",
    "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_mean_168h", "rolling_std_24h",
    "temp", "rhum", "wspd", "temp_sq",
]
TARGET_COL = "load_mw"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_merge() -> pd.DataFrame:
    """Load energy + weather CSVs and merge on Datetime index."""
    df_energy = pd.read_csv(PJME_FILE, parse_dates=["Datetime"])
    df_energy = df_energy.rename(columns={"PJME_MW": "load_mw"})
    df_energy = df_energy.set_index("Datetime").sort_index()

    if os.path.exists(WEATHER_FILE):
        df_weather = pd.read_csv(WEATHER_FILE, parse_dates=["Datetime"])
        df_weather = df_weather.set_index("Datetime").sort_index()
        df = df_energy.join(df_weather, how="left")
        df[["temp", "rhum", "wspd"]] = (
            df[["temp", "rhum", "wspd"]]
            .ffill()
            .bfill()
        )
    else:
        print("⚠️  weather_hourly.csv not found – weather features will be zero.")
        df = df_energy.copy()
        df["temp"] = 10.0
        df["rhum"] = 60.0
        df["wspd"] = 8.0

    return df


# ---------------------------------------------------------------------------
# Feature constructors
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    us_hols = holidays.US()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["day_of_year"] = df.index.dayofyear
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = df.index.to_series().apply(
        lambda x: 1 if x.date() in us_hols else 0
    )
    # Cyclical encodings (avoids ordinal discontinuities)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lag_1h"] = df["load_mw"].shift(1)
    df["lag_24h"] = df["load_mw"].shift(24)
    df["lag_168h"] = df["load_mw"].shift(168)
    df["rolling_mean_24h"] = df["load_mw"].shift(1).rolling(24).mean()
    df["rolling_mean_168h"] = df["load_mw"].shift(1).rolling(168).mean()
    df["rolling_std_24h"] = df["load_mw"].shift(1).rolling(24).std()
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df["temp_sq"] = df["temp"] ** 2  # Proxy for heating / cooling demand
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_set() -> pd.DataFrame:
    """Full pipeline: load → calendar → lag → weather → dropna."""
    df = load_and_merge()
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_weather_features(df)
    df = df.dropna()
    return df


def get_train_test(df: pd.DataFrame):
    """Hard cutoff split at TRAIN_TEST_SPLIT_DATE."""
    train = df[df.index < TRAIN_TEST_SPLIT_DATE]
    test = df[df.index >= TRAIN_TEST_SPLIT_DATE]
    return (
        train[FEATURE_COLS], train[TARGET_COL],
        test[FEATURE_COLS], test[TARGET_COL],
    )


def get_tscv(n_splits: int = 5) -> TimeSeriesSplit:
    """Walk-forward cross-validation splitter."""
    return TimeSeriesSplit(n_splits=n_splits)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_feature_set()
    print(f"✅ Feature set built: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Date range : {df.index.min()} → {df.index.max()}")
    print(f"   Features   : {FEATURE_COLS}")
    X_train, y_train, X_test, y_test = get_train_test(df)
    print(f"   Train      : {len(X_train):,} rows")
    print(f"   Test       : {len(X_test):,} rows")
