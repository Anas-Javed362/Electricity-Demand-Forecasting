"""
src/features.py
===============
Feature engineering for PJME energy consumption forecasting.
Reused by both src/train.py and app.py (zero leakage guaranteed).

Public API
----------
build_features(df)          -> DataFrame with all features + target
FEATURE_COLS                -> ordered list of feature column names
TARGET_COL                  -> 'PJME_MW'
TRAIN_CUTOFF                -> chronological split date string
get_xy(df)                  -> (X_train, y_train, X_test, y_test)
make_sequences(X, y, n=24)  -> (Xs, ys) numpy arrays for LSTM
"""
import numpy as np
import pandas as pd
import holidays as hd

TARGET_COL = "PJME_MW"
# Final ~1 year as test set (matches well-known convention on this dataset)
TRAIN_CUTOFF = "2017-01-01"

FEATURE_COLS = [
    "hour", "day", "month", "dayofweek", "weekofyear", "is_weekend", "is_holiday",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_std_24",
    "rolling_mean_168", "rolling_std_168",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar, holiday, lag and rolling features to a DataFrame
    whose index is a DatetimeIndex and which contains TARGET_COL.

    Rolling/lag features use only PAST values (shift(1) before rolling)
    to ensure zero data leakage.
    """
    df = df.copy()
    us_hols = hd.US()

    # -- Calendar --
    df["hour"]       = df.index.hour
    df["day"]        = df.index.day
    df["month"]      = df.index.month
    df["dayofweek"]  = df.index.dayofweek
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = df.index.to_series().apply(
        lambda x: 1 if x.date() in us_hols else 0
    )

    # -- Lag features (shift ensures no leakage) --
    df["lag_1"]   = df[TARGET_COL].shift(1)
    df["lag_24"]  = df[TARGET_COL].shift(24)
    df["lag_168"] = df[TARGET_COL].shift(168)

    # -- Rolling features (shift(1) first, then roll on past only) --
    shifted = df[TARGET_COL].shift(1)
    df["rolling_mean_24"]  = shifted.rolling(24).mean()
    df["rolling_std_24"]   = shifted.rolling(24).std()
    df["rolling_mean_168"] = shifted.rolling(168).mean()
    df["rolling_std_168"]  = shifted.rolling(168).std()

    # Drop NaN rows created by lagging (only first 168 rows)
    df = df.dropna(subset=FEATURE_COLS)
    return df


def get_xy(df: pd.DataFrame):
    """
    Chronological train/test split at TRAIN_CUTOFF.
    Returns X_train, y_train, X_test, y_test (all pandas objects).
    """
    df_feat = build_features(df)
    train = df_feat[df_feat.index < TRAIN_CUTOFF]
    test  = df_feat[df_feat.index >= TRAIN_CUTOFF]

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL]

    print(f"[features] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, y_train, X_test, y_test


def make_sequences(X: np.ndarray, y: np.ndarray, n_steps: int = 24):
    """
    Create sliding window sequences of length n_steps for LSTM.
    Returns (Xs shape [samples, n_steps, features], ys shape [samples]).
    """
    Xs, ys = [], []
    for i in range(n_steps, len(X)):
        Xs.append(X[i - n_steps : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)
