"""
src/train.py
============
Trains ALL four models on real PJME_hourly.csv and writes:
  models/linear_model.pkl
  models/rf_model.pkl
  models/xgboost_model.joblib   ← .joblib (canonical)
  models/lstm_model.keras        ← .keras  (canonical)
  models/lstm_scalers.joblib     ← scalers (canonical)
  models/metrics.json

Usage
-----
    python src/train.py

Models
------
  1. Linear Regression  (baseline)
  2. Random Forest Regressor
  3. XGBoost Regressor  (tuned: n_estimators, max_depth, learning_rate)
  4. LSTM               (168-hr lookback, MinMaxScaler fit on train only)

All models evaluated on the same chronological test set.
Metrics: MAE, RMSE, MAPE, R^2 -> models/metrics.json

Feature pipeline
----------------
Uses src/feature_engineering.py (23 features) — the canonical single source of truth.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# ---------------------------------------------------------------------------
# Single source of truth for feature engineering
# ---------------------------------------------------------------------------
from src.feature_engineering import (
    build_feature_set,
    get_train_test,
    FEATURE_COLS,
    TARGET_COL,
)

MODELS_DIR = "models"
IMAGES_DIR = "images"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

LSTM_LOOKBACK = 168  # 1 week (matches train_lstm.py)
LSTM_EPOCHS   = 20
LSTM_BATCH    = 512


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_metrics(y_true, y_pred, name):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mp   = mape(y_true, y_pred)
    r2   = float(r2_score(y_true, y_pred))
    print(f"  [{name}]  MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mp:.2f}%  R2={r2:.4f}")
    return {"MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE": round(mp,4), "R2": round(r2,4)}


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def plot_actual_vs_pred(y_test, preds_dict, best_model):
    fig, ax = plt.subplots(figsize=(14, 4))
    sample = y_test.iloc[:2000]
    ax.plot(sample.index, sample.values, label="Actual", color="#1f77b4", linewidth=1)
    pred = preds_dict[best_model][:2000]
    ax.plot(sample.index, pred, label=f"{best_model} (predicted)", color="#ff7f0e", linewidth=1, linestyle="--")
    ax.set_title(f"Actual vs Predicted – {best_model} (first 2000 test hours)")
    ax.set_ylabel("Load (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "actual_vs_predicted.png"), dpi=120)
    plt.close()
    print(f"  Saved images/actual_vs_predicted.png")

def plot_residuals(y_test, preds_dict, best_model):
    residuals = np.array(y_test) - np.array(preds_dict[best_model])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(residuals[:2000], color="#d62728", linewidth=0.5, alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"Residuals over time – {best_model}")
    axes[0].set_ylabel("Residual (MW)")
    axes[1].hist(residuals, bins=80, color="#9467bd", edgecolor="none", alpha=0.8)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Error (MW)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "residuals.png"), dpi=120)
    plt.close()
    print(f"  Saved images/residuals.png")

def plot_feature_importance(xgb_model):
    imp = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="#2ca02c")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=120)
    plt.close()
    print(f"  Saved images/feature_importance.png")

def plot_model_comparison(metrics_dict):
    models = list(metrics_dict.keys())
    maes  = [metrics_dict[m]["MAE"]  for m in models]
    rmses = [metrics_dict[m]["RMSE"] for m in models]
    mapes = [metrics_dict[m]["MAPE"] for m in models]

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, title, color in zip(
        axes,
        [maes, rmses, mapes],
        ["MAE (MW)", "RMSE (MW)", "MAPE (%)"],
        ["#1f77b4", "#ff7f0e", "#2ca02c"],
    ):
        bars = ax.bar(x, vals, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_title(title)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "model_comparison.png"), dpi=120)
    plt.close()
    print(f"  Saved images/model_comparison.png")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_linear(X_train, y_train, X_test, y_test):
    print("\n[1/4] Linear Regression (baseline)...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    joblib.dump(model, os.path.join(MODELS_DIR, "linear_model.pkl"))
    return model, preds, compute_metrics(y_test, preds, "LinearRegression")


def train_rf(X_train, y_train, X_test, y_test):
    print("\n[2/4] Random Forest (n_estimators=200, max_depth=16)...")
    model = RandomForestRegressor(n_estimators=200, max_depth=16,
                                  min_samples_leaf=4, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    joblib.dump(model, os.path.join(MODELS_DIR, "rf_model.pkl"))
    return model, preds, compute_metrics(y_test, preds, "RandomForest")


def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n[3/4] XGBoost (n_estimators=500, max_depth=6, lr=0.05)...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    preds = model.predict(X_test)
    # Canonical filename: xgboost_model.joblib
    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    return model, preds, compute_metrics(y_test, preds, "XGBoost")


def train_lstm(X_train, y_train, X_test, y_test):
    print(f"\n[4/4] LSTM (lookback={LSTM_LOOKBACK}h, epochs={LSTM_EPOCHS})...")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    # Scale – fit on train only
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_tr_s = scaler_X.fit_transform(X_train)
    X_te_s = scaler_X.transform(X_test)
    y_tr_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_te_s = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    # Sequences
    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)

    X_tr_seq, y_tr_seq = make_sequences(X_tr_s, y_tr_s, LSTM_LOOKBACK)
    X_te_seq, y_te_seq = make_sequences(X_te_s, y_te_s, LSTM_LOOKBACK)

    # Model
    n_feat = X_tr_seq.shape[2]
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=(LSTM_LOOKBACK, n_feat)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
    ]
    model.fit(X_tr_seq, y_tr_seq,
              validation_split=0.1,
              epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
              callbacks=cb, verbose=1)

    preds_s = model.predict(X_te_seq, verbose=0).ravel()
    preds   = scaler_y.inverse_transform(preds_s.reshape(-1, 1)).ravel()
    # LSTM predictions are offset by LSTM_LOOKBACK
    y_test_aligned = y_test.values[LSTM_LOOKBACK:]

    # Canonical filenames: lstm_model.keras + lstm_scalers.joblib
    model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
    joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y},
                os.path.join(MODELS_DIR, "lstm_scalers.joblib"))

    m = compute_metrics(y_test_aligned, preds, "LSTM")
    return model, preds, m, len(y_test_aligned)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PJM Energy Demand Forecasting – Training Pipeline")
    print("=" * 60)

    # Load data using unified feature engineering pipeline
    print("📊 Building feature set …")
    df = build_feature_set()
    X_train, y_train, X_test, y_test = get_train_test(df)
    print(f"   Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"   Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    metrics = {}
    preds_dict = {}

    # Train all models
    lr_model, lr_preds, metrics["Linear Regression"] = train_linear(X_train, y_train, X_test, y_test)
    preds_dict["Linear Regression"] = lr_preds

    rf_model, rf_preds, metrics["Random Forest"] = train_rf(X_train, y_train, X_test, y_test)
    preds_dict["Random Forest"] = rf_preds

    xgb_model, xgb_preds, metrics["XGBoost"] = train_xgboost(X_train, y_train, X_test, y_test)
    preds_dict["XGBoost"] = xgb_preds

    lstm_model, lstm_preds, metrics["LSTM"], lstm_len = train_lstm(X_train, y_train, X_test, y_test)
    preds_dict["LSTM"] = lstm_preds

    # Find best model by MAE
    best = min(metrics, key=lambda m: metrics[m]["MAE"])
    print(f"\nBest model by MAE: {best}")

    # Save metrics.json
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {metrics_path}")

    # Evaluation plots
    print("\nGenerating evaluation plots...")
    best_plot_model = "XGBoost" if "XGBoost" in preds_dict else best
    plot_actual_vs_pred(y_test, preds_dict, best_plot_model)
    plot_residuals(y_test, preds_dict, best_plot_model)
    plot_feature_importance(xgb_model)
    plot_model_comparison(metrics)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE – Model Metrics Summary")
    print("=" * 60)
    header = f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R2':>8}"
    print(header)
    print("-" * 60)
    for name, m in metrics.items():
        print(f"{name:<20} {m['MAE']:>8.1f} {m['RMSE']:>8.1f} {m['MAPE']:>8.2f} {m['R2']:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
