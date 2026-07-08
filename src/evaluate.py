"""
src/evaluate.py
===============
Cross-model evaluation on the held-out test set.

Loads all trained models from models/ and evaluates them on the same
chronological test set (2017-01-01 onward) used during training.

Feature pipeline
----------------
Uses src/feature_engineering.py (23 features) — the canonical single source of truth.

Model filenames (canonical)
---------------------------
  xgboost_model.joblib   (falls back to xgboost_model.pkl)
  rf_model.pkl
  linear_model.pkl
  lstm_model.keras       (falls back to lstm_model.h5)
  lstm_scalers.joblib    (falls back to lstm_scaler.pkl)

Outputs
-------
  - Console table: MAE / RMSE / MAPE / R² for each model
  - models/model_metrics.csv   – machine-readable
  - models/metrics.json        – refreshed (dashboard reads this)

Usage
-----
    python src/evaluate.py
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Single source of truth for feature engineering
# ---------------------------------------------------------------------------
from src.feature_engineering import build_feature_set, get_train_test, FEATURE_COLS

MODEL_DIR = "models"


# ── Metric helpers ────────────────────────────────────────────────────────────

def _mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def _metrics(y_true, y_pred, name):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mp   = _mape(y_true, y_pred)
    r2   = float(r2_score(y_true, y_pred))
    print(f"  {name:<22}  MAE={mae:>8.1f}  RMSE={rmse:>8.1f}  MAPE={mp:>6.2f}%  R2={r2:.4f}")
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mp, 4), "R2": round(r2, 4)}


def _try_paths(*paths):
    """Return the first existing path from the list, or None."""
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ── Individual evaluators ─────────────────────────────────────────────────────

def eval_linear(X_test, y_test):
    p = _try_paths(os.path.join(MODEL_DIR, "linear_model.pkl"))
    if not p:
        print("  [SKIP] linear_model.pkl not found."); return None
    model = joblib.load(p)
    return _metrics(y_test, model.predict(X_test), "Linear Regression")

def eval_rf(X_test, y_test):
    p = _try_paths(os.path.join(MODEL_DIR, "rf_model.pkl"))
    if not p:
        print("  [SKIP] rf_model.pkl not found."); return None
    model = joblib.load(p)
    return _metrics(y_test, model.predict(X_test), "Random Forest")

def eval_xgboost(X_test, y_test):
    # Try canonical (.joblib) first, fall back to legacy (.pkl)
    p = _try_paths(
        os.path.join(MODEL_DIR, "xgboost_model.joblib"),
        os.path.join(MODEL_DIR, "xgboost_model.pkl"),
    )
    if not p:
        print("  [SKIP] xgboost_model.joblib / .pkl not found."); return None
    model = joblib.load(p)
    return _metrics(y_test, model.predict(X_test), "XGBoost")

def eval_lstm(X_test, y_test):
    # Try canonical filenames first, fall back to legacy
    mp = _try_paths(
        os.path.join(MODEL_DIR, "lstm_model.keras"),
        os.path.join(MODEL_DIR, "lstm_model.h5"),
    )
    sp = _try_paths(
        os.path.join(MODEL_DIR, "lstm_scalers.joblib"),
        os.path.join(MODEL_DIR, "lstm_scaler.pkl"),
    )
    if not mp or not sp:
        print("  [SKIP] lstm_model.keras / lstm_scalers.joblib not found."); return None
    try:
        import tensorflow as tf
        scalers = joblib.load(sp)
        model   = tf.keras.models.load_model(mp, compile=False)

        # Must match SEQUENCE_LEN in train_lstm.py
        LOOKBACK = 24  # 24-hour look-back
        X_s = scalers["scaler_X"].transform(X_test.values)
        y_s = scalers["scaler_y"].transform(y_test.values.reshape(-1, 1)).ravel()

        # Create sequences
        Xs, ys = [], []
        for i in range(len(X_s) - LOOKBACK):
            Xs.append(X_s[i : i + LOOKBACK])
            ys.append(y_s[i + LOOKBACK])
        Xs = np.array(Xs)

        preds_s = model.predict(Xs, verbose=0).ravel()
        preds   = scalers["scaler_y"].inverse_transform(preds_s.reshape(-1, 1)).ravel()
        y_aligned = y_test.values[LOOKBACK:]
        return _metrics(y_aligned, preds, "LSTM")
    except Exception as e:
        print(f"  [FAIL] LSTM: {e}"); return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 66
    print(f"\n{sep}")
    print("  PJM Energy Demand Forecasting — Model Evaluation")
    print(sep)

    # Use unified feature engineering pipeline
    df = build_feature_set()
    X_train, y_train, X_test, y_test = get_train_test(df)
    print(f"  Features  : {len(FEATURE_COLS)} columns")
    print(f"  Test set  : {len(X_test):,} rows  ({X_test.index.min().date()} -> {X_test.index.max().date()})\n")

    results = {}

    m = eval_linear(X_test, y_test)
    if m: results["Linear Regression"] = m

    m = eval_rf(X_test, y_test)
    if m: results["Random Forest"] = m

    m = eval_xgboost(X_test, y_test)
    if m: results["XGBoost"] = m

    m = eval_lstm(X_test, y_test)
    if m: results["LSTM"] = m

    if not results:
        print("\nNo models found. Run: python src/train.py")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    df_out = (
        pd.DataFrame(results).T
        .rename(columns={"MAE": "MAE (MW)", "RMSE": "RMSE (MW)", "MAPE": "MAPE (%)", "R2": "R2"})
        .sort_values("MAE (MW)")
    )

    print(f"\n{sep}")
    print("  MODEL COMPARISON SUMMARY  (sorted by MAE, best first)")
    print(sep)
    print(df_out.to_string(float_format=lambda x: f"{x:.2f}"))
    print(sep)

    best = df_out.index[0]
    print(f"\n  Best model: {best}  (MAE = {df_out.loc[best,'MAE (MW)']:.1f} MW)")

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    csv_path = os.path.join(MODEL_DIR, "model_metrics.csv")
    df_out.to_csv(csv_path)
    print(f"  Saved -> {csv_path}")

    json_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {json_path}\n")


if __name__ == "__main__":
    main()
