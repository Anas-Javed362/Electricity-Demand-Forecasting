"""
LSTM trainer using TensorFlow / Keras with MLflow experiment tracking.

Architecture: LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1)

Input sequences: 168 time-steps (1 week of hourly data)
Target         : next-hour load_mw

Pipeline
--------
1. Build features via feature_engineering.py
2. MinMaxScale X and y independently
3. Create sliding-window sequences of length SEQUENCE_LEN
4. Train with EarlyStopping + ReduceLROnPlateau
5. Log all params / metrics / model to MLflow
6. Persist Keras model + scalers to models/

Usage
-----
    python src/train_lstm.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # silence oneDNN verbose output

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
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
from sklearn.preprocessing import MinMaxScaler

from src.feature_engineering import build_feature_set, get_train_test, FEATURE_COLS

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MLFLOW_EXPERIMENT = "pjm_forecasting"
# SEQUENCE_LEN=24 avoids OOM on CPU-only Windows machines.
# 168-step sequences on 131k rows require ~7.5GB RAM; 24-step requires ~450MB.
# For production with sufficient RAM, increase to 168.
SEQUENCE_LEN = 24  # 24-hour look-back


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def _build_keras_model(seq_len: int, n_features: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mae",
    )
    return model


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train():
    import tensorflow as tf

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("[INFO] Building feature set ...")
    df = build_feature_set()
    X_train, y_train, X_test, y_test = get_train_test(df)
    print(f"   Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"   Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # ------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    # ------------------------------------------------------------------
    # Sequences
    # ------------------------------------------------------------------
    X_train_seq, y_train_seq = create_sequences(X_train_s, y_train_s, SEQUENCE_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test_s, y_test_s, SEQUENCE_LEN)
    print(f"   Sequence shape: {X_train_seq.shape}")

    params = {
        "sequence_len": SEQUENCE_LEN,
        "lstm_units_1": 128,
        "lstm_units_2": 64,
        "dense_units": 32,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 512,
        "max_epochs": 30,
        "early_stopping_patience": 5,
        "lr_reduce_patience": 3,
    }

    with mlflow.start_run(run_name="lstm") as run:
        mlflow.log_params(params)

        print("[INFO] Building LSTM model ...")
        model = _build_keras_model(SEQUENCE_LEN, X_train_seq.shape[2])
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=params["early_stopping_patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=params["lr_reduce_patience"],
                factor=0.5,
                min_lr=1e-6,
            ),
        ]

        print("[INFO] Training ...")
        history = model.fit(
            X_train_seq,
            y_train_seq,
            validation_split=0.1,
            epochs=params["max_epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )
        epochs_trained = len(history.history["loss"])

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        preds_s = model.predict(X_test_seq).ravel()
        preds = scaler_y.inverse_transform(preds_s.reshape(-1, 1)).ravel()
        y_test_actual = y_test.values[SEQUENCE_LEN:]

        test_mae = float(mean_absolute_error(y_test_actual, preds))
        test_rmse = float(np.sqrt(mean_squared_error(y_test_actual, preds)))
        test_mape = mape(y_test_actual, preds)

        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mape", test_mape)
        mlflow.log_metric("epochs_trained", epochs_trained)

        print(f"📈 Test  MAE : {test_mae:.2f} MW")
        print(f"   Test RMSE : {test_rmse:.2f} MW")
        print(f"   Test MAPE : {test_mape:.2f} %")
        print(f"   Epochs    : {epochs_trained}")

        # ------------------------------------------------------------------
        # Save model + scalers
        # Canonical filenames: lstm_model.keras + lstm_scalers.joblib
        # ------------------------------------------------------------------
        model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
        model.save(model_path)
        scaler_path = os.path.join(MODEL_DIR, "lstm_scalers.joblib")
        joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, scaler_path)

        # Log model to MLflow — use tensorflow flavour (keras module removed in MLflow 3.x)
        try:
            mlflow.tensorflow.log_model(model, "lstm_model")
        except Exception:
            try:
                mlflow.keras.log_model(model, "lstm_model")
            except Exception:
                pass  # Non-critical if mlflow TF integration not available

        print(f"💾 Model saved → {model_path}")
        print(f"   Scalers  → {scaler_path}")
        print(f"📝 MLflow run ID : {run.info.run_id}")

    return model, test_mae, test_rmse, test_mape


if __name__ == "__main__":
    train()
