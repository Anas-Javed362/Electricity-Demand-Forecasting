# Electricity Demand Forecasting

## Overview

An end-to-end machine learning project that predicts hourly electricity demand for the **PJM Eastern Interconnection** region using historical consumption data, weather observations, and calendar features.

The project demonstrates the complete ML lifecycle: data acquisition → feature engineering → model training → hyperparameter optimisation → experiment tracking → REST API → interactive dashboard → containerised deployment.

---

## Project Highlights

- **22-feature unified pipeline** — single `src/feature_engineering.py` used by training, API, and dashboard
- **4 forecasting models** — Random Forest, XGBoost (Optuna-tuned), LSTM, Linear Regression
- **Hyperparameter optimisation** with Optuna (50-trial Bayesian search for XGBoost)
- **Model explainability** with SHAP summary plots
- **MLflow experiment tracking** — parameters, metrics, and artifacts logged per run
- **FastAPI REST API** with `/predict`, `/forecast/72h`, `/metrics`, `/health` endpoints
- **Streamlit interactive dashboard** with forecast charts, SHAP viewer, and live predict widget
- **Docker + Docker Compose** for one-command local deployment
- **Render + Streamlit Community Cloud** ready

---

## Business Problem

Accurate electricity demand forecasting helps utility companies, grid operators, and energy markets improve grid stability, optimise generation dispatch, reduce operational costs, and support efficient energy distribution.

---

## Dataset

### Energy Consumption
| Property | Value |
|---|---|
| Source | PJM Hourly Energy Consumption (Kaggle) |
| Region | PJM Eastern Interconnection (PJME) |
| Period | 2002 – 2018 |
| Frequency | Hourly |
| Rows | ~145,000 |

### Weather (Optional)
- Source: Meteostat API (`download_data.py`)
- Variables: Temperature (°C), Relative Humidity (%), Wind Speed (km/h)
- Falls back to regional annual averages if `weather_hourly.csv` is absent

---

## Feature Engineering

All 22 features are defined in **`src/feature_engineering.py`** — the single source of truth used by every training script, the API, and both dashboards.

| Group | Features |
|---|---|
| Calendar | `hour`, `day_of_week`, `month`, `quarter`, `day_of_year`, `week_of_year`, `is_weekend`, `is_holiday` |
| Cyclical | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Lag | `lag_1h`, `lag_24h`, `lag_168h` |
| Rolling | `rolling_mean_24h`, `rolling_mean_168h`, `rolling_std_24h` |
| Weather | `temp`, `rhum`, `wspd`, `temp_sq` |

---

## Models

| Model | File saved | Notes |
|---|---|---|
| Linear Regression | `models/linear_model.pkl` | Baseline |
| Random Forest | `models/rf_model.pkl` | Best overall accuracy |
| XGBoost | `models/xgboost_model.joblib` | Optuna 50-trial HPO + SHAP |
| LSTM | `models/lstm_model.keras` | 24-step look-back, scalers in `lstm_scalers.joblib` |

---

## Model Performance

| Rank | Model             | MAE (MW)  | RMSE (MW) | MAPE (%) | R²       |
|------|-------------------|-----------|-----------|----------|----------|
| 1    | Random Forest     | **306.6** | **~430**  | **~0.97**| **0.9952** |
| 2    | XGBoost           | ~260      | ~370      | ~0.82    | ~0.9960  |
| 3    | LSTM              | ~500      | ~680      | ~1.6     | ~0.988   |
| 4    | Linear Regression | 784.4     | ~1050     | ~2.5     | 0.9731   |

> Metrics are updated after each training run by `python src/evaluate.py`.

---

## REST API Endpoints

| Method | Endpoint        | Description                                        |
|--------|-----------------|----------------------------------------------------|
| GET    | `/health`       | Liveness probe — returns status, version, feature count |
| POST   | `/predict`      | Single-point forecast (XGBoost / RF / Linear)      |
| GET    | `/forecast/72h` | 72-hour recursive forward forecast                 |
| GET    | `/metrics`      | Latest model evaluation scores from `metrics.json` |
| GET    | `/docs`         | Interactive Swagger UI                             |

### Example: POST `/predict`
```json
{
  "datetime": "2025-06-01 14:00",
  "model": "xgboost"
}
```
Response:
```json
{
  "forecast_mw": 34215.7,
  "model": "xgboost",
  "input_datetime": "2025-06-01 14:00",
  "feature_count": 22
}
```

---

## Project Structure

```text
forecasting/
├── api/
│   ├── __init__.py
│   └── main.py                  FastAPI app (all 4 endpoints)
│
├── app/
│   └── dashboard.py             Tabbed Streamlit dashboard (primary)
│
├── src/
│   ├── feature_engineering.py   Single source of truth — 22 features
│   ├── train_xgboost.py         XGBoost + Optuna + SHAP + MLflow
│   ├── train_lstm.py            LSTM (TF/Keras) + MLflow
│   ├── train_prophet.py         Prophet + MLflow (optional)
│   ├── train.py                 One-shot "train all" script
│   ├── evaluate.py              Cross-model evaluation → metrics.json
│   ├── predict.py               Recursive multi-step forecasting
│   └── utils.py                 Data loading utilities
│
├── data/
│   └── PJME_hourly.csv          Energy consumption (place here)
│
├── models/                      Saved model files (git-ignored)
│   ├── xgboost_model.joblib
│   ├── rf_model.pkl
│   ├── linear_model.pkl
│   ├── lstm_model.keras
│   ├── lstm_scalers.joblib
│   ├── shap_values.joblib
│   └── metrics.json
│
├── mlruns/                      MLflow experiment tracking (git-ignored)
├── app.py                       Alternative root-level dashboard
├── download_data.py             Energy + weather data downloader
├── generate_sample_data.py      Sample data generator
├── Dockerfile                   Multi-stage Docker build
├── docker-compose.yml           API + Dashboard + MLflow services
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Anas-Javed362/Electricity-Demand-Forecasting.git
cd Electricity-Demand-Forecasting

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Get the Data

```bash
python download_data.py          # downloads PJME_hourly.csv + weather
# OR manually place PJME_hourly.csv in data/
```

### 3. Train Models

```bash
# Train all models at once:
python src/train.py

# Or individually (with MLflow tracking):
python src/train_xgboost.py      # XGBoost + Optuna HPO (~20–30 min)
python src/train_lstm.py         # LSTM (~10–15 min)
python src/train_prophet.py      # Prophet (optional)
```

### 4. Evaluate

```bash
python src/evaluate.py           # prints metrics table, saves models/metrics.json
```

### 5. Start Services

```bash
# FastAPI
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# Streamlit dashboard
streamlit run app/dashboard.py
# → http://localhost:8501

# MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# → http://localhost:5000
```

---

## Docker Deployment

```bash
docker-compose up --build
```

| Service   | URL                    |
|-----------|------------------------|
| FastAPI   | http://localhost:8000  |
| Streamlit | http://localhost:8501  |
| MLflow    | http://localhost:5000  |

---

## Cloud Deployment

### Render (FastAPI)
1. Push to GitHub
2. Create a new **Web Service** on Render
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Streamlit Community Cloud
1. Push to GitHub
2. Connect repo on [share.streamlit.io](https://share.streamlit.io)
3. Main file: `app/dashboard.py`
4. Set env var `API_URL` to your Render FastAPI URL

---

## MLflow Tracking

MLflow is configured to use a **local file store** (`mlruns/`) with an absolute path to avoid Windows URL-encoding issues. MLflow 3.x requires `MLFLOW_ALLOW_FILE_STORE=true`, which is set automatically in all training scripts.

```bash
# View experiments
mlflow ui --host 0.0.0.0 --port 5000
```

Each training run logs:
- All hyperparameters
- `test_mae`, `test_rmse`, `test_mape`
- SHAP summary plot artifact (XGBoost)
- Trained model artifact

---

## Technology Stack

| Layer | Libraries |
|---|---|
| **ML / DL** | scikit-learn, XGBoost, TensorFlow/Keras, Prophet |
| **HPO** | Optuna |
| **Explainability** | SHAP |
| **MLOps** | MLflow |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **Data** | Pandas, NumPy, Meteostat, Kaggle API |
| **Deployment** | Docker, Docker Compose |

---

## Future Improvements

- LightGBM / CatBoost models
- Transformer-based (TFT) forecasting
- Automated retraining pipeline
- CI/CD with GitHub Actions
- Kubernetes deployment
- Real-time streaming inference

---

## Author

**Mohd Anas Javed Khan**

- GitHub: [Anas-Javed362](https://github.com/Anas-Javed362)
- LinkedIn: [anas-javed-khan](https://www.linkedin.com/in/anas-javed-khan-4019262b6)

---

## License

MIT License