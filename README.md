# Electricity Demand Forecasting

## Overview

Electricity Demand Forecasting is an end-to-end machine learning project that predicts hourly electricity demand for the PJM Eastern Interconnection region using historical electricity consumption, weather observations, and calendar-based features.

The project demonstrates the complete machine learning lifecycle, including data acquisition, preprocessing, feature engineering, model development, hyperparameter optimization, experiment tracking, REST API deployment, interactive visualization, and containerized deployment.

---

## Project Highlights

* End-to-end machine learning pipeline for time series forecasting
* Historical electricity demand forecasting (2002–2018)
* Integration of weather and calendar data
* Automated feature engineering pipeline
* Four forecasting models

  * Random Forest
  * XGBoost
  * LSTM
  * Linear Regression
* Hyperparameter optimization using Optuna
* Model explainability using SHAP
* TimeSeriesSplit cross-validation
* MLflow experiment tracking
* FastAPI REST API
* Streamlit interactive dashboard
* Dockerized deployment

---

## Business Problem

Accurate electricity demand forecasting is essential for utility companies, power grid operators, and energy markets. Reliable forecasts help improve grid stability, optimize energy generation, reduce operational costs, and support efficient energy distribution.

This project develops machine learning models capable of forecasting hourly electricity demand using historical consumption data combined with weather and calendar information.

---

## Dataset

### Energy Consumption Dataset

* Source: PJM Hourly Energy Consumption Dataset
* Region: PJM Eastern Interconnection (PJME)
* Duration: 2002–2018
* Frequency: Hourly

### Weather Dataset

* Source: Meteostat

Weather variables include:

* Temperature
* Relative Humidity
* Wind Speed

---

## Machine Learning Pipeline

```text
Raw Energy Data
        │
        ▼
Weather Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
        │
        ▼
Hyperparameter Optimization
        │
        ▼
Model Evaluation
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
FastAPI Deployment
        │
        ▼
Streamlit Dashboard
```

---

## Feature Engineering

### Calendar Features

* Hour
* Day of Week
* Month
* Quarter
* Day of Year
* Week of Year
* Weekend Indicator
* US Holiday Indicator

### Cyclical Features

* Hour (Sine/Cosine)
* Month (Sine/Cosine)

### Lag Features

* Previous Hour
* Previous 24 Hours
* Previous Week (168 Hours)

### Rolling Statistics

* 24-Hour Rolling Mean
* 168-Hour Rolling Mean
* 24-Hour Rolling Standard Deviation

### Weather Features

* Temperature
* Relative Humidity
* Wind Speed
* Temperature Squared

---

## Models Implemented

### Random Forest

* Ensemble Decision Tree Regressor
* Feature Importance Analysis
* Best Performing Model

### XGBoost

* Gradient Boosting Regressor
* Optuna Hyperparameter Optimization
* SHAP Explainability

### LSTM

* Three-layer LSTM Network
* MinMax Feature Scaling
* EarlyStopping
* TensorFlow/Keras

### Linear Regression

* Baseline Forecasting Model
* Performance Benchmarking

---

## Model Evaluation

The models were evaluated using walk-forward validation with `TimeSeriesSplit`, ensuring the temporal order of observations was maintained throughout training and testing.

### Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* Coefficient of Determination (R²)

---

## Model Performance

| Rank | Model             | MAE (MW)  | RMSE (MW) | MAPE (%) | R² Score   |
| ---- | ----------------- | --------- | --------- | -------- | ---------- |
| 1    | Random Forest     | **310.8** | **429.6** | **0.98** | **0.9951** |
| 2    | XGBoost           | 325.4     | 435.2     | 1.03     | 0.9949     |
| 3    | LSTM              | 513.3     | 686.9     | 1.63     | 0.9874     |
| 4    | Linear Regression | 966.1     | 1229.9    | 3.12     | 0.9597     |

### Performance Summary

The Random Forest model achieved the best overall forecasting performance, producing the lowest prediction errors and the highest coefficient of determination. XGBoost delivered comparable results, while LSTM effectively captured temporal patterns but exhibited higher forecasting errors. Linear Regression served as the baseline model and demonstrated significantly lower predictive performance due to its inability to model complex nonlinear relationships.

---

## REST API

The project exposes prediction services through FastAPI.

### Endpoints

| Method | Endpoint        | Description                            |
| ------ | --------------- | -------------------------------------- |
| GET    | `/health`       | Health check endpoint                  |
| POST   | `/predict`      | Generate electricity demand prediction |
| GET    | `/forecast/72h` | Generate 72-hour forecast              |
| GET    | `/metrics`      | Retrieve model evaluation metrics      |
| GET    | `/docs`         | Interactive Swagger API documentation  |

---

## Interactive Dashboard

The Streamlit dashboard provides:

* Historical demand visualization
* Forecast visualization
* Model comparison
* Weather analysis
* Interactive filtering
* Performance metrics
* Prediction interface

---

## Experiment Tracking

MLflow is used for:

* Experiment logging
* Parameter tracking
* Metric tracking
* Model versioning
* Artifact storage
* Model comparison

---

## Project Structure

```text
forecasting/
│
├── api/
│   └── main.py
│
├── app/
│   └── dashboard.py
│
├── data/
│
├── mlruns/
│
├── models/
│
├── src/
│   ├── feature_engineering.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── train_linear_regression.py
│   ├── evaluate.py
│   └── predict.py
│
├── download_data.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Local Setup

### Clone the Repository

```bash
git clone https://github.com/Anas-Javed362/pjm-energy-forecasting.git

cd pjm-energy-forecasting
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download Dataset

Run:

```bash
python download_data.py
```

or manually place the PJME dataset inside the `data/` directory.

---

## Train Models

Random Forest

```bash
python src/train_random_forest.py
```

XGBoost

```bash
python src/train_xgboost.py
```

LSTM

```bash
python src/train_lstm.py
```

Linear Regression

```bash
python src/train_linear_regression.py
```

---

## Evaluate Models

```bash
python src/evaluate.py
```

---

## Run FastAPI

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```
http://localhost:8000
```

API documentation:

```
http://localhost:8000/docs
```

---

## Run Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Dashboard URL:

```
http://localhost:8501
```

---

## MLflow

Start the MLflow UI:

```bash
mlflow ui
```

Open:

```
http://localhost:5000
```

---

## Docker Deployment

Build and run all services:

```bash
docker-compose up --build
```

Services:

| Service   | URL                   |
| --------- | --------------------- |
| FastAPI   | http://localhost:8000 |
| Streamlit | http://localhost:8501 |
| MLflow    | http://localhost:5000 |

---

## Technology Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* XGBoost
* TensorFlow
* Keras
* Optuna
* SHAP

### Data Processing

* Pandas
* NumPy
* Meteostat
* Kaggle API

### MLOps

* MLflow
* Docker
* Docker Compose

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* Streamlit
* Plotly

---

## Skills Demonstrated

* Time Series Forecasting
* Machine Learning
* Deep Learning
* Feature Engineering
* Hyperparameter Optimization
* Explainable AI
* REST API Development
* MLOps
* Experiment Tracking
* Docker
* Data Visualization
* Model Evaluation
* Cross Validation

---

## Future Improvements

* LightGBM implementation
* CatBoost implementation
* Transformer-based forecasting models
* Automated retraining pipeline
* CI/CD with GitHub Actions
* Kubernetes deployment
* Cloud deployment using AWS or Azure
* Real-time streaming inference

---

## Repository Contents

```text
api/                    FastAPI backend
app/                    Streamlit dashboard
data/                   Raw datasets
models/                 Trained models
mlruns/                 MLflow experiments
src/                    Training and evaluation scripts
Dockerfile              Docker configuration
docker-compose.yml      Multi-container deployment
requirements.txt        Project dependencies
README.md               Project documentation
```

---

## Author

**Mohd Anas Javed Khan**

GitHub:
https://github.com/Anas-Javed362

LinkedIn:
https://www.linkedin.com/in/anas-javed-khan-4019262b6

---

## License

This project is licensed under the MIT License.
#   E l e c t r i c i t y - D e m a n d - F o r e c a s t i n g  
 