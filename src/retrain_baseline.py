"""Quick script to retrain Linear Regression and Random Forest on the new feature pipeline."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.feature_engineering import build_feature_set, get_train_test

print("Building feature set...")
df = build_feature_set()
X_train, y_train, X_test, y_test = get_train_test(df)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {X_train.shape[1]}")

def mape(yt, yp):
    mask = yt != 0
    return float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100)

def metrics(yt, yp):
    return {
        "MAE": round(float(mean_absolute_error(yt, yp)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 2),
        "MAPE": round(mape(yt, yp), 4),
        "R2": round(float(r2_score(yt, yp)), 4),
    }

results = {}

print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
p = lr.predict(X_test)
joblib.dump(lr, "models/linear_model.pkl")
results["Linear Regression"] = metrics(y_test.values, p)
print(f"  MAE={results['Linear Regression']['MAE']:.1f}  R2={results['Linear Regression']['R2']:.4f}")

print("\nTraining Random Forest (n_estimators=200)...")
rf = RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=4, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
p = rf.predict(X_test)
joblib.dump(rf, "models/rf_model.pkl")
results["Random Forest"] = metrics(y_test.values, p)
print(f"  MAE={results['Random Forest']['MAE']:.1f}  R2={results['Random Forest']['R2']:.4f}")

# Write metrics (will be updated when XGBoost finishes)
mpath = "models/metrics.json"
existing = {}
if os.path.exists(mpath):
    with open(mpath) as f:
        existing = json.load(f)
existing.update(results)
with open(mpath, "w") as f:
    json.dump(existing, f, indent=2)
print(f"\nSaved {mpath}")
print("Done - Linear + RF retrained on 22-feature pipeline")
