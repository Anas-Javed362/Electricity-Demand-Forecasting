import time
import requests
import subprocess
import threading
import sys

def start_api():
    print("Starting API...")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

api_proc = start_api()
time.sleep(5)  # wait for startup

errors = 0

try:
    print("Testing GET /health...")
    r = requests.get("http://localhost:8000/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    
    print("Testing GET /metrics...")
    r = requests.get("http://localhost:8000/metrics")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    
    print("Testing POST /predict (Valid)...")
    r = requests.post("http://localhost:8000/predict", json={"datetime": "2025-06-01 14:00", "model": "xgboost"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print(f"  Result: {r.json()['forecast_mw']} MW")
    
    print("Testing POST /predict (Stress - Invalid JSON)...")
    r = requests.post("http://localhost:8000/predict", data="invalid json")
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    
    print("Testing POST /predict (Stress - Invalid Datetime)...")
    r = requests.post("http://localhost:8000/predict", json={"datetime": "not_a_date", "model": "xgboost"})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    
    print("Testing POST /predict (Stress - Unknown Model)...")
    r = requests.post("http://localhost:8000/predict", json={"datetime": "2025-06-01 14:00", "model": "magic_model"})
    assert r.status_code == 503 or r.status_code == 422, f"Expected error status, got {r.status_code}"
    
    print("Testing GET /forecast/72h...")
    r = requests.get("http://localhost:8000/forecast/72h?model=xgboost")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert len(r.json()["forecast"]) == 72, "Expected 72 hourly forecasts"
    
    print("Testing GET /docs...")
    r = requests.get("http://localhost:8000/docs")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    print("ALL API TESTS PASSED.")
except Exception as e:
    print(f"FAILED: {e}")
    errors += 1
finally:
    api_proc.terminate()
    api_proc.wait()

sys.exit(errors)
