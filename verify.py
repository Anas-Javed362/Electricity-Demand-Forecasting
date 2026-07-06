"""
Quick verification script – checks all modules import correctly.
Run with: python verify.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "src.feature_engineering",
    "src.train_xgboost",
    "src.train_lstm",
    "src.train_prophet",
    "src.evaluate",
    "src.predict",
    "api.main",
]

import importlib

print("=" * 55)
print("PJM Forecasting – Module Import Verification")
print("=" * 55)

all_ok = True
for mod in MODULES:
    try:
        importlib.import_module(mod)
        print(f"  PASS  {mod}")
    except Exception as e:
        print(f"  FAIL  {mod}  -->  {e}")
        all_ok = False

print("=" * 55)
if all_ok:
    print("All modules imported successfully.")
else:
    print("Some modules failed. See errors above.")
    sys.exit(1)
