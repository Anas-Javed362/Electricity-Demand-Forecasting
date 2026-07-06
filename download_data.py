"""
Download_data.py – fixed for Windows cp1252 terminal encoding.
All emoji replaced with ASCII equivalents for compatibility.
"""
import os
import sys
from datetime import datetime

def check_kaggle_credentials():
    home = os.path.expanduser("~")
    kaggle_path = os.path.join(home, ".kaggle", "kaggle.json")
    return os.path.exists(kaggle_path)

def download_pjm_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    target_file = os.path.join(data_dir, "PJME_hourly.csv")

    if os.path.exists(target_file):
        print(f"[OK] PJM load data already exists at: {target_file}")
        return True

    print("[...] Attempting to download PJM Hourly Energy Consumption dataset from Kaggle...")

    if not check_kaggle_credentials():
        print("\n" + "="*80)
        print("[WARN] KAGGLE API CREDENTIALS NOT FOUND!")
        print("Please set up Kaggle API credentials to download the data automatically:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' to download kaggle.json")
        print(f"3. Place kaggle.json in: {os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')}")
        print("\nAlternatively, download the dataset manually:")
        print("1. Go to: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption")
        print("2. Download the zip archive and extract 'PJME_hourly.csv'")
        print(f"3. Save the file directly as: {os.path.abspath(target_file)}")
        print("="*80 + "\n")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[OK] Kaggle API authenticated successfully. Downloading dataset...")
        api.dataset_download_file(
            dataset="robikscube/hourly-energy-consumption",
            file_name="PJME_hourly.csv",
            path=data_dir,
            force=False,
            quiet=False
        )
        zip_file = os.path.join(data_dir, "PJME_hourly.csv.zip")
        if os.path.exists(zip_file):
            import zipfile
            print("[...] Extracting PJME_hourly.csv from zip...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_file)
            print("[OK] Removed zip archive.")

        if os.path.exists(target_file):
            print("[OK] PJM load data downloaded and extracted successfully!")
            return True
        else:
            print("[ERR] Download finished but PJME_hourly.csv was not found.")
            return False
    except Exception as e:
        print(f"[ERR] Error downloading PJM data via Kaggle API: {e}")
        print("Please download manually and place PJME_hourly.csv in data/ directory.")
        return False

def download_weather_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    target_file = os.path.join(data_dir, "weather_hourly.csv")

    if os.path.exists(target_file):
        print(f"[OK] Weather data already exists at: {target_file}")
        return True

    print("[...] Fetching historical weather data via Meteostat...")
    try:
        from meteostat import Hourly, Stations

        start = datetime(2002, 1, 1)
        end = datetime(2018, 12, 31, 23)

        print("[...] Searching for weather station near Philadelphia...")
        stations = Stations()
        stations = stations.nearby(39.9526, -75.1652)
        station_df = stations.fetch(1)

        if station_df.empty:
            print("[WARN] Could not find a weather station nearby. Falling back to station ID 72408.")
            station_id = "72408"
        else:
            station_id = station_df.index[0]
            station_name = station_df.iloc[0]['name']
            print(f"[OK] Selected station: {station_name} (ID: {station_id})")

        print(f"[...] Downloading hourly weather from {start.date()} to {end.date()}...")
        data = Hourly(station_id, start, end)
        df = data.fetch()

        if df.empty:
            print("[ERR] Weather data fetch returned an empty DataFrame.")
            return False

        df = df.reset_index()
        required_cols = ['time', 'temp', 'rhum', 'wspd']
        for col in required_cols:
            if col not in df.columns:
                df[col] = float('nan')
        df = df[required_cols]
        df = df.rename(columns={'time': 'Datetime'})
        df.to_csv(target_file, index=False)
        print(f"[OK] Weather data saved to {target_file} ({len(df)} rows)")
        return True
    except Exception as e:
        print(f"[ERR] Error downloading weather data: {e}")
        return False

if __name__ == "__main__":
    pjm_ok = download_pjm_data()
    weather_ok = download_weather_data()

    if pjm_ok and weather_ok:
        print("\n[OK] All data download steps completed successfully!")
    else:
        print("\n[WARN] Some steps failed or require manual action. See logs above.")
        sys.exit(1)
