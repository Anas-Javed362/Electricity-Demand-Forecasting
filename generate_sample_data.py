"""
Generate realistic synthetic PJM + weather data for pipeline testing.
Produces:
  data/PJME_hourly.csv  – hourly energy load (MW), 2002-2018
  data/weather_hourly.csv – hourly weather, 2002-2018

The load model is:
  base_load + daily_cycle + weekly_cycle + seasonal_cycle
  + temp_effect (heating/cooling) + holiday_dip + noise

Run: python generate_sample_data.py
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Date range ───────────────────────────────────────────────────────────────
dates = pd.date_range(start="2002-01-01", end="2018-06-30 23:00", freq="h")
n = len(dates)
print(f"Generating {n:,} hourly records ({dates[0]} to {dates[-1]})")

# ── Weather (Philadelphia-like) ──────────────────────────────────────────────
# Temperature: seasonal sinusoid + daily variation + noise
day_of_year = dates.dayofyear
hour = dates.hour
# Annual seasonal cycle: cold winter (~2C), hot summer (~30C)
temp_seasonal = 16 + 14 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
# Diurnal variation: coolest at 5am, warmest at 3pm
temp_diurnal = 4 * np.sin(2 * np.pi * (hour - 5) / 24)
# Year-over-year trend (slight warming)
year_frac = (dates.year - 2002) / 16
temp_trend = 0.5 * year_frac
# Noise
temp_noise = np.random.normal(0, 2, n)
temp = temp_seasonal + temp_diurnal + temp_trend + temp_noise

# Humidity: inversely correlated with temp + noise
rhum = np.clip(75 - 0.8 * temp + np.random.normal(0, 8, n), 20, 100)

# Wind speed: random with slight seasonal pattern
wspd = np.clip(
    10 + 4 * np.sin(2 * np.pi * (day_of_year - 300) / 365) + np.random.exponential(3, n),
    0, 60
)

weather_df = pd.DataFrame({
    "Datetime": dates,
    "temp": np.round(temp, 1),
    "rhum": np.round(rhum, 1),
    "wspd": np.round(wspd, 1),
})
weather_df.to_csv(os.path.join(DATA_DIR, "weather_hourly.csv"), index=False)
print(f"  Saved weather_hourly.csv  ({len(weather_df):,} rows)")

# ── Energy load (MW) ─────────────────────────────────────────────────────────
# Base load
base = 31_000

# Daily cycle: low at 4am (~0.80x), peak at 6pm (~1.15x)
daily_amp = 4_000
daily_cycle = daily_amp * np.sin(2 * np.pi * (hour - 4) / 24)

# Weekly cycle: weekdays higher than weekends
dow = dates.dayofweek  # 0=Mon, 6=Sun
weekly_adj = np.where(dow >= 5, -2_500, 1_000)  # weekend dip

# Seasonal cycle: summer & winter peaks, spring/fall trough
seasonal_amp = 5_000
seasonal = seasonal_amp * (
    0.5 * np.cos(2 * np.pi * (day_of_year - 180) / 365)  # summer peak
    + 0.3 * np.cos(4 * np.pi * day_of_year / 365)        # secondary winter peak
)

# Temperature effect: heating below 10C, cooling above 18C
heating = np.where(temp < 10, (10 - temp) * 250, 0)
cooling = np.where(temp > 18, (temp - 18) * 300, 0)
temp_effect = heating + cooling

# Holiday dip (Christmas, New Year, etc.)
import holidays as hd
us_hols = hd.US()
is_holiday = np.array([1 if d.date() in us_hols else 0 for d in dates])
holiday_adj = -3_000 * is_holiday

# Noise
noise = np.random.normal(0, 600, n)

load_mw = base + daily_cycle + weekly_adj + seasonal + temp_effect + holiday_adj + noise
load_mw = np.clip(load_mw, 12_000, 62_000)

energy_df = pd.DataFrame({
    "Datetime": dates,
    "PJME_MW": np.round(load_mw, 1),
})
energy_df.to_csv(os.path.join(DATA_DIR, "PJME_hourly.csv"), index=False)
print(f"  Saved PJME_hourly.csv     ({len(energy_df):,} rows)")
print(f"  Load stats: min={load_mw.min():,.0f} MW | mean={load_mw.mean():,.0f} MW | max={load_mw.max():,.0f} MW")
print("Done. Run: python src/feature_engineering.py")
