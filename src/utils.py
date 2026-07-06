"""
src/utils.py
============
Data loading and cleaning utilities for PJME hourly energy consumption.

Functions
---------
load_and_clean(path)
    - Parse Datetime, sort ascending, set as index
    - Deduplicate timestamps (keep mean)
    - Reindex to full hourly range and interpolate small gaps
    - Flag and clip z-score > 5 outliers (does NOT delete rows)
    - Returns cleaned DataFrame with column 'PJME_MW'
"""
import pandas as pd
import numpy as np

RAW_COL = "PJME_MW"


def load_and_clean(path: str = "data/PJME_hourly.csv") -> pd.DataFrame:
    # 1. Load
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    df = df.set_index("Datetime")
    df = df[[RAW_COL]]

    # 2. Deduplicate timestamps – keep mean of duplicates
    n_dupes = df.index.duplicated().sum()
    if n_dupes:
        print(f"[utils] Removing {n_dupes} duplicate timestamps (keeping mean).")
        df = df.groupby(df.index).mean()

    # 3. Reindex to full hourly range – interpolate gaps (linear, limit=6 hrs)
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    n_gaps = len(full_idx) - len(df)
    if n_gaps > 0:
        print(f"[utils] Filling {n_gaps} missing hourly gaps via linear interpolation.")
    df = df.reindex(full_idx)
    df = df.interpolate(method="time", limit=6)
    df.index.name = "Datetime"

    # 4. Outlier clipping: z-score > 5 -> clip to mean ± 5*std (flag column added)
    mu, sigma = df[RAW_COL].mean(), df[RAW_COL].std()
    z = (df[RAW_COL] - mu) / sigma
    n_outliers = (z.abs() > 5).sum()
    if n_outliers:
        print(f"[utils] Clipping {n_outliers} outlier rows (|z|>5).")
    df["outlier_flag"] = (z.abs() > 5).astype(int)
    df[RAW_COL] = df[RAW_COL].clip(lower=mu - 5 * sigma, upper=mu + 5 * sigma)

    # Drop any remaining NaN rows (gaps > 6 hrs)
    df = df.dropna(subset=[RAW_COL])

    print(f"[utils] Loaded {len(df):,} rows | {df.index.min()} -> {df.index.max()}")
    return df
