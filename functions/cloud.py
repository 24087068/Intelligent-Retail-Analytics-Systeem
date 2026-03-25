import logging
import os
import pandas as pd
from PIL import Image
import pickle
import hashlib
import json
from datetime import datetime

def cloud_load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def cloud_validate(df):
    print(f"Shape: {df.shape}")
    print(f"Null: {df.isnull().sum()}")
    print(f"Negatives: {(df['sales'] < 0).sum()}")

def cloud_transform(df):
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df

def cloud_save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def cloud_validate_output(path):
    df = pd.read_parquet(path)
    print(f"Validated shape: {df.shape}")
    print(df.head())
    return df

def cloud_monitor_data(df, stats_path="../data/monitoring/cloud_stats.json"):
    stats = {
        "timestamp": datetime.now().isoformat(),
        "row_count": len(df),
        "sales_mean": float(df["sales"].mean()),
        "sales_std": float(df["sales"].std()),
        "sales_min": int(df["sales"].min()),
        "sales_max": int(df["sales"].max()),
        "date_range": [str(df["date"].min()), str(df["date"].max())],
        "store_count": int(df["store"].nunique()),
        "item_count": int(df["item"].nunique())
    }
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Cloud stats: {stats}")
    return stats

def cloud_pipeline():
    logging.info("Running cloud pipeline")

    df = cloud_load_data("../data/raw/text/train.csv")
    cloud_validate(df)
    cloud_monitor_data(df)
    df_transformed = cloud_transform(df)
    cloud_save_data(df_transformed, "../data/processed/sales.parquet")
    cloud_validate_output("../data/processed/sales.parquet")

    logging.info("Cloud pipeline finished")