from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

TARGET = "is_late"
ID_COLUMNS = ["order_id", "order_created_at", "promised_delivery_at", "actual_delivery_at"]
CATEGORICAL_FEATURES = ["warehouse_id", "carrier_id", "destination_zone", "payment_type"]
NUMERIC_FEATURES = [
    "item_count",
    "order_value",
    "weight_kg",
    "volume_dm3",
    "is_fragile",
    "is_premium_customer",
    "promised_lead_days",
    "warehouse_backlog_index",
    "available_capacity_ratio",
    "carrier_ontime_30d",
    "weather_severity",
    "is_holiday_period",
    "is_weekend_order",
    "order_hour",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_feature_table(data_dir: Path, query: str, output_path: Path) -> pd.DataFrame:
    orders = pd.read_csv(data_dir / "synthetic_orders.csv")
    context = pd.read_csv(data_dir / "synthetic_network_context.csv")
    with sqlite3.connect(":memory:") as connection:
        orders.to_sql("orders", connection, index=False)
        context.to_sql("network_context", connection, index=False)
        features = pd.read_sql_query(query, connection)
    features["order_created_at"] = pd.to_datetime(features["order_created_at"])
    features = features.sort_values(["order_created_at", "order_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    return features


def temporal_split(
    features: pd.DataFrame, train_end: str, tuning_end: str, calibration_end: str
) -> dict[str, pd.DataFrame]:
    timestamp = pd.to_datetime(features["order_created_at"])
    one_day = pd.Timedelta(1, unit="D")
    train_cutoff = pd.Timestamp(train_end) + one_day
    tuning_cutoff = pd.Timestamp(tuning_end) + one_day
    calibration_cutoff = pd.Timestamp(calibration_end) + one_day
    splits = {
        "train": features.loc[timestamp < train_cutoff].copy(),
        "tuning": features.loc[(timestamp >= train_cutoff) & (timestamp < tuning_cutoff)].copy(),
        "calibration": features.loc[
            (timestamp >= tuning_cutoff) & (timestamp < calibration_cutoff)
        ].copy(),
        "holdout": features.loc[timestamp >= calibration_cutoff].copy(),
    }
    if any(frame.empty for frame in splits.values()):
        raise ValueError("Temporal split produced an empty partition")
    return splits
