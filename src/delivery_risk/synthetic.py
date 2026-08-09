from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WAREHOUSES = np.array(["WH-A", "WH-B", "WH-C"])
CARRIERS = np.array(["CR-1", "CR-2", "CR-3", "CR-4"])
ZONES = np.array(["metro", "regional", "remote"])


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_synthetic_data(config: dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate fictional orders and point-in-time logistics context."""
    rng = np.random.default_rng(config["random_seed"])
    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])
    dates = pd.date_range(start, end, freq="D")

    context_rows: list[dict] = []
    warehouse_base = {"WH-A": 0.05, "WH-B": 0.18, "WH-C": -0.05}
    carrier_base = {"CR-1": 0.94, "CR-2": 0.91, "CR-3": 0.88, "CR-4": 0.93}
    backlog_state = {warehouse: 0.0 for warehouse in WAREHOUSES}

    for date in dates:
        seasonal = 0.45 * np.sin(2 * np.pi * date.dayofyear / 365.25)
        peak = 0.9 if date.month in (11, 12) else 0.0
        weekend = float(date.dayofweek >= 5)
        holiday = float(
            (date.month == 12 and date.day >= 20) or (date.month == 1 and date.day <= 3)
        )
        weather = float(np.clip(rng.gamma(1.2, 0.35) + 0.35 * (date.month in (1, 2)), 0, 3))
        for warehouse in WAREHOUSES:
            backlog_state[warehouse] = (
                0.65 * backlog_state[warehouse]
                + 0.25 * seasonal
                + 0.40 * peak
                + warehouse_base[warehouse]
                + rng.normal(0, 0.35)
            )
            backlog = float(np.clip(1.5 + backlog_state[warehouse], 0.15, 4.5))
            capacity = float(
                np.clip(
                    1.15 - 0.12 * peak - 0.08 * weekend + rng.normal(0, 0.08),
                    0.65,
                    1.35,
                )
            )
            for carrier in CARRIERS:
                ontime = float(
                    np.clip(
                        carrier_base[carrier]
                        - 0.018 * weather
                        - 0.025 * peak
                        + rng.normal(0, 0.012),
                        0.74,
                        0.98,
                    )
                )
                context_rows.append(
                    {
                        "snapshot_date": date.strftime("%Y-%m-%d"),
                        "warehouse_id": warehouse,
                        "carrier_id": carrier,
                        "warehouse_backlog_index": round(backlog, 5),
                        "available_capacity_ratio": round(capacity, 5),
                        "carrier_ontime_30d": round(ontime, 5),
                        "weather_severity": round(weather, 5),
                        "is_holiday_period": int(holiday),
                    }
                )

    context = pd.DataFrame(context_rows)
    n_orders = int(config["n_orders"])
    day_offsets = rng.integers(0, len(dates), size=n_orders)
    minute_offsets = rng.integers(7 * 60, 22 * 60, size=n_orders)
    created_at = (
        start + pd.to_timedelta(day_offsets, unit="D") + pd.to_timedelta(minute_offsets, unit="m")
    )
    warehouse = rng.choice(WAREHOUSES, size=n_orders, p=[0.48, 0.32, 0.20])
    carrier = rng.choice(CARRIERS, size=n_orders, p=[0.33, 0.27, 0.22, 0.18])
    zone = rng.choice(ZONES, size=n_orders, p=[0.55, 0.32, 0.13])
    item_count = np.clip(rng.poisson(2.1, n_orders) + 1, 1, 12)
    order_value = np.clip(rng.lognormal(4.25, 0.72, n_orders), 8, 1400)
    weight_kg = np.clip(rng.lognormal(0.15, 0.75, n_orders) + 0.18 * item_count, 0.15, 35)
    volume_dm3 = np.clip(weight_kg * rng.uniform(1.2, 4.5, n_orders), 0.4, 110)
    fragile = rng.binomial(1, np.clip(0.08 + 0.015 * item_count, 0, 0.35))
    premium = rng.binomial(1, 0.22, n_orders)
    payment = rng.choice(["prepaid", "cash_on_delivery"], size=n_orders, p=[0.83, 0.17])
    promised_days = (
        1
        + (zone == "regional").astype(int)
        + 2 * (zone == "remote").astype(int)
        + rng.binomial(1, 0.28, n_orders)
    )

    orders = pd.DataFrame(
        {
            "order_id": [f"SYN-O{i:06d}" for i in range(1, n_orders + 1)],
            "order_created_at": created_at,
            "warehouse_id": warehouse,
            "carrier_id": carrier,
            "destination_zone": zone,
            "payment_type": payment,
            "item_count": item_count,
            "order_value": order_value.round(2),
            "weight_kg": weight_kg.round(3),
            "volume_dm3": volume_dm3.round(3),
            "is_fragile": fragile,
            "is_premium_customer": premium,
            "promised_lead_days": promised_days,
        }
    )
    orders["snapshot_date"] = orders["order_created_at"].dt.strftime("%Y-%m-%d")
    joined = orders.merge(context, on=["snapshot_date", "warehouse_id", "carrier_id"], how="left")

    zone_risk = joined["destination_zone"].map({"metro": 0.0, "regional": 0.45, "remote": 1.0})
    carrier_risk = (0.93 - joined["carrier_ontime_30d"]) * 10
    nonlinear_load = np.maximum(joined["warehouse_backlog_index"] - 1.8, 0) ** 1.35
    fragile_load = joined["is_fragile"] * (joined["item_count"] >= 4)
    logit = (
        -2.55
        + 0.65 * zone_risk
        + 0.62 * nonlinear_load
        + 0.85 * carrier_risk
        + 0.25 * joined["weather_severity"]
        + 0.32 * joined["is_holiday_period"]
        + 0.18 * (joined["volume_dm3"] > 18)
        + 0.38 * fragile_load
        + 0.42 * (joined["available_capacity_ratio"] < 0.9)
        - 0.22 * (joined["promised_lead_days"] >= 4)
        + rng.normal(0, 0.18, n_orders)
    )
    late_probability = _sigmoid(logit.to_numpy())
    is_late = rng.binomial(1, late_probability)
    early_hours = rng.integers(3, 20, n_orders)
    late_hours = np.clip(rng.gamma(2.1, 8.0, n_orders).astype(int) + 1, 1, 72)
    promised_at = joined["order_created_at"] + pd.to_timedelta(
        joined["promised_lead_days"], unit="D"
    )
    actual_at = promised_at + pd.to_timedelta(np.where(is_late, late_hours, -early_hours), unit="h")

    orders["promised_delivery_at"] = promised_at.dt.strftime("%Y-%m-%dT%H:%M:%S")
    orders["actual_delivery_at"] = actual_at.dt.strftime("%Y-%m-%dT%H:%M:%S")
    orders["order_created_at"] = orders["order_created_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    orders = orders.drop(columns=["snapshot_date"])
    orders = orders.sort_values(["order_created_at", "order_id"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    orders.to_csv(output_dir / "synthetic_orders.csv", index=False)
    context.to_csv(output_dir / "synthetic_network_context.csv", index=False)
    return orders, context
