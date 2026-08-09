from __future__ import annotations

import hashlib

import pandas as pd

from delivery_risk.synthetic import generate_synthetic_data


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generator_is_deterministic(tmp_path):
    config = {
        "random_seed": 7,
        "n_orders": 300,
        "start_date": "2024-01-01",
        "end_date": "2024-02-15",
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_synthetic_data(config, first)
    generate_synthetic_data(config, second)
    assert _digest(first / "synthetic_orders.csv") == _digest(second / "synthetic_orders.csv")
    assert _digest(first / "synthetic_network_context.csv") == _digest(
        second / "synthetic_network_context.csv"
    )


def test_generated_orders_have_valid_identifiers_and_timestamps(tmp_path):
    config = {
        "random_seed": 11,
        "n_orders": 250,
        "start_date": "2024-01-01",
        "end_date": "2024-03-01",
    }
    orders, _ = generate_synthetic_data(config, tmp_path)
    assert orders["order_id"].is_unique
    created = pd.to_datetime(orders["order_created_at"])
    promised = pd.to_datetime(orders["promised_delivery_at"])
    actual = pd.to_datetime(orders["actual_delivery_at"])
    assert (promised > created).all()
    assert (actual > created).all()
