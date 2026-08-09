from __future__ import annotations

import math

import numpy as np
import pandas as pd

from delivery_risk.policy import add_decision_policy, weighted_capture


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [f"O-{index}" for index in range(20)],
            "order_value": np.linspace(10, 300, 20),
            "is_premium_customer": [0, 1] * 10,
            "is_fragile": [0, 0, 1, 0] * 5,
            "is_late": [1, 0, 0, 1, 0] * 4,
        }
    )


def test_policy_enforces_review_capacity():
    frame = _fixture()
    risk = np.linspace(0.05, 0.95, len(frame))
    scores = add_decision_policy(frame, risk, risk[::-1], 0.10, 0.25)
    assert (scores["decision_tier"] == "priority_review").sum() == math.ceil(len(frame) * 0.10)
    assert scores["priority_rank"].is_unique


def test_weighted_capture_stays_in_unit_interval():
    frame = _fixture()
    risk = np.linspace(0.05, 0.95, len(frame))
    scores = add_decision_policy(frame, risk, risk[::-1], 0.10, 0.25)
    capture = weighted_capture(scores, "priority_score", 0.10)
    assert 0 <= capture <= 1
