from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from delivery_risk.evaluation import (
    fixed_capacity_policy_decision,
    paired_day_bootstrap_capture,
)
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


def test_weighted_capture_uses_order_id_for_score_ties():
    frame = pd.DataFrame(
        {
            "order_id": ["O-3", "O-1", "O-2", "O-4"],
            "is_late": [1, 0, 1, 0],
            "impact_weight": [1.0, 1.0, 1.0, 1.0],
            "baseline_priority_score": [0.5, 0.5, 0.5, 0.5],
        }
    )

    first = weighted_capture(frame, "baseline_priority_score", 0.25)
    second = weighted_capture(
        frame.sample(frac=1, random_state=7),
        "baseline_priority_score",
        0.25,
    )

    assert first == second == 0.0


def test_fixed_capacity_rule_preserves_baseline_when_evidence_conflicts():
    decision, reason = fixed_capacity_policy_decision(
        capture_difference=0.04,
        capture_ci_lower=-0.01,
        capture_ci_upper=0.08,
        model_average_precision=0.39,
        baseline_average_precision=0.42,
    )
    assert decision == "Shadow-test model queue"
    assert "interval includes zero" in reason
    assert "average precision is below baseline" in reason

    assert fixed_capacity_policy_decision(0.04, 0.01, 0.08, 0.43, 0.42)[0] == (
        "Adopt model queue"
    )
    assert fixed_capacity_policy_decision(-0.01, -0.04, 0.02, 0.43, 0.42)[0] == (
        "Keep rule baseline"
    )


def test_paired_day_bootstrap_is_deterministic_and_paired():
    rows = []
    for day in range(10):
        for within_day in range(4):
            rows.append(
                {
                    "order_id": f"O-{day:02d}-{within_day}",
                    "order_created_at": pd.Timestamp("2025-11-01")
                    + pd.to_timedelta(day, unit="D"),
                    "is_late": int(within_day == 0),
                    "impact_weight": 1.0,
                    "priority_score": 1.0 if within_day == 0 else 0.1,
                    "baseline_priority_score": 1.0 if within_day == 1 else 0.1,
                }
            )
    scores = pd.DataFrame(rows)
    first_summary, first_distribution = paired_day_bootstrap_capture(
        scores,
        review_capacity=0.25,
        replicates=200,
        confidence_level=0.95,
        seed=19,
    )
    second_summary, second_distribution = paired_day_bootstrap_capture(
        scores,
        review_capacity=0.25,
        replicates=200,
        confidence_level=0.95,
        seed=19,
    )

    assert first_summary == second_summary
    assert_frame_equal(first_distribution, second_distribution)
    assert first_summary["weighted_harm_capture_difference_ci_lower"] > 0
