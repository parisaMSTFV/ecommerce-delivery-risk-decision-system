from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_metrics_match_holdout_artifact():
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(ROOT / "artifacts" / "holdout_scores.csv")
    assert metrics["holdout_orders"] == len(scores)
    assert metrics["holdout_late_orders"] == int(scores["is_late"].sum())
    assert abs(metrics["holdout_late_rate"] - scores["is_late"].mean()) < 1e-12
    assert abs(
        metrics["weighted_harm_capture_difference"]
        - (
            metrics["weighted_harm_capture_at_capacity"]
            - metrics["baseline_weighted_harm_capture_at_capacity"]
        )
    ) < 1e-12
    assert metrics["weighted_harm_capture_difference_ci_lower"] < (
        metrics["weighted_harm_capture_difference"]
    )
    assert metrics["weighted_harm_capture_difference_ci_upper"] > (
        metrics["weighted_harm_capture_difference"]
    )
    assert (ROOT / "reports" / "paired_policy_bootstrap.csv").exists()
    assert (ROOT / "reports" / "figures" / "paired_capture_difference.png").exists()


def test_priority_queue_contains_only_priority_review_orders():
    queue = pd.read_csv(ROOT / "artifacts" / "priority_review_queue.csv")
    assert set(queue["decision_tier"]) == {"priority_review"}
    assert queue["priority_rank"].max() == len(queue)


def test_risk_decile_ten_is_highest_risk():
    deciles = pd.read_csv(ROOT / "artifacts" / "risk_deciles.csv").set_index("risk_decile")
    assert deciles.loc[10, "predicted_risk"] > deciles.loc[1, "predicted_risk"]
