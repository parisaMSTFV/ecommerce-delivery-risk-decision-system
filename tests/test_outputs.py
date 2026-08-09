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


def test_priority_queue_contains_only_priority_review_orders():
    queue = pd.read_csv(ROOT / "artifacts" / "priority_review_queue.csv")
    assert set(queue["decision_tier"]) == {"priority_review"}
    assert queue["priority_rank"].max() == len(queue)


def test_risk_decile_ten_is_highest_risk():
    deciles = pd.read_csv(ROOT / "artifacts" / "risk_deciles.csv").set_index("risk_decile")
    assert deciles.loc[10, "predicted_risk"] > deciles.loc[1, "predicted_risk"]
