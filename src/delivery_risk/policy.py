from __future__ import annotations

import math

import numpy as np
import pandas as pd


def add_decision_policy(
    holdout: pd.DataFrame,
    predicted_risk: np.ndarray,
    baseline_risk: np.ndarray,
    review_capacity: float,
    monitor_capacity: float,
) -> pd.DataFrame:
    if not 0 < review_capacity < monitor_capacity <= 1:
        raise ValueError("Capacity thresholds must satisfy 0 < review < monitor <= 1")

    scores = holdout.copy()
    value_component = np.clip(np.log1p(scores["order_value"]) / np.log(1401), 0, 1)
    scores["impact_weight"] = (
        1.0
        + 0.45 * scores["is_premium_customer"]
        + 0.35 * value_component
        + 0.20 * scores["is_fragile"]
    ).round(6)
    scores["predicted_late_risk"] = predicted_risk
    scores["baseline_risk_score"] = baseline_risk
    scores["priority_score"] = scores["predicted_late_risk"] * scores["impact_weight"]
    scores["baseline_priority_score"] = scores["baseline_risk_score"] * scores["impact_weight"]
    scores = scores.sort_values(
        ["priority_score", "order_id"], ascending=[False, True]
    ).reset_index(drop=True)
    scores["priority_rank"] = np.arange(1, len(scores) + 1)

    review_count = math.ceil(len(scores) * review_capacity)
    monitor_count = math.ceil(len(scores) * monitor_capacity)
    scores["decision_tier"] = "standard_flow"
    scores.loc[: monitor_count - 1, "decision_tier"] = "monitor"
    scores.loc[: review_count - 1, "decision_tier"] = "priority_review"
    return scores


def weighted_capture(scores: pd.DataFrame, ranking_column: str, capacity: float) -> float:
    selected_count = math.ceil(len(scores) * capacity)
    ranked = scores.sort_values(ranking_column, ascending=False)
    harm = scores["is_late"] * scores["impact_weight"]
    denominator = float(harm.sum())
    if denominator == 0:
        return 0.0
    selected = ranked.head(selected_count)
    selected_harm = selected["is_late"] * selected["impact_weight"]
    return float(selected_harm.sum() / denominator)
