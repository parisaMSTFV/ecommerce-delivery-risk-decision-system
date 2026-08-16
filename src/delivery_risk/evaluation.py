from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .policy import weighted_capture


def fixed_capacity_policy_decision(
    capture_difference: float,
    capture_ci_lower: float,
    capture_ci_upper: float,
    model_average_precision: float,
    baseline_average_precision: float,
) -> tuple[str, str]:
    """Apply the fixed managerial rule to model and baseline evidence."""

    if capture_difference <= 0 or capture_ci_upper <= 0:
        return (
            "Keep rule baseline",
            "The model has no positive fixed-capacity capture advantage.",
        )
    if capture_ci_lower > 0 and model_average_precision >= baseline_average_precision:
        return (
            "Adopt model queue",
            "The capture advantage excludes zero and average precision is not worse.",
        )
    interval_includes_zero = capture_ci_lower <= 0
    average_precision_is_worse = model_average_precision < baseline_average_precision
    if interval_includes_zero and average_precision_is_worse:
        return (
            "Shadow-test model queue",
            "The capture interval includes zero and average precision is below baseline.",
        )
    if interval_includes_zero:
        return (
            "Shadow-test model queue",
            "The point capture advantage is positive, but its interval includes zero.",
        )
    return (
        "Shadow-test model queue",
        "The capture advantage excludes zero, but average precision is below baseline.",
    )


def paired_day_bootstrap_capture(
    scores: pd.DataFrame,
    review_capacity: float,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Bootstrap paired policy differences by resampling holdout order dates."""

    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between zero and one.")

    frame = scores.copy()
    frame["_bootstrap_date"] = pd.to_datetime(frame["order_created_at"]).dt.date
    day_groups = [
        group.index.to_numpy()
        for _, group in frame.groupby("_bootstrap_date", sort=True)
    ]
    if len(day_groups) < 2:
        raise ValueError("Day-block bootstrap requires at least two order dates.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for replicate in range(replicates):
        sampled_days = rng.integers(0, len(day_groups), size=len(day_groups))
        sampled_index = np.concatenate([day_groups[index] for index in sampled_days])
        sample = frame.loc[sampled_index]
        model_capture = weighted_capture(sample, "priority_score", review_capacity)
        baseline_capture = weighted_capture(
            sample,
            "baseline_priority_score",
            review_capacity,
        )
        rows.append(
            {
                "replicate": replicate + 1,
                "model_weighted_harm_capture": model_capture,
                "baseline_weighted_harm_capture": baseline_capture,
                "capture_difference": model_capture - baseline_capture,
            }
        )

    distribution = pd.DataFrame(rows)
    alpha = 1 - confidence_level
    lower, upper = distribution["capture_difference"].quantile(
        [alpha / 2, 1 - alpha / 2]
    )
    observed_model = weighted_capture(scores, "priority_score", review_capacity)
    observed_baseline = weighted_capture(
        scores,
        "baseline_priority_score",
        review_capacity,
    )
    summary: dict[str, float | int | str] = {
        "bootstrap_unit": "order_date",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bootstrap_confidence_level": confidence_level,
        "weighted_harm_capture_difference": observed_model - observed_baseline,
        "weighted_harm_capture_difference_ci_lower": float(lower),
        "weighted_harm_capture_difference_ci_upper": float(upper),
        "bootstrap_share_difference_above_zero": float(
            (distribution["capture_difference"] > 0).mean()
        ),
    }
    return summary, distribution


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(y_true)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        upper_mask = probability < upper if index < bins - 1 else probability <= upper
        mask = (probability >= lower) & upper_mask
        if mask.any():
            value += mask.mean() * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return float(value if total else 0.0)


def evaluate_scores(scores: pd.DataFrame, review_capacity: float) -> dict:
    y_true = scores["is_late"].to_numpy()
    probability = scores["predicted_late_risk"].to_numpy()
    baseline = scores["baseline_risk_score"].to_numpy()

    review_count = math.ceil(len(scores) * review_capacity)
    selected = scores.nlargest(review_count, "priority_score")
    y_pred = scores["order_id"].isin(selected["order_id"]).astype(int).to_numpy()
    baseline_selected = scores.nlargest(review_count, "baseline_priority_score")
    baseline_pred = scores["order_id"].isin(baseline_selected["order_id"]).astype(int).to_numpy()

    total_late = int(y_true.sum())
    model_late_selected = int(selected["is_late"].sum())
    baseline_late_selected = int(baseline_selected["is_late"].sum())
    model_harm = weighted_capture(scores, "priority_score", review_capacity)
    baseline_harm = weighted_capture(scores, "baseline_priority_score", review_capacity)

    return {
        "holdout_orders": int(len(scores)),
        "holdout_late_orders": total_late,
        "holdout_late_rate": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, probability)),
        "baseline_average_precision": float(average_precision_score(y_true, baseline)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "baseline_roc_auc": float(roc_auc_score(y_true, baseline)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability)),
        "expected_calibration_error": expected_calibration_error(y_true, probability),
        "review_capacity": float(review_capacity),
        "review_queue_orders": review_count,
        "precision_at_capacity": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_capacity": float(recall_score(y_true, y_pred, zero_division=0)),
        "baseline_precision_at_capacity": float(
            precision_score(y_true, baseline_pred, zero_division=0)
        ),
        "baseline_recall_at_capacity": float(recall_score(y_true, baseline_pred, zero_division=0)),
        "late_order_capture_at_capacity": float(model_late_selected / total_late),
        "baseline_late_order_capture_at_capacity": float(baseline_late_selected / total_late),
        "weighted_harm_capture_at_capacity": model_harm,
        "baseline_weighted_harm_capture_at_capacity": baseline_harm,
        "weighted_capture_lift": float(model_harm / baseline_harm) if baseline_harm else None,
    }


def risk_deciles(scores: pd.DataFrame) -> pd.DataFrame:
    result = scores.copy()
    result["risk_decile"] = pd.qcut(
        result["predicted_late_risk"].rank(method="first"), 10, labels=range(1, 11)
    ).astype(int)
    return (
        result.groupby("risk_decile", as_index=False)
        .agg(
            orders=("order_id", "size"),
            predicted_risk=("predicted_late_risk", "mean"),
            actual_late_rate=("is_late", "mean"),
        )
        .sort_values("risk_decile")
    )
