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
