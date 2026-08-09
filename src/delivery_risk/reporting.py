from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from .features import MODEL_FEATURES
from .model import ModelBundle
from .policy import weighted_capture

COLORS = {"model": "#1F6F78", "baseline": "#D28C45", "neutral": "#5C6470"}


def feature_importance(
    bundle: ModelBundle, holdout: pd.DataFrame, seed: int, max_rows: int = 3000
) -> pd.DataFrame:
    sample = holdout.sample(min(len(holdout), max_rows), random_state=seed)
    result = permutation_importance(
        bundle.risk_model,
        sample[MODEL_FEATURES],
        sample["is_late"],
        scoring="average_precision",
        n_repeats=4,
        random_state=seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def create_figures(
    scores: pd.DataFrame,
    deciles: pd.DataFrame,
    importance: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(deciles["risk_decile"], deciles["predicted_risk"], marker="o", label="Predicted")
    ax.plot(deciles["risk_decile"], deciles["actual_late_rate"], marker="o", label="Observed")
    ax.set(
        xlabel="Risk decile (10 = highest risk)",
        ylabel="Late-delivery rate",
        title="Predicted vs observed risk by decile",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "risk_by_decile.png", dpi=160)
    plt.close(fig)

    y_true = scores["is_late"].to_numpy()
    probability = scores["predicted_late_risk"].to_numpy()
    observed, predicted = calibration_curve(y_true, probability, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["neutral"], label="Perfect calibration")
    ax.plot(predicted, observed, marker="o", color=COLORS["model"], label="Model")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed late rate",
        title="Holdout calibration",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "calibration.png", dpi=160)
    plt.close(fig)

    capacities = np.arange(0.02, 0.51, 0.02)
    model_capture = [
        weighted_capture(scores, "priority_score", float(value)) for value in capacities
    ]
    baseline_capture = [
        weighted_capture(scores, "baseline_priority_score", float(value)) for value in capacities
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(capacities, model_capture, color=COLORS["model"], linewidth=2.2, label="Risk model")
    ax.plot(
        capacities,
        baseline_capture,
        color=COLORS["baseline"],
        linewidth=2.2,
        label="Rule baseline",
    )
    ax.plot(capacities, capacities, linestyle="--", color=COLORS["neutral"], label="Random ranking")
    ax.set(
        xlabel="Share of orders reviewed",
        ylabel="Weighted late-order harm captured",
        title="Capacity capture curve",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "capacity_capture.png", dpi=160)
    plt.close(fig)

    top = importance.head(10).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color=COLORS["model"],
        alpha=0.9,
    )
    ax.set(
        xlabel="Decrease in average precision after permutation",
        title="Holdout permutation importance",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def write_executive_summary(metrics: dict, output_path: Path) -> None:
    lift = metrics["weighted_capture_lift"]
    text = (
        "# Executed decision report\n\n"
        "## Holdout result\n\n"
        f"The untouched holdout contains **{metrics['holdout_orders']:,} synthetic orders** "
        f"with a late-delivery rate of **{metrics['holdout_late_rate']:.1%}**. The calibrated "
        f"model reached average precision **{metrics['average_precision']:.3f}**, compared with "
        f"**{metrics['baseline_average_precision']:.3f}** for the transparent rule baseline.\n\n"
        f"At a fixed **{metrics['review_capacity']:.0%} review capacity**, the model-ranked queue "
        f"captured **{metrics['late_order_capture_at_capacity']:.1%}** of late orders and "
        f"**{metrics['weighted_harm_capture_at_capacity']:.1%}** of weighted late-order harm. "
        f"The baseline captured **{metrics['baseline_weighted_harm_capture_at_capacity']:.1%}** "
        f"of weighted harm, so the model ranking produced **{lift:.2f}x** the baseline capture "
        "on this synthetic holdout.\n\n"
        "## Decision boundary\n\n"
        "The output is a triage queue, not an automated operational action. "
        "`priority_review` means an order should be reviewed within the stated capacity. "
        "It does not prove that expediting, rerouting, or contacting the customer will "
        "prevent a delay.\n\n"
        "## Calibration and use\n\n"
        f"The holdout Brier score is **{metrics['brier_score']:.3f}** and expected calibration "
        f"error is **{metrics['expected_calibration_error']:.3f}**. Probability quality must be "
        "rechecked after any data or network change. Capacity and impact weights are explicit "
        "policy choices, not learned causal effects.\n"
    )
    output_path.write_text(text, encoding="utf-8")


def write_metrics(metrics: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
