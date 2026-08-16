from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from .config import load_config, project_root
from .evaluation import (
    evaluate_scores,
    fixed_capacity_policy_decision,
    paired_day_bootstrap_capture,
    risk_deciles,
)
from .features import MODEL_FEATURES, TARGET, build_feature_table, temporal_split
from .model import baseline_risk_score, build_risk_model, fit_model
from .policy import add_decision_policy
from .reporting import (
    create_figures,
    feature_importance,
    write_executive_summary,
    write_metrics,
)
from .synthetic import generate_synthetic_data


def _fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def run_pipeline(root: Path | None = None) -> dict:
    root = root or project_root()
    config = load_config(root / "configs" / "pipeline.json")
    data_dir = root / "data"
    artifacts_dir = root / "artifacts"
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    for directory in (data_dir, artifacts_dir, reports_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    generate_synthetic_data(config, data_dir)
    features = build_feature_table(
        data_dir,
        root / "sql" / "build_features.sql",
        artifacts_dir / "feature_table.csv",
    )
    splits = temporal_split(
        features,
        config["train_end"],
        config["tuning_end"],
        config["calibration_end"],
    )
    candidate = build_risk_model(config)
    candidate.fit(splits["train"][MODEL_FEATURES], splits["train"][TARGET])
    tuning_probability = candidate.predict_proba(splits["tuning"][MODEL_FEATURES])[:, 1]
    tuning_baseline = baseline_risk_score(splits["tuning"])
    development = pd.concat([splits["train"], splits["tuning"]], ignore_index=True)
    bundle = fit_model(development, splits["calibration"], config)
    predicted_risk = bundle.predict_proba(splits["holdout"])
    baseline = baseline_risk_score(splits["holdout"])
    scores = add_decision_policy(
        splits["holdout"],
        predicted_risk,
        baseline,
        config["review_capacity"],
        config["monitor_capacity"],
    )
    metrics = evaluate_scores(scores, config["review_capacity"])
    bootstrap_summary, bootstrap_distribution = paired_day_bootstrap_capture(
        scores,
        review_capacity=config["review_capacity"],
        replicates=config["policy_bootstrap"]["replicates"],
        confidence_level=config["policy_bootstrap"]["confidence_level"],
        seed=config["policy_bootstrap"]["seed"],
    )
    metrics.update(bootstrap_summary)
    recommendation, recommendation_reason = fixed_capacity_policy_decision(
        float(metrics["weighted_harm_capture_difference"]),
        float(metrics["weighted_harm_capture_difference_ci_lower"]),
        float(metrics["weighted_harm_capture_difference_ci_upper"]),
        float(metrics["average_precision"]),
        float(metrics["baseline_average_precision"]),
    )
    metrics.update(
        {
            "policy_recommendation": recommendation,
            "policy_recommendation_reason": recommendation_reason,
            "policy_decision_rule_version": config[
                "policy_decision_rule_version"
            ],
        }
    )
    metrics.update(
        {
            "train_orders": int(len(splits["train"])),
            "tuning_orders": int(len(splits["tuning"])),
            "calibration_orders": int(len(splits["calibration"])),
            "train_end": config["train_end"],
            "tuning_end": config["tuning_end"],
            "calibration_end": config["calibration_end"],
            "tuning_average_precision": float(
                average_precision_score(splits["tuning"][TARGET], tuning_probability)
            ),
            "tuning_baseline_average_precision": float(
                average_precision_score(splits["tuning"][TARGET], tuning_baseline)
            ),
            "random_seed": config["random_seed"],
        }
    )

    deciles = risk_deciles(scores)
    importance = feature_importance(bundle, splits["holdout"], config["random_seed"])
    public_columns = [
        "order_id",
        "order_created_at",
        "predicted_late_risk",
        "impact_weight",
        "priority_score",
        "priority_rank",
        "decision_tier",
        "is_late",
    ]
    scores[public_columns].to_csv(artifacts_dir / "holdout_scores.csv", index=False)
    scores.loc[scores["decision_tier"] == "priority_review", public_columns].to_csv(
        artifacts_dir / "priority_review_queue.csv", index=False
    )
    deciles.to_csv(artifacts_dir / "risk_deciles.csv", index=False)
    importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    bootstrap_distribution.to_csv(
        reports_dir / "paired_policy_bootstrap.csv",
        index=False,
    )
    metrics["data_fingerprint"] = _fingerprint(
        [data_dir / "synthetic_orders.csv", data_dir / "synthetic_network_context.csv"]
    )
    write_metrics(metrics, reports_dir / "metrics.json")
    write_executive_summary(metrics, reports_dir / "executive_summary.md")
    create_figures(
        scores,
        deciles,
        importance,
        bootstrap_distribution,
        metrics,
        figures_dir,
    )
    return metrics


def main() -> None:
    metrics = run_pipeline()
    print(
        "Pipeline complete: "
        f"AP={metrics['average_precision']:.3f}, "
        f"weighted capture@{metrics['review_capacity']:.0%}="
        f"{metrics['weighted_harm_capture_at_capacity']:.1%}, "
        f"decision={metrics['policy_recommendation']}"
    )


if __name__ == "__main__":
    main()
