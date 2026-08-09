# Methodology

## Problem formulation

The unit of analysis is an order after warehouse and carrier assignment but before dispatch. The binary target equals one when `actual_delivery_at` is later than `promised_delivery_at`.

The model estimates late-delivery probability. The downstream decision ranks orders for manual review under a fixed capacity. Prediction and intervention are intentionally separate: an order can be high risk without having an effective or economical treatment.

## Synthetic generator

The fixed-seed generator creates orders from 2024-01-01 through 2025-12-31. Daily warehouse-carrier context contains backlog, capacity, on-time rate, weather, and holiday fields. Late outcomes are sampled from a nonlinear probability that responds to these fictional conditions plus order attributes.

The peak period changes both prevalence and context. This produces a meaningful time-shift test rather than an identical random split.

## Point-in-time feature construction

SQLite executes `sql/build_features.sql`. Each order joins to the context row matching its order date, assigned warehouse, and assigned carrier. Model features exclude:

- `actual_delivery_at`;
- `promised_delivery_at`;
- `is_late`.

The promise duration itself is allowed because it is known at the decision timestamp.

## Temporal development protocol

1. Fit candidate design on 9,866 orders through 2025-04-30.
2. Compare ranking quality with the rule baseline on 2,572 orders from May through August.
3. Refit the fixed model on train plus tuning data.
4. Fit Platt-style logistic calibration on 1,249 orders from September and October.
5. Evaluate once on 1,313 November-December orders.

No random cross-validation mixes later orders into earlier training periods.

## Model and baseline

The model is a gradient-boosting classifier with one-hot encoded categorical variables. Hyperparameters are stored in `configs/pipeline.json`.

The baseline is a weighted rule using warehouse backlog, carrier on-time rate, destination zone, weather, and available capacity. It is intentionally credible rather than trivial.

## Evaluation

Prediction metrics include average precision, ROC AUC, Brier score, log loss, and expected calibration error. The operational metrics are evaluated at the configured 10% review capacity:

- precision in the review queue;
- share of late orders captured;
- share of impact-weighted late orders captured.

Average precision measures ranking over all thresholds. Capacity metrics measure the part of the ranking the team can act on. Both are reported because they can disagree.

