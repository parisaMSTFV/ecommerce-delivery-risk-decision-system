# Executed decision report

## Holdout result

The untouched holdout contains **1,313 synthetic orders** with a late-delivery rate of **24.6%**. The calibrated model reached average precision **0.389**, compared with **0.417** for the transparent rule baseline.

At a fixed **10% review capacity**, the model-ranked queue captured **19.5%** of late orders and **21.4%** of weighted late-order harm. The baseline captured **17.1%** of weighted harm, so the model ranking produced **1.26x** the baseline capture on this synthetic holdout.

## Decision boundary

The output is a triage queue, not an automated operational action. `priority_review` means an order should be reviewed within the stated capacity. It does not prove that expediting, rerouting, or contacting the customer will prevent a delay.

## Calibration and use

The holdout Brier score is **0.172** and expected calibration error is **0.028**. Probability quality must be rechecked after any data or network change. Capacity and impact weights are explicit policy choices, not learned causal effects.
