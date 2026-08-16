# Executed decision report

## Management decision

**Shadow-test model queue.** The capture interval includes zero and average precision is below baseline. The transparent rule remains the fallback until a later operational sample resolves the disagreement between the full-ranking and fixed-capacity evidence.

## Holdout result

The untouched holdout contains **1,313 synthetic orders** with a late-delivery rate of **24.6%**. The calibrated model reached average precision **0.389**, compared with **0.417** for the transparent rule baseline.

At a fixed **10% review capacity**, the model-ranked queue captured **19.5%** of late orders and **21.4%** of weighted late-order harm. The baseline captured **17.1%** of weighted harm, so the model ranking produced **1.26x** the baseline capture on this synthetic holdout. The paired day-block bootstrap estimates the model-minus-baseline capture difference at **+4.4%**, with a 95% interval from **-0.4% to +8.1%**. **96.1%** of bootstrap replicates were above zero; this descriptive share is not a posterior probability or proof of business impact.

## Fixed decision rule

The versioned `fixed-capacity-rule-v1` rule adopts the model queue only when the paired capture interval excludes zero and model average precision is not below baseline. A positive point advantage with an interval crossing zero or lower average precision leads to Shadow-test. A non-positive point advantage or optimistic bound leads to Keep rule baseline.

## Decision boundary

The output is a triage queue, not an automated operational action. `priority_review` means an order should be reviewed within the stated capacity. It does not prove that expediting, rerouting, or contacting the customer will prevent a delay.

## Calibration and use

The holdout Brier score is **0.172** and expected calibration error is **0.028**. Probability quality must be rechecked after any data or network change. Capacity and impact weights are explicit policy choices, not learned causal effects. The bootstrap resamples observed synthetic order dates and does not establish stability in another season or network.
