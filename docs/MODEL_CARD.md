# Model card

## Intended use

Prioritize fictional ecommerce orders for analyst review after carrier and warehouse assignment and before dispatch. The implementation demonstrates temporal validation, probability calibration, and decision-policy separation.

## Out-of-scope use

- automatic cancellation, rerouting, customer contact, or compensation;
- employee, seller, warehouse, or carrier performance management;
- financial or service-impact claims;
- deployment on real data without rebuilding the data contract and validation.

## Model

- estimator: gradient-boosting binary classifier;
- categorical handling: one-hot encoding with unknown-category support;
- calibration: logistic mapping fitted on a separate later period;
- output: calibrated late-delivery probability.

## Final synthetic evaluation

| Metric | Result |
|---|---:|
| Average precision | 0.389 |
| ROC AUC | 0.680 |
| Brier score | 0.172 |
| Expected calibration error | 0.028 |
| Precision at 10% review capacity | 47.7% |
| Weighted harm capture at 10% capacity | 21.4% |

The baseline achieved higher holdout average precision (0.417) but lower precision and weighted capture at the fixed decision capacity. This trade-off must remain visible in any model review.

## Monitoring requirements

A production version would monitor late-rate shift, feature missingness, calibration, average precision, capacity-level precision and capture, category drift, and queue stability. A simple baseline should remain available as a fallback.

## Ethical and operational considerations

Premium status and order value influence the explicit impact policy, not the probability model's definition of lateness. A real implementation would require review of whether such prioritization is acceptable and whether protected or proxy attributes create unfair service outcomes.

