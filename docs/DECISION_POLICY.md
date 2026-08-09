# Decision policy

## Policy objective

Use a limited review capacity on orders combining higher late-delivery risk with higher stated customer-impact weight.

## Score

`priority_score = predicted_late_risk × impact_weight`

The impact weight is:

`1 + 0.45 × premium + 0.35 × normalized_log_order_value + 0.20 × fragile`

The value component is capped by the synthetic generator's maximum order value. The formula prevents a high-value order with negligible predicted risk from automatically dominating the queue.

## Tiers

| Tier | Share | Meaning |
|---|---:|---|
| `priority_review` | Top 10% | Enter the constrained manual triage queue |
| `monitor` | Next 15% | Retain for monitoring if capacity becomes available |
| `standard_flow` | Remaining 75% | Follow the normal process |

Ties are resolved deterministically using synthetic order ID.

## Guardrails

- The system recommends review, not an intervention.
- No cost saving or prevented-delay claim is calculated.
- Capacity and weights must be approved by the operating team.
- If calibration or capacity-level capture deteriorates, fall back to the documented rule.
- Treatment effectiveness must be measured separately.

