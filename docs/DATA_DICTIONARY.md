# Data dictionary

## Synthetic orders

| Field | Meaning | Available at decision time? |
|---|---|---|
| `order_id` | Fictional identifier with `SYN-O` prefix | Yes |
| `order_created_at` | Fictional order timestamp | Yes |
| `warehouse_id` | Assigned fictional warehouse | Yes |
| `carrier_id` | Assigned fictional carrier | Yes |
| `destination_zone` | Metro, regional, or remote class | Yes |
| `payment_type` | Prepaid or cash on delivery | Yes |
| `item_count` | Number of items | Yes |
| `order_value` | Synthetic currency amount | Yes |
| `weight_kg` | Shipment weight | Yes |
| `volume_dm3` | Shipment volume | Yes |
| `is_fragile` | Fictional handling flag | Yes |
| `is_premium_customer` | Fictional service-tier flag | Yes |
| `promised_lead_days` | Promise duration | Yes |
| `promised_delivery_at` | Promise timestamp | Yes, excluded from model |
| `actual_delivery_at` | Observed outcome timestamp | No, label only |

## Synthetic network context

| Field | Meaning |
|---|---|
| `snapshot_date` | Date of the point-in-time context |
| `warehouse_backlog_index` | Fictional relative workload indicator |
| `available_capacity_ratio` | Fictional capacity-to-plan ratio |
| `carrier_ontime_30d` | Fictional trailing on-time rate |
| `weather_severity` | Fictional severity index from 0 to 3 |
| `is_holiday_period` | Fictional peak-calendar indicator |

## Derived fields

| Field | Meaning |
|---|---|
| `is_weekend_order` | Weekend indicator from order timestamp |
| `order_hour` | Hour from order timestamp |
| `is_late` | Actual timestamp later than promise timestamp |
| `predicted_late_risk` | Calibrated probability |
| `impact_weight` | Explicit policy weight |
| `priority_score` | Risk multiplied by impact weight |
| `decision_tier` | Capacity-based triage result |

