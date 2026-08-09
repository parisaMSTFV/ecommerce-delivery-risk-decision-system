SELECT
    o.order_id,
    o.order_created_at,
    o.promised_delivery_at,
    o.actual_delivery_at,
    o.warehouse_id,
    o.carrier_id,
    o.destination_zone,
    o.payment_type,
    o.item_count,
    o.order_value,
    o.weight_kg,
    o.volume_dm3,
    o.is_fragile,
    o.is_premium_customer,
    o.promised_lead_days,
    n.warehouse_backlog_index,
    n.available_capacity_ratio,
    n.carrier_ontime_30d,
    n.weather_severity,
    n.is_holiday_period,
    CAST(strftime('%w', o.order_created_at) IN ('0', '6') AS INTEGER) AS is_weekend_order,
    CAST(strftime('%H', o.order_created_at) AS INTEGER) AS order_hour,
    CAST(julianday(o.actual_delivery_at) > julianday(o.promised_delivery_at) AS INTEGER) AS is_late
FROM orders AS o
INNER JOIN network_context AS n
    ON date(o.order_created_at) = n.snapshot_date
   AND o.warehouse_id = n.warehouse_id
   AND o.carrier_id = n.carrier_id;

