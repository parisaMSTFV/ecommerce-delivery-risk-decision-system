# Synthetic data

Both CSV files in this directory are generated locally by `scripts/run_pipeline.py` with NumPy seed 42.

- `synthetic_orders.csv` contains fictional order characteristics, promise timestamps, and delivery outcomes.
- `synthetic_network_context.csv` contains fictional daily warehouse and carrier conditions known at the decision timestamp.

The data does not reproduce a real company's distribution, network, thresholds, customers, orders, warehouses, carriers, or performance. Identifiers beginning with `SYN-`, `WH-`, and `CR-` are invented for this project.

