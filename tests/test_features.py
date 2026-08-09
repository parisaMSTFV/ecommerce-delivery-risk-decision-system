from __future__ import annotations

import pandas as pd

from delivery_risk.features import MODEL_FEATURES, temporal_split


def test_model_features_exclude_outcome_timestamps_and_label():
    forbidden = {"actual_delivery_at", "promised_delivery_at", "is_late"}
    assert forbidden.isdisjoint(MODEL_FEATURES)


def test_temporal_split_respects_cutoffs():
    frame = pd.DataFrame(
        {
            "order_created_at": pd.to_datetime(
                ["2025-01-10", "2025-05-01", "2025-09-01", "2025-12-01"]
            ),
            "order_id": ["A", "B", "C", "D"],
        }
    )
    frame = pd.concat(
        [
            frame,
            pd.DataFrame({"order_created_at": pd.to_datetime(["2025-10-01"]), "order_id": ["E"]}),
        ],
        ignore_index=True,
    )
    split = temporal_split(frame, "2025-04-30", "2025-08-31", "2025-10-31")
    assert split["train"]["order_id"].tolist() == ["A"]
    assert split["tuning"]["order_id"].tolist() == ["B"]
    assert split["calibration"]["order_id"].tolist() == ["C", "E"]
    assert split["holdout"]["order_id"].tolist() == ["D"]
