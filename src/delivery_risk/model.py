from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET


@dataclass
class ModelBundle:
    risk_model: Pipeline
    calibrator: LogisticRegression

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.risk_model.predict_proba(frame[MODEL_FEATURES])[:, 1]
        logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
        return self.calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]


def build_risk_model(config: dict) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model_config = config["model"]
    classifier = GradientBoostingClassifier(
        n_estimators=model_config["n_estimators"],
        learning_rate=model_config["learning_rate"],
        max_depth=model_config["max_depth"],
        min_samples_leaf=model_config["min_samples_leaf"],
        subsample=model_config["subsample"],
        random_state=config["random_seed"],
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def fit_model(train: pd.DataFrame, calibration: pd.DataFrame, config: dict) -> ModelBundle:
    pipeline = build_risk_model(config)
    pipeline.fit(train[MODEL_FEATURES], train[TARGET])

    raw_calibration = pipeline.predict_proba(calibration[MODEL_FEATURES])[:, 1]
    calibration_logit = np.log(
        np.clip(raw_calibration, 1e-6, 1 - 1e-6) / np.clip(1 - raw_calibration, 1e-6, 1)
    )
    calibrator = LogisticRegression(C=1000, solver="lbfgs", random_state=config["random_seed"])
    calibrator.fit(calibration_logit.reshape(-1, 1), calibration[TARGET])
    return ModelBundle(risk_model=pipeline, calibrator=calibrator)


def baseline_risk_score(frame: pd.DataFrame) -> np.ndarray:
    """A transparent operational rule used as a ranking baseline."""
    zone = frame["destination_zone"].map({"metro": 0.0, "regional": 0.5, "remote": 1.0})
    backlog = np.clip((frame["warehouse_backlog_index"] - 0.5) / 3.5, 0, 1)
    carrier = np.clip((0.97 - frame["carrier_ontime_30d"]) / 0.22, 0, 1)
    weather = np.clip(frame["weather_severity"] / 3.0, 0, 1)
    capacity = np.clip((1.2 - frame["available_capacity_ratio"]) / 0.55, 0, 1)
    score = 0.30 * backlog + 0.24 * carrier + 0.18 * zone + 0.14 * weather + 0.14 * capacity
    return np.asarray(score)
