from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def load_config(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        content = path.read_text(encoding="utf-8")
    else:
        content = files("delivery_risk").joinpath("resources/pipeline.json").read_text(
            encoding="utf-8"
        )
    return json.loads(content)


def load_feature_query(path: Path | None = None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    return files("delivery_risk").joinpath("resources/build_features.sql").read_text(
        encoding="utf-8"
    )
