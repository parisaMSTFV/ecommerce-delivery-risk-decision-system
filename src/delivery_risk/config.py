from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or project_root() / "configs" / "pipeline.json"
    return json.loads(config_path.read_text(encoding="utf-8"))
