from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_risk.pipeline import run_pipeline  # noqa: E402

if __name__ == "__main__":
    result = run_pipeline(ROOT)
    print(
        "Pipeline complete: "
        f"AP={result['average_precision']:.3f}, "
        f"weighted capture@{result['review_capacity']:.0%}="
        f"{result['weighted_harm_capture_at_capacity']:.1%}"
    )
