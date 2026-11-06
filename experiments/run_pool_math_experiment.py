"""CLI wrapper for the public-pool vs market ten-step OLS experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.pool_math_experiment import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
