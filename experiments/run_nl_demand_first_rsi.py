"""CLI wrapper for the natural-language demand-first RSI pilot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.nl_demand_first_rsi import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
