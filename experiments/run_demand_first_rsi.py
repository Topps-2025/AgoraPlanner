"""CLI for the demand-first symbolic RSI pilot.

Examples::

    PYTHONPATH=src python experiments/run_demand_first_rsi.py --seeds 0,1,2 \
      --rounds 3 --meta-tasks 6 --test-tasks 6 --candidates 8 \
      --output-dir results/demand_first_rsi_pilot_3seeds

The pilot is symbolic and deterministic: it tests the buyer-first protocol,
scoped seller offers, bundle valuation, demand-policy RSI, and joint demand+
valuation RSI without calling an LLM.  The LLM parsing/rewriting layer is
represented by the typed factor schema and the AST mutation hooks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.demand_first_rsi import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
