"""CLI wrapper for the task-collaboration-grounded N4 RSI pilot.

Example (live DeepSeek):
    python experiments/run_task_collab_rsi.py \
        --seeds 0 --rounds 4 --steps 10 \
        --modes fixed,u_fixed,meta_rsi,meta_param,self_validated,no_probe \
        --max-calls 3000
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.task_collab_rsi import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
