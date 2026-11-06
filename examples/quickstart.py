"""Minimal AgoraPlanner quick start.

Run from the repository root:

    python examples/quickstart.py

If ``DEEPSEEK_API_KEY`` is set, the script uses the live model; otherwise it uses
the deterministic offline fallbacks so the API path can be smoke-tested without
credentials. Offline success numbers are not meaningful research results.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.config import LLMConfig  # noqa: E402
from agora_planner.llm import DeepSeekClient, _load_dotenv  # noqa: E402
from agora_planner.nl_demand_first_rsi import NLWorkflow  # noqa: E402
from agora_planner.task_collab_rsi import TaskCollabRSIRunner  # noqa: E402


def main() -> int:
    _load_dotenv()
    live = bool(os.environ.get("DEEPSEEK_API_KEY"))
    if live:
        client = DeepSeekClient(
            LLMConfig(
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
                max_tokens=2500,
                temperature=0.0,
                max_calls=500,
            )
        )
        workflow = NLWorkflow(client, use_llm=True, json_retries=1)
    else:
        workflow = NLWorkflow(None, use_llm=False)

    settings = {
        "seeds": [0],
        "rounds": 1,
        "steps": 6,
        "modes": ["fixed", "u_fixed", "meta_rsi"],
        "delta": 0.0,
        "meta_delta": 0.0,
        "meta_gate": "strict",
        "gate_signal": "success",
        "model": "deepseek" if live else "offline",
    }
    payload = TaskCollabRSIRunner(settings, workflow).run()
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["rsi_evidence"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
