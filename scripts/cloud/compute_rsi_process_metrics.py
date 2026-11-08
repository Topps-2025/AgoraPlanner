"""Post-process the RSI replay rows into process/mechanism metrics.

The public memory-QA benchmarks are single-agent and expose no ownership or
agent trace, so these metrics are computed on the controlled task-collaboration
world, where every row records the selected memory ids, buyer, success,
pollution, and spend.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("AGORA_CLOUD_ROOT", "/root/autodl-tmp/agora_planner_cli"))
SUMMARY = ROOT / "results" / "rsi_qfocus_multiseed" / "summary.json"
OUT = ROOT / "results" / "rsi_process_metrics.json"


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((n / total) * math.log(n / total + 1e-12) for n in counts.values())


def episode_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: row["step_id"])
    selected = [mid for row in rows for mid in row.get("selected", [])]
    counts: dict[str, int] = defaultdict(int)
    for mid in selected:
        counts[mid] += 1
    reused = sum(1 for mid in selected if counts[mid] > 1)
    by_buyer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_buyer[str(row.get("buyer_id"))].append(row)
    buyer_success = {buyer: _mean(row["success"] for row in items) for buyer, items in by_buyer.items()}
    buyer_spend = {buyer: sum(float(row.get("spend", 0.0)) for row in items) for buyer, items in by_buyer.items()}
    spend_values = list(buyer_spend.values())
    spend_mean = _mean(spend_values) or 0.0
    load_cv = (statistics.pstdev(spend_values) / spend_mean) if len(spend_values) > 1 and spend_mean else 0.0
    # Error propagation: next-step failure after a polluted step.
    propagated = 0
    polluted_transitions = 0
    for prev, nxt in zip(rows, rows[1:]):
        if float(prev.get("pollution", 0.0)) > 0:
            polluted_transitions += 1
            if float(nxt.get("success", 0.0)) == 0:
                propagated += 1
    base_failures = sum(1 for row in rows[1:] if float(row.get("success", 0.0)) == 0)
    return {
        "n_steps": len(rows),
        "memory_reuse": reused / len(selected) if selected else 0.0,
        "no_trade_rate": sum(1 for row in rows if not row.get("selected")) / max(1, len(rows)),
        "worst_agent_success": min(buyer_success.values()) if buyer_success else None,
        "load_balance_cv": load_cv,
        "error_propagation": propagated / polluted_transitions if polluted_transitions else None,
        "base_next_failure": base_failures / max(1, len(rows) - 1),
        "buyer_success": buyer_success,
    }


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"source": str(SUMMARY), "modes": {}}
    for seed_index, seed_result in enumerate(payload.get("results", [])):
        seed = seed_result.get("seed", seed_index)
        for mode, body in seed_result.get("results", {}).items():
            if not isinstance(body, dict):
                continue
            for split in ("final_test", "final_test_seen", "final_test_unseen"):
                part = body.get(split) or {}
                rows = part.get("rows") or []
                if not rows:
                    continue
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    grouped[str(row.get("episode_id"))].append(row)
                per_episode = {ep: episode_metrics(items) for ep, items in grouped.items()}
                out["modes"].setdefault(f"{mode}_seed{seed}", {})[split] = {
                    "episodes": per_episode,
                    "mean_memory_reuse": _mean(m["memory_reuse"] for m in per_episode.values()),
                    "mean_no_trade_rate": _mean(m["no_trade_rate"] for m in per_episode.values()),
                    "worst_agent_success": _mean(
                        m["worst_agent_success"] for m in per_episode.values() if m["worst_agent_success"] is not None
                    ),
                    "mean_load_balance_cv": _mean(m["load_balance_cv"] for m in per_episode.values()),
                    "mean_error_propagation": _mean(
                        m["error_propagation"] for m in per_episode.values() if m["error_propagation"] is not None
                    ),
                    "mean_base_next_failure": _mean(m["base_next_failure"] for m in per_episode.values()),
                }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({mode: body.get("final_test") for mode, body in out["modes"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
