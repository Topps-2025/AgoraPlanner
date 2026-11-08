"""Official AutoGen multi-agent memory-sharing pilot.

This is a small orchestration check, not a MultiAgentBench run.  It asks
whether a team of official AutoGen agents, each grounded in a different memory
scope, can recover an access code that only one member holds.

Run with ``PYTHONPATH=src``.  The adapter inserts the repository-local
``.vendor_autogen`` directory automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.autogen_adapter import AutoGenMemoryTeam, AutoGenSettings  # noqa: E402
from agora_planner.baselines import baseline_allocations  # noqa: E402
from agora_planner.config import DEFAULT_RESULTS_DIR  # noqa: E402
from agora_planner.memory_systems import GovMemReimpl  # noqa: E402
from agora_planner.text_benchmark import (  # noqa: E402
    TextBenchmarkConfig,
    TextMemoryBenchmark,
)


def _bench(seed: int) -> TextMemoryBenchmark:
    return TextMemoryBenchmark(
        TextBenchmarkConfig(
            seed=seed,
            n_agents=3,
            n_facets=16,
            facts_per_agent=4,
            n_public_facts=4,
            decoy_fraction=0.40,
            decoy_style="exact",
            n_train_tasks=24,
            n_eval_tasks=24,
            max_memories_per_agent=3,
            solver="beam",
        )
    )


def _scopes(bench: TextMemoryBenchmark, seed: int) -> dict[str, dict[str, frozenset[str]]]:
    assets = list(bench.memories)
    allocations = baseline_allocations(bench, seed=seed, include_proxies=False, include_oracle=False)
    gate = GovMemReimpl()
    gate.ingest(assets)
    promoted = frozenset(asset.id for asset in gate.promoted)
    return {
        "private": {
            agent.id: frozenset(asset.id for asset in assets if asset.owner == agent.id)
            for agent in bench.agents
        },
        "shared_all": {
            agent.id: frozenset(asset.id for asset in assets) for agent in bench.agents
        },
        "central_bundle": {
            agent.id: frozenset(allocations["central_oracle"].get(agent.id, frozenset()))
            for agent in bench.agents
        },
        "govmem_promoted": {agent.id: promoted for agent in bench.agents},
    }


async def _run(seed: int, n_queries: int, top_k: int, output_dir: Path) -> dict:
    bench = _bench(seed)
    scopes = _scopes(bench, seed)
    team = AutoGenMemoryTeam(AutoGenSettings(max_tokens=512))
    rows: dict[str, list[dict]] = {}
    for mode, memory_ids in scopes.items():
        rows[mode] = []
        queries = [
            bench.queries[bench.eval_tasks[idx].id] for idx in range(min(n_queries, len(bench.eval_tasks)))
        ]
        for query in queries:
            answer = await team.answer_query(
                bench, query, memory_ids_by_agent=memory_ids, top_k=top_k
            )
            rows[mode].append(
                {
                    "query_id": answer.query_id,
                    "agent_id": answer.agent_id,
                    "gold": answer.gold,
                    "parsed": answer.parsed_answer,
                    "correct": answer.correct,
                    "judge": answer.judge_answer,
                    "agent_answers": answer.agent_answers,
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens,
                    "latency_ms": answer.latency_ms,
                }
            )
        print(f"seed {seed} {mode}: {sum(r['correct'] for r in rows[mode])}/{len(rows[mode])}", flush=True)
    payload = {"seed": seed, "n_queries": n_queries, "top_k": top_k, "modes": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "autogen_pilot.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-queries", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "autogen_pilot_v1"),
    )
    args = parser.parse_args()
    payload = asyncio.run(
        _run(args.seed, args.n_queries, args.top_k, Path(args.output_dir))
    )
    for mode, rows in payload["modes"].items():
        accuracy = sum(row["correct"] for row in rows) / max(1, len(rows))
        tokens = sum(row["prompt_tokens"] + row["completion_tokens"] for row in rows) / max(1, len(rows))
        latency = sum(row["latency_ms"] for row in rows) / max(1, len(rows))
        print(f"{mode:18s} accuracy={accuracy:.3f} tokens={tokens:.0f} latency_ms={latency:.0f}")


if __name__ == "__main__":
    main()
