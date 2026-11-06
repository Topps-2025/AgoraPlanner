"""Run the text-memory baseline suite.

This script compares three families on the same natural-language benchmark:

1. allocation policies evaluated with a shared deterministic retriever;
2. the *official* Mem0 package (``mem0ai``), with different ingestion scopes;
3. faithful reimplementations of MemGPT/Letta, RAPTOR, GraphRAG, CAMA,
   GovMem, RCR-Router, and MemPrice.

The official Mem0 rows are the only rows produced by an official package in
this table.  GraphRAG has an official package but its full indexing pipeline is
run separately (``run_official_graphrag.py``) because it needs an LLM and a
local embedding endpoint.  Everything else is explicitly labelled a
reimplementation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.baselines import baseline_allocations  # noqa: E402
from agora_planner.config import DEFAULT_RESULTS_DIR  # noqa: E402
from agora_planner.embeddings import CosineRetriever  # noqa: E402
from agora_planner.llm import _load_dotenv  # noqa: E402
from agora_planner.memory import MemoryAsset  # noqa: E402
from agora_planner.memory_systems import (  # noqa: E402
    CAMAReimpl,
    GovMemReimpl,
    GraphRAGReimpl,
    MemGPTReimpl,
    MemPriceReimpl,
    RAPTORReimpl,
    RCRRouterReimpl,
)
from agora_planner.metrics import summarize_allocation  # noqa: E402
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


def _queries_for_agent(bench: TextMemoryBenchmark, agent_id: str):
    return [bench.queries[task.id] for task in bench.eval_tasks if bench.queries[task.id].agent_id == agent_id]


def _retrieve_with_bundle(
    bench: TextMemoryBenchmark,
    allocation: dict[str, frozenset[str]],
    *,
    top_k: int = 3,
) -> dict[str, list[str]]:
    retrieved: dict[str, list[str]] = {}
    for agent in bench.agents:
        retriever = CosineRetriever()
        for mid in allocation.get(agent.id, frozenset()):
            if mid in bench.memories:
                retriever.add(mid, bench.memories.get(mid).content)
        for query in _queries_for_agent(bench, agent.id):
            retrieved[query.id] = [item.text for item in retriever.search(query.question, top_k=top_k)]
    return retrieved


def _run_allocation_methods(
    bench: TextMemoryBenchmark,
    *,
    seed: int,
    top_k: int = 3,
) -> dict[str, dict]:
    allocations = baseline_allocations(
        bench,
        seed=seed,
        include_proxies=True,
        include_oracle=False,
    )
    results: dict[str, dict] = {}
    for method, allocation in allocations.items():
        start = time.perf_counter()
        retrieved = _retrieve_with_bundle(bench, allocation, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(bench.eval_tasks))
        retrieval = bench.evaluate_retrieval(retrieved, k=top_k)
        welfare = summarize_allocation(bench, allocation)
        results[method] = {
            "kind": "allocation",
            "official": "no",
            "retrieval": retrieval,
            "welfare": {
                "task_success": welfare["task_success"],
                "gt_welfare": welfare["gt_welfare"],
                "gt_value_sum": welfare["gt_value_sum"],
                "memory_count": welfare["memory_count"],
                "common_mode_penalty": welfare["common_mode_penalty"],
                "redundancy_rate": welfare["redundancy_rate"],
                "between_agent_error_corr": welfare["between_agent_error_corr"],
            },
            "latency_ms": latency_ms,
        }
    return results


def _run_system(
    bench: TextMemoryBenchmark,
    system,
    *,
    name: str,
    top_k: int = 3,
    extra: dict | None = None,
) -> dict:
    assets = list(bench.memories)
    start = time.perf_counter()
    system.ingest(assets)
    ingest_s = time.perf_counter() - start
    if hasattr(system, "set_roles"):
        roles = {}
        role_names = ["researcher", "coder", "ops"]
        for idx, agent in enumerate(bench.agents):
            roles[agent.id] = role_names[idx % len(role_names)]
        system.set_roles(roles)
    start = time.perf_counter()
    retrieved: dict[str, list[str]] = {}
    for task in bench.eval_tasks:
        query = bench.queries[task.id]
        items = system.search(query.agent_id, query.question, top_k=top_k)
        retrieved[query.id] = [item.text for item in items]
    latency_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(bench.eval_tasks))
    payload = {
        "kind": "reimplementation",
        "official": "no",
        "retrieval": bench.evaluate_retrieval(retrieved, k=top_k),
        "ingest_seconds": ingest_s,
        "latency_ms": latency_ms,
    }
    if extra:
        payload["extra"] = extra
    if hasattr(system, "summary"):
        try:
            payload["extra"] = {**(payload.get("extra") or {}), **system.summary()}
        except Exception:
            pass
    return payload


def _assets_for_allocation(
    bench: TextMemoryBenchmark, allocation: dict[str, frozenset[str]]
) -> dict[str, list[MemoryAsset]]:
    return {
        agent.id: [bench.memories.get(mid) for mid in allocation.get(agent.id, frozenset())]
        for agent in bench.agents
    }


def _run_mem0(
    bench: TextMemoryBenchmark,
    *,
    mode: str,
    top_k: int = 3,
    infer: bool = False,
    seed: int = 0,
) -> dict:
    from agora_planner.mem0_adapter import Mem0MemoryBank, Mem0Settings

    assets = list(bench.memories)
    allocations = baseline_allocations(
        bench, seed=seed, include_proxies=False, include_oracle=False
    )
    if mode == "private":
        by_agent = {
            agent.id: [asset for asset in assets if asset.owner == agent.id]
            for agent in bench.agents
        }
        shared = False
    elif mode == "shared":
        by_agent = {agent.id: assets for agent in bench.agents}
        shared = True
    elif mode == "central":
        # A budget-matched bundle from the central planner.  The auction's
        # allocation is welfare-identical on this benchmark; this row isolates
        # Mem0's retrieval from the allocation rule.
        by_agent = _assets_for_allocation(bench, allocations["central_oracle"])
        shared = False
    elif mode == "top_k":
        by_agent = _assets_for_allocation(bench, allocations["top_k"])
        shared = False
    elif mode == "govmem":
        gate = GovMemReimpl()
        gate.ingest(assets)
        promoted = list(gate.promoted)
        by_agent = {agent.id: promoted for agent in bench.agents}
        shared = True
    else:
        raise ValueError(f"Unknown Mem0 mode: {mode}")

    start = time.perf_counter()
    bank = Mem0MemoryBank(
        [agent.id for agent in bench.agents],
        shared=shared,
        settings=Mem0Settings(infer=infer, embedding_dims=384),
    )
    bank.ingest(by_agent)
    ingest_s = time.perf_counter() - start
    start = time.perf_counter()
    retrieved: dict[str, list[str]] = {}
    for task in bench.eval_tasks:
        query = bench.queries[task.id]
        items = bank.search(query.agent_id, query.question, top_k=top_k)
        retrieved[query.id] = [item.text for item in items]
    latency_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(bench.eval_tasks))
    bank.close()
    return {
        "kind": "official_mem0",
        "official": "yes",
        "infer": infer,
        "mode": mode,
        "retrieval": bench.evaluate_retrieval(retrieved, k=top_k),
        "ingest_seconds": ingest_s,
        "latency_ms": latency_ms,
        "n_ingested": sum(len(v) for v in by_agent.values()) if not shared else len(assets),
    }


def _aggregate(per_seed: list[dict]) -> dict:
    """Aggregate numeric leaves across seeds (mean/std)."""

    buckets: dict[str, list[float]] = defaultdict(list)

    def walk(prefix: str, value):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            buckets[prefix].append(float(value))

    for row in per_seed:
        walk("", row)
    return {
        key: {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
        for key, values in sorted(buckets.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "text_baselines_v1"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--skip-mem0", action="store_true")
    parser.add_argument("--skip-reimpl", action="store_true")
    parser.add_argument("--mem0-infer-seed", type=int, default=None)
    args = parser.parse_args()

    _load_dotenv()
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for seed in seeds:
        bench = _bench(seed)
        row: dict = {
            "seed": seed,
            "benchmark": bench.describe(),
            "allocation": _run_allocation_methods(bench, seed=seed, top_k=args.top_k),
            "systems": {},
        }
        if not args.skip_mem0:
            for mode in ["private", "shared", "central", "top_k", "govmem"]:
                row["systems"][f"mem0_{mode}"] = _run_mem0(
                    bench, mode=mode, top_k=args.top_k, seed=seed
                )
            if args.mem0_infer_seed is not None and seed == args.mem0_infer_seed:
                row["systems"]["mem0_shared_infer"] = _run_mem0(
                    bench, mode="shared", top_k=args.top_k, infer=True, seed=seed
                )
        if not args.skip_reimpl:
            row["systems"].update(
                {
                    "memgpt_reimpl": _run_system(
                        bench, MemGPTReimpl(core_slots=2, recall_slots=4), name="memgpt_reimpl", top_k=args.top_k
                    ),
                    "raptor_reimpl": _run_system(
                        bench, RAPTORReimpl(branch_facts=4), name="raptor_reimpl", top_k=args.top_k
                    ),
                    "graphrag_reimpl": _run_system(
                        bench, GraphRAGReimpl(expand_neighbors=3), name="graphrag_reimpl", top_k=args.top_k
                    ),
                    "cama_reimpl": _run_system(
                        bench, CAMAReimpl(candidate_k=12), name="cama_reimpl", top_k=args.top_k
                    ),
                    "govmem_reimpl": _run_system(
                        bench, GovMemReimpl(), name="govmem_reimpl", top_k=args.top_k
                    ),
                    "rcr_reimpl": _run_system(
                        bench, RCRRouterReimpl(token_budget=48), name="rcr_reimpl", top_k=args.top_k
                    ),
                    "memprice_reimpl": _run_system(
                        bench, MemPriceReimpl(target_tokens=48), name="memprice_reimpl", top_k=args.top_k
                    ),
                    # No-quality ablations: the same mechanisms without the
                    # benchmark's quality metadata.
                    "raptor_noquality": _run_system(
                        bench,
                        RAPTORReimpl(branch_facts=4, use_quality=False),
                        name="raptor_noquality",
                        top_k=args.top_k,
                    ),
                    "graphrag_noquality": _run_system(
                        bench,
                        GraphRAGReimpl(expand_neighbors=3, use_quality=False),
                        name="graphrag_noquality",
                        top_k=args.top_k,
                    ),
                    "cama_noquality": _run_system(
                        bench,
                        CAMAReimpl(candidate_k=12, use_quality=False),
                        name="cama_noquality",
                        top_k=args.top_k,
                    ),
                    "rcr_noquality": _run_system(
                        bench,
                        RCRRouterReimpl(token_budget=48, use_quality=False),
                        name="rcr_noquality",
                        top_k=args.top_k,
                    ),
                    "memprice_noquality": _run_system(
                        bench,
                        MemPriceReimpl(target_tokens=48, use_quality=False),
                        name="memprice_noquality",
                        top_k=args.top_k,
                    ),
                }
            )
        all_rows.append(row)
        with open(output_dir / "text_baselines.json", "w", encoding="utf-8") as handle:
            json.dump(all_rows, handle, indent=2)
        print(
            f"seed {seed}: "
            f"auction_acc={row['allocation']['central_oracle']['retrieval']['answer_accuracy@1']:.3f} "
            f"shared_all_acc={row['allocation']['shared_all']['retrieval']['answer_accuracy@1']:.3f} "
            f"mem0_shared_acc={row['systems'].get('mem0_shared', {}).get('retrieval', {}).get('answer_accuracy@1', float('nan')):.3f}"
        )

    aggregate = _aggregate(all_rows)
    with open(output_dir / "aggregate.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(f"wrote {output_dir / 'text_baselines.json'}")
    print(f"wrote {output_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
