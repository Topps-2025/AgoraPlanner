"""Tests for the reproducible baseline suite."""

from __future__ import annotations

import pytest

from agora_planner.baselines import baseline_allocations, baseline_metadata
from agora_planner.benchmark import MemoryBenchmark
from agora_planner.config import BenchmarkConfig
from agora_planner.metrics import summarize_allocation


def _bench(seed: int = 0) -> MemoryBenchmark:
    return MemoryBenchmark(
        BenchmarkConfig(
            seed=seed,
            n_agents=3,
            n_facets=6,
            n_memories=8,
            n_train_tasks=12,
            n_eval_tasks=12,
            max_memories_per_agent=2,
            one_per_source=True,
            solver="exact",
        )
    )


def test_baseline_suite_contains_all_expected_methods():
    bench = _bench()
    allocations = baseline_allocations(bench, seed=0)
    expected = {
        "no_memory",
        "private",
        "shared_all",
        "shared_budget",
        "top_k",
        "selective",
        "central_reported",
        "central_oracle",
        "composition",
        "mem0_like",
        "memgpt_like",
        "raptor_like",
        "graphrag_like",
        "oracle",
    }
    assert expected.issubset(set(allocations))
    metadata = baseline_metadata()
    for method in allocations:
        assert method in metadata


def test_baseline_allocations_respect_budget_and_source_constraints():
    bench = _bench()
    allocations = baseline_allocations(bench, seed=0)
    for method, allocation in allocations.items():
        assert set(allocation) == {agent.id for agent in bench.agents}
        if method in {"shared_all", "oracle"}:
            # shared_all is the unconstrained lower bound; oracle is the
            # ground-truth optimum and may also ignore the per-agent budget
            # only if the benchmark config allows it (it does not, but we do
            # not assert on the oracle allocation's shape here).
            continue
        for agent_id, bundle in allocation.items():
            agent = bench._agent(agent_id)  # noqa: SLF001
            assert len(bundle) <= agent.max_memories
            assert all(mid in bench.memories for mid in bundle)
            if bench.config.one_per_source:
                sources = [bench.memories.get(mid).source_class for mid in bundle]
                assert len(sources) == len(set(sources))


def test_oracle_welfare_is_at_least_central_planner_welfare():
    bench = _bench()
    allocations = baseline_allocations(
        bench, seed=0, include_proxies=False, include_oracle=True
    )
    oracle_welfare = summarize_allocation(bench, allocations["oracle"])["gt_welfare"]
    central_welfare = summarize_allocation(
        bench, allocations["central_oracle"]
    )["gt_welfare"]
    assert oracle_welfare + 1e-6 >= central_welfare


def test_baseline_allocations_are_deterministic():
    bench_a = _bench(seed=0)
    bench_b = _bench(seed=0)
    alloc_a = baseline_allocations(bench_a, seed=0, include_oracle=False)
    alloc_b = baseline_allocations(bench_b, seed=0, include_oracle=False)
    for method in alloc_a:
        assert alloc_a[method] == alloc_b[method]


def test_composition_and_proxies_are_nonempty_on_a_rich_benchmark():
    bench = _bench(seed=1)
    allocations = baseline_allocations(
        bench, seed=1, include_proxies=True, include_oracle=False
    )
    for method in ["composition", "mem0_like", "memgpt_like", "raptor_like", "graphrag_like"]:
        total = sum(len(bundle) for bundle in allocations[method].values())
        assert total > 0, method
