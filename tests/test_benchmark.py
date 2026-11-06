from __future__ import annotations

import math

from agora_planner.benchmark import MemoryBenchmark
from agora_planner.config import BenchmarkConfig


def build_bench():
    return MemoryBenchmark(
        BenchmarkConfig(
            seed=11,
            n_agents=3,
            n_facets=6,
            n_memories=6,
            n_train_tasks=12,
            n_eval_tasks=12,
            max_memories_per_agent=2,
        )
    )


def test_benchmark_shapes_and_feasible_bundles():
    bench = build_bench()
    assert len(bench.agents) == 3
    assert len(bench.facets) == 6
    assert len(bench.train_tasks) == 12
    for agent in bench.agents:
        bundles = bench.feasible_bundles(agent.id)
        assert frozenset() in bundles
        assert all(len(bundle) <= agent.max_memories for bundle in bundles)
        for bundle in bundles:
            sources = [bench.memories.get(mid).source_class for mid in bundle]
            assert len(sources) == len(set(sources))


def test_values_are_finite_and_success_in_range():
    bench = build_bench()
    for agent in bench.agents:
        value = bench.agent_value(agent.id, frozenset())
        assert math.isfinite(value)
        for memory in bench.memories:
            v = bench.agent_value(agent.id, {memory.id})
            assert math.isfinite(v)
        for task in bench.eval_tasks:
            p = bench.success_probability(agent.id, frozenset(), task)
            assert 0.0 <= p <= 1.0


def test_common_knowledge_has_near_zero_or_negative_value():
    bench = build_bench()
    common = [m for m in bench.memories if m.origin == "common_knowledge"]
    assert common
    for asset in common:
        best = max(bench.agent_value(agent.id, {asset.id}) for agent in bench.agents)
        assert best <= 0.05


def test_clone_with_store_isolated():
    bench = build_bench()
    clone = bench.clone_with_store(bench.memories, seed=99)
    assert len(clone.memories) == len(bench.memories)
    assert clone.eval_tasks is not bench.eval_tasks
    assert len(clone.agents) == len(bench.agents)
    # Mutating the clone's store must not affect the original.
    first = clone.memories.ids()[0]
    clone.memories.remove(first)
    assert first in bench.memories


def test_error_correlation_is_bounded():
    bench = build_bench()
    allocation = {agent.id: frozenset() for agent in bench.agents}
    corr = bench.error_correlation(allocation)
    assert -1.0 <= corr <= 1.0


def test_clone_with_shifted_tasks_changes_facet_distribution():
    bench = build_bench()
    shifted = bench.clone_with_shifted_tasks(seed=123, shift_strength=0.7)
    weights = shifted.config.task_facet_weights
    assert weights
    counts = {facet: 0 for facet in bench.facets}
    for task in bench.train_tasks:
        for facet in task.required_facets:
            counts[facet] += 1
    rare = min(counts, key=counts.get)
    common = max(counts, key=counts.get)
    assert weights[rare] > weights[common]
    assert len(shifted.eval_tasks) == len(bench.eval_tasks)


def test_clone_with_shifted_fresh_agents_keeps_store_and_agents():
    bench = build_bench()
    shifted = bench.clone_with_fresh_agents(
        seed=321, shifted=True, shift_strength=0.7
    )
    assert shifted.config.task_facet_weights
    assert len(shifted.memories) == len(bench.memories)
    assert len(shifted.agents) == len(bench.agents)
    assert shifted.eval_tasks is not bench.eval_tasks
