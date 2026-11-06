from __future__ import annotations

import math

import pytest

from agora_planner.auction import MemoryAuction
from agora_planner.benchmark import MemoryBenchmark
from agora_planner.config import BenchmarkConfig
from agora_planner.metrics import summarize_auction
from agora_planner.valuation import GroundTruthValuator


def tiny_benchmark(**overrides):
    params = dict(
        seed=3,
        n_agents=2,
        n_facets=4,
        n_memories=4,
        n_train_tasks=10,
        n_eval_tasks=10,
        max_memories_per_agent=2,
        max_combinations=2_000_000,
        solver="exact",
        common_mode_lambda=0.5,
    )
    params.update(overrides)
    return MemoryBenchmark(BenchmarkConfig(**params))


def run_auction(bench, valuator=None, solver="exact"):
    valuator = valuator or GroundTruthValuator(bench)
    auction = MemoryAuction(bench.memories, bench.agents, valuator, bench.config)
    return auction, auction.run(split="eval") if solver == "auto" else auction.run(split="eval")


def test_exact_is_at_least_as_good_as_beam_and_greedy():
    bench = tiny_benchmark()
    valuator = GroundTruthValuator(bench)
    exact = MemoryAuction(bench.memories, bench.agents, valuator, bench.config)
    exact_welfare, _, _ = exact.solve(split="eval", solver="exact")
    beam_welfare, _, _ = exact.solve(split="eval", solver="beam")
    greedy_welfare, _, _ = exact.solve(split="eval", solver="greedy")
    assert exact_welfare + 1e-9 >= beam_welfare
    assert exact_welfare + 1e-9 >= greedy_welfare


def test_individual_rationality_and_budget():
    bench = tiny_benchmark()
    _, result = run_auction(bench)
    for agent in bench.agents:
        assert result.utilities[agent.id] >= -1e-7
        assert len(result.allocation[agent.id]) <= agent.max_memories
        sources = [bench.memories.get(mid).source_class for mid in result.allocation[agent.id]]
        assert len(sources) == len(set(sources))


def test_common_knowledge_memories_are_not_admitted():
    bench = tiny_benchmark()
    _, result = run_auction(bench)
    common = [m for m in bench.memories if m.origin == "common_knowledge"]
    assert common, "benchmark should contain common-knowledge memories"
    for asset in common:
        assert asset.id not in result.admitted


def test_redundant_memories_are_not_double_allocated_per_agent():
    bench = tiny_benchmark()
    _, result = run_auction(bench)
    for agent in bench.agents:
        bundle = result.allocation[agent.id]
        keys = [
            (bench.memories.get(mid).facet, bench.memories.get(mid).source_class)
            for mid in bundle
        ]
        assert len(keys) == len(set(keys))


def test_vcg_truthfulness_sanity_check():
    bench = tiny_benchmark()
    base = GroundTruthValuator(bench)
    _, truthful = run_auction(bench)
    agent_id = bench.agents[0].id
    truthful_value = base.value(agent_id, truthful.allocation[agent_id])
    truthful_utility = truthful_value - truthful.payments[agent_id]

    class ScaledValuator:
        def __init__(self, base, agent_id, factor):
            self.base = base
            self.agent_id = agent_id
            self.factor = factor

        def value(self, aid, bundle, *, split="eval"):
            value = self.base.value(aid, bundle, split=split)
            return value * self.factor if aid == self.agent_id else value

        def marginal(self, aid, memory_id, bundle, *, split="eval"):
            return self.value(aid, bundle | {memory_id}, split=split) - self.value(
                aid, bundle, split=split
            )

    for factor in (0.5, 1.5):
        scaled = ScaledValuator(base, agent_id, factor)
        _, result = run_auction(bench, scaled)
        true_value = base.value(agent_id, result.allocation[agent_id])
        true_utility = true_value - result.payments[agent_id]
        assert true_utility <= truthful_utility + 1e-6


def test_common_mode_penalty_is_charged():
    bench = tiny_benchmark(common_mode_lambda=2.0)
    auction, result = run_auction(bench)
    summary = summarize_auction(bench, result)
    shared = [
        mid
        for mid in result.admitted
        if sum(1 for bundle in result.allocation.values() if mid in bundle) > 1
    ]
    if shared:
        assert result.common_mode_penalty > 0
        assert summary["common_mode_penalty"] > 0


def test_prices_are_non_negative():
    bench = tiny_benchmark()
    _, result = run_auction(bench)
    assert all(price >= 0 for price in result.prices.values())
