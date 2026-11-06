"""Metrics for memory-market experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from .auction import AuctionResult
from .benchmark import MemoryBenchmark
from .memory import MemoryStore


def gini(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    if arr.mean() <= 1e-12:
        return 0.0
    diff = np.abs(arr[:, None] - arr[None, :]).sum()
    return float(diff / (2.0 * len(arr) ** 2 * arr.mean()))


def common_mode_penalty(
    benchmark: MemoryBenchmark,
    allocation: dict[str, frozenset[str]],
    *,
    common_mode_lambda: float | None = None,
) -> float:
    counts: dict[str, int] = {}
    for bundle in allocation.values():
        for mid in bundle:
            counts[mid] = counts.get(mid, 0) + 1
    lam = (
        benchmark.config.common_mode_lambda
        if common_mode_lambda is None
        else float(common_mode_lambda)
    )
    penalty = 0.0
    for mid, count in counts.items():
        if count <= 1 or mid not in benchmark.memories:
            continue
        asset = benchmark.memories.get(mid)
        penalty += lam * asset.correlation_risk * (count * (count - 1) / 2.0)
    return float(penalty)


def summarize_allocation(
    benchmark: MemoryBenchmark,
    allocation: dict[str, frozenset[str]],
    *,
    payments: dict[str, float] | None = None,
    prices: dict[str, float] | None = None,
    tasks=None,
    seed: int = 7,
    common_mode_lambda: float | None = None,
) -> dict[str, Any]:
    """Compute ground-truth, independence, and cost metrics for an allocation."""

    tasks = benchmark.eval_tasks if tasks is None else tasks
    payments = payments or {agent.id: 0.0 for agent in benchmark.agents}
    prices = prices or {}
    agent_ids = [agent.id for agent in benchmark.agents]
    allocation = {aid: frozenset(allocation.get(aid, frozenset())) for aid in agent_ids}

    success = [
        benchmark.expected_success(aid, allocation[aid], tasks) for aid in agent_ids
    ]
    gt_values = {
        aid: benchmark.agent_value(aid, allocation[aid], tasks) for aid in agent_ids
    }
    gt_utilities = {
        aid: gt_values[aid] - payments.get(aid, 0.0) for aid in agent_ids
    }
    admitted = sorted(set().union(*allocation.values()) if allocation else set())
    total_cost = float(
        sum(benchmark.memories.get(mid).production_cost for mid in admitted)
    )
    cm_penalty = common_mode_penalty(
        benchmark, allocation, common_mode_lambda=common_mode_lambda
    )
    gt_welfare = float(
        sum(gt_values.values()) - total_cost - cm_penalty
    )

    # Redundancy: a memory is redundant if its total ground-truth marginal value
    # across the agents holding it is (near) zero.
    marginal_values: dict[str, float] = {}
    for mid in admitted:
        total = 0.0
        for aid in agent_ids:
            if mid in allocation[aid]:
                total += benchmark.memory_marginal_value(
                    aid, mid, allocation[aid] - {mid}, tasks
                )
        marginal_values[mid] = float(total)
    redundant = [mid for mid, value in marginal_values.items() if value <= 1e-6]

    sources = [benchmark.memories.get(mid).source_class for mid in admitted]
    unique_sources = len(set(sources))
    independence_index = unique_sources / len(admitted) if admitted else 0.0

    # Pairwise team redundancy: fraction of (agent, memory) pairs whose memory
    # duplicates another memory held by the same agent on facet + source.
    pair_count = 0
    duplicate_pairs = 0
    for aid in agent_ids:
        bundle = sorted(allocation[aid])
        for i in range(len(bundle)):
            for j in range(i + 1, len(bundle)):
                left = benchmark.memories.get(bundle[i])
                right = benchmark.memories.get(bundle[j])
                pair_count += 1
                if (
                    left.facet == right.facet
                    and left.source_class == right.source_class
                ):
                    duplicate_pairs += 1
    team_redundancy = duplicate_pairs / pair_count if pair_count else 0.0

    admitted_common = sum(
        1
        for mid in admitted
        if benchmark.memories.get(mid).origin == "common_knowledge"
    )
    admitted_redundant = sum(
        1 for mid in admitted if benchmark.memories.get(mid).origin == "redundant"
    )
    admitted_poison = sum(
        1 for mid in admitted if benchmark.memories.get(mid).metadata.get("poison")
    )
    pool_ids = benchmark.memories.ids()
    retired_common = sum(
        1
        for mid in pool_ids
        if mid not in admitted
        and benchmark.memories.get(mid).origin == "common_knowledge"
    )
    retired_redundant = sum(
        1
        for mid in pool_ids
        if mid not in admitted and benchmark.memories.get(mid).origin == "redundant"
    )
    retired_poison = sum(
        1
        for mid in pool_ids
        if mid not in admitted
        and benchmark.memories.get(mid).metadata.get("poison")
    )

    return {
        "task_success": float(np.mean(success)) if success else 0.0,
        "gt_welfare": gt_welfare,
        "gt_value_sum": float(sum(gt_values.values())),
        "gt_utility_sum": float(sum(gt_utilities.values())),
        "gt_min_utility": float(min(gt_utilities.values())) if gt_utilities else 0.0,
        "gt_avg_utility": float(np.mean(list(gt_utilities.values()))) if gt_utilities else 0.0,
        "memory_count": len(admitted),
        "admitted_count": len(admitted),
        "pool_size": len(pool_ids),
        "admitted_common_knowledge": admitted_common,
        "admitted_redundant": admitted_redundant,
        "admitted_poison": admitted_poison,
        "retired_common_knowledge": retired_common,
        "retired_redundant": retired_redundant,
        "retired_poison": retired_poison,
        "redundancy_rate": len(redundant) / len(admitted) if admitted else 0.0,
        "zero_marginal_rate": len(redundant) / len(admitted) if admitted else 0.0,
        "admitted_zero_price_rate": (
            sum(1 for mid in admitted if prices.get(mid, 0.0) <= 1e-9) / len(admitted)
            if admitted
            else 0.0
        ),
        "zero_price_rate": (
            sum(1 for mid in pool_ids if prices.get(mid, 0.0) <= 1e-9) / len(pool_ids)
            if pool_ids
            else 0.0
        ),
        "independence_index": float(independence_index),
        "team_redundancy": float(team_redundancy),
        "between_agent_error_corr": benchmark.error_correlation(
            allocation, tasks, seed=seed
        ),
        "total_cost": total_cost,
        "common_mode_penalty": cm_penalty,
        "revenue": float(sum(payments.values())),
        "surplus": float(sum(payments.values()) - total_cost),
        "mean_price": float(np.mean(list(prices.values()))) if prices else 0.0,
        "price_gini": gini(list(prices.values())),
        "marginal_values": {mid: round(value, 6) for mid, value in marginal_values.items()},
        "prices": {mid: round(value, 6) for mid, value in prices.items()},
        "allocation": {aid: sorted(bundle) for aid, bundle in allocation.items()},
    }


def summarize_auction(
    benchmark: MemoryBenchmark,
    result: AuctionResult,
    *,
    tasks=None,
    seed: int = 7,
    common_mode_lambda: float | None = None,
) -> dict[str, Any]:
    summary = summarize_allocation(
        benchmark,
        result.allocation,
        payments=result.payments,
        prices=result.prices,
        tasks=tasks,
        seed=seed,
        common_mode_lambda=common_mode_lambda,
    )
    summary.update(
        {
            "mechanism_welfare": result.welfare,
            "mechanism_utility_sum": float(sum(result.utilities.values())),
            "mechanism_min_utility": float(min(result.utilities.values()))
            if result.utilities
            else 0.0,
            "solver": result.solver,
            "mechanism_common_mode_penalty": result.common_mode_penalty,
            "mechanism_total_cost": result.total_cost,
            "mechanism_revenue": result.revenue,
            "mechanism_surplus": result.surplus,
        }
    )
    return summary


def oracle_welfare(
    benchmark: MemoryBenchmark,
    *,
    tasks=None,
    solver: str = "exact",
) -> tuple[float, dict[str, frozenset[str]]]:
    """Ground-truth welfare optimum (upper bound for mechanism evaluation)."""

    from .auction import MemoryAuction
    from .valuation import GroundTruthValuator

    valuator = GroundTruthValuator(benchmark)
    auction = MemoryAuction(
        benchmark.memories,
        benchmark.agents,
        valuator,
        benchmark.config,
    )
    # Temporarily override the task split used by the valuator.  The valuator
    # reads benchmark.agent_value(..., split="eval"), so for custom tasks we
    # fall back to the benchmark's eval set.
    del tasks
    welfare, allocation, used_solver = auction.solve(split="eval", solver=solver)
    return welfare, allocation
