"""Reproducible baseline policies for the memory-allocation benchmark.

The baselines fall into three groups:

1. **Allocation baselines**: no memory, private, shared-all, budget-matched
   shared, top-k, selective (greedy marginal value), central planner.
2. **Composition baselines**: a retrieval + quality-gate + correlation-aware
   arbitration pipeline, representing the "RCR + CAMA + GovMem + MemPrice"
   composition without a joint market.
3. **Memory-system proxies**: lightweight offline proxies for Mem0 / MemGPT /
   RAPTOR / GraphRAG policies on the synthetic benchmark.  These are *proxies*,
   not official reproductions; the official systems require their own
   datasets/LLM APIs and are handled by separate adapters.

All policies respect the benchmark's per-agent context budget and source
matroid, except ``shared_all``, which is intentionally unconstrained and serves
as the over-sharing lower bound.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .auction import MemoryAuction
from .benchmark import MemoryBenchmark
from .memory import MemoryAsset
from .metrics import oracle_welfare
from .valuation import GroundTruthValuator, NoisyBidder

Allocation = dict[str, frozenset[str]]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _agent_ids(bench: MemoryBenchmark) -> list[str]:
    return [agent.id for agent in bench.agents]


def _empty(bench: MemoryBenchmark) -> Allocation:
    return {agent_id: frozenset() for agent_id in _agent_ids(bench)}


def _task_mass(bench: MemoryBenchmark, split: str = "eval") -> dict[str, float]:
    mass = {facet: 0.0 for facet in bench.facets}
    for task in bench._tasks(split):  # noqa: SLF001 - benchmark helper
        for facet, weight in task.required_facets.items():
            mass[facet] = mass.get(facet, 0.0) + float(weight)
    return mass


def _source_penalty(bench: MemoryBenchmark, asset: MemoryAsset) -> float:
    return float(asset.correlation_risk)


def _constraints_ok(bench: MemoryBenchmark, agent_id: str, bundle: Iterable[str]) -> bool:
    bundle = frozenset(bundle)
    agent = bench._agent(agent_id)  # noqa: SLF001
    if len(bundle) > int(agent.max_memories):
        return False
    if bench.config.one_per_source:
        sources = [bench.memories.get(mid).source_class for mid in bundle]
        if len(set(sources)) != len(sources):
            return False
    return all(mid in bench.memories for mid in bundle)


def _take_ranked(
    bench: MemoryBenchmark,
    agent_id: str,
    ranked_ids: Sequence[str],
    *,
    quality_gate: float | None = None,
) -> frozenset[str]:
    agent = bench._agent(agent_id)  # noqa: SLF001
    budget = max(0, int(agent.max_memories))
    bundle: list[str] = []
    sources: set[str] = set()
    for mid in ranked_ids:
        if len(bundle) >= budget:
            break
        if mid not in bench.memories:
            continue
        asset = bench.memories.get(mid)
        if quality_gate is not None and asset.quality < quality_gate:
            continue
        if bench.config.one_per_source and asset.source_class in sources:
            continue
        bundle.append(mid)
        sources.add(asset.source_class)
    return frozenset(bundle)


def _rank_by_score(
    bench: MemoryBenchmark,
    scores: dict[str, float],
) -> list[str]:
    return [
        mid
        for mid, _ in sorted(
            scores.items(),
            key=lambda item: (item[1], bench.memories.get(item[0]).quality),
            reverse=True,
        )
    ]


def _quality_scores(bench: MemoryBenchmark) -> dict[str, float]:
    return {
        asset.id: float(asset.quality)
        for asset in bench.memories
    }


def _topk_scores(bench: MemoryBenchmark, split: str = "eval") -> dict[str, float]:
    mass = _task_mass(bench, split)
    return {
        asset.id: float(asset.quality) * (0.2 + mass.get(asset.facet, 0.0))
        for asset in bench.memories
    }


def _rcr_scores(bench: MemoryBenchmark, agent_id: str, split: str = "eval") -> dict[str, float]:
    mass = _task_mass(bench, split)
    agent = bench._agent(agent_id)  # noqa: SLF001
    scores: dict[str, float] = {}
    for asset in bench.memories:
        # Role-aware routing: task facet mass, discounted by what the agent
        # already knows, and by source correlation.
        known = float(agent.private_knowledge.get(asset.facet, 0.0))
        relevance = mass.get(asset.facet, 0.0)
        scores[asset.id] = (
            float(asset.quality)
            * (0.2 + relevance)
            * (1.0 - 0.7 * known)
            * (1.0 - 0.3 * _source_penalty(bench, asset))
        )
    return scores


def _importance_scores(bench: MemoryBenchmark, split: str = "eval") -> dict[str, float]:
    mass = _task_mass(bench, split)
    max_round = max([asset.created_round for asset in bench.memories] + [0])
    scores: dict[str, float] = {}
    for asset in bench.memories:
        recency = 1.0 if max_round <= 0 else 1.0 + 0.1 * (asset.created_round / max_round)
        scores[asset.id] = (
            float(asset.quality)
            * (0.2 + mass.get(asset.facet, 0.0))
            * recency
        )
    return scores


# ----------------------------------------------------------------------
# Allocation baselines
# ----------------------------------------------------------------------


def no_memory(bench: MemoryBenchmark) -> Allocation:
    return _empty(bench)


def private(bench: MemoryBenchmark, split: str = "eval") -> Allocation:
    scores = _quality_scores(bench)
    allocation: Allocation = {}
    for agent_id in _agent_ids(bench):
        owned = [
            mid
            for mid in bench.memories.ids()
            if bench.memories.get(mid).owner == agent_id
        ]
        ranked = sorted(owned, key=lambda mid: scores[mid], reverse=True)
        allocation[agent_id] = _take_ranked(bench, agent_id, ranked)
    return allocation


def shared_all(bench: MemoryBenchmark) -> Allocation:
    # Intentionally unconstrained: the over-sharing lower bound.
    all_ids = frozenset(bench.memories.ids())
    return {agent_id: all_ids for agent_id in _agent_ids(bench)}


def shared_budget(bench: MemoryBenchmark, split: str = "eval") -> Allocation:
    ranked = _rank_by_score(bench, _quality_scores(bench))
    return {
        agent_id: _take_ranked(bench, agent_id, ranked)
        for agent_id in _agent_ids(bench)
    }


def top_k(bench: MemoryBenchmark, split: str = "eval") -> Allocation:
    ranked = _rank_by_score(bench, _topk_scores(bench, split))
    return {
        agent_id: _take_ranked(bench, agent_id, ranked)
        for agent_id in _agent_ids(bench)
    }


def _greedy_agent_bundle(
    bench: MemoryBenchmark,
    agent_id: str,
    *,
    split: str = "eval",
    threshold: float = 1e-4,
    quality_gate: float | None = None,
) -> frozenset[str]:
    agent = bench._agent(agent_id)  # noqa: SLF001
    bundle: frozenset[str] = frozenset()
    remaining = set(bench.memories.ids())
    while len(bundle) < int(agent.max_memories) and remaining:
        best_mid = None
        best_gain = threshold
        for mid in sorted(remaining):
            asset = bench.memories.get(mid)
            if quality_gate is not None and asset.quality < quality_gate:
                continue
            if bench.config.one_per_source and any(
                bench.memories.get(x).source_class == asset.source_class
                for x in bundle
            ):
                continue
            gain = bench.memory_marginal_value(
                agent_id, mid, bundle, split=split
            )
            if gain > best_gain:
                best_gain = gain
                best_mid = mid
        if best_mid is None:
            break
        bundle = bundle | {best_mid}
        remaining.remove(best_mid)
    return bundle


def selective(bench: MemoryBenchmark, split: str = "eval") -> Allocation:
    return {
        agent_id: _greedy_agent_bundle(bench, agent_id, split=split)
        for agent_id in _agent_ids(bench)
    }


def central_planner(
    bench: MemoryBenchmark,
    *,
    seed: int = 0,
    reported: bool = False,
    solver: str | None = None,
) -> Allocation:
    """Centralized welfare maximization with the same information as the market.

    ``reported=False`` uses ground-truth values (an oracle central planner);
    ``reported=True`` uses the same noisy bids as the auction, so the comparison
    isolates the mechanism (payments/IR) from the allocation computation.
    """

    valuator = (
        NoisyBidder(bench, bench.config, seed=seed)
        if reported
        else GroundTruthValuator(bench)
    )
    auction = MemoryAuction(
        bench.memories,
        bench.agents,
        valuator,
        bench.config,
    )
    _, allocation, _ = auction.solve(
        split="eval", solver=solver or bench.config.solver
    )
    return {aid: frozenset(bundle) for aid, bundle in allocation.items()}


# ----------------------------------------------------------------------
# Composition baseline: RCR + CAMA + GovMem + MemPrice (proxy)
# ----------------------------------------------------------------------


def composition(bench: MemoryBenchmark, split: str = "eval") -> Allocation:
    """A retrieval + quality-gate + correlation-aware arbitration pipeline.

    This is a *proxy* for composing recent components (role-aware routing,
    correlation arbitration, false-promotion gating, price-aware selection),
    not an official reproduction of any single system.
    """

    # 1) Role-aware retrieval (RCR-like).
    allocation: dict[str, set[str]] = {}
    for agent_id in _agent_ids(bench):
        ranked = _rank_by_score(bench, _rcr_scores(bench, agent_id, split))
        allocation[agent_id] = set(
            _take_ranked(bench, agent_id, ranked, quality_gate=0.35)
        )

    # 2) Quality / poison gate (GovMem-like).
    for agent_id in list(allocation):
        allocation[agent_id] = {
            mid
            for mid in allocation[agent_id]
            if not bench.memories.get(mid).metadata.get("poison")
            and bench.memories.get(mid).quality >= 0.35
        }

    # 3) Correlation-aware arbitration (CAMA-like): if a high-correlation memory
    # is held by multiple agents, keep the holder with the highest marginal
    # value and try to replace it for the others with a different-source
    # alternative.
    counts: dict[str, int] = {}
    for bundle in allocation.values():
        for mid in bundle:
            counts[mid] = counts.get(mid, 0) + 1
    for mid, count in counts.items():
        if count <= 1:
            continue
        asset = bench.memories.get(mid)
        if asset.correlation_risk < 0.5:
            continue
        holders = sorted(
            [aid for aid, bundle in allocation.items() if mid in bundle],
            key=lambda aid: bench.memory_marginal_value(
                aid, mid, frozenset(allocation[aid] - {mid}), split=split
            ),
            reverse=True,
        )
        for aid in holders[1:]:
            allocation[aid].discard(mid)
            # Price-aware fill (MemPrice-like): best quality/cost alternative
            # from a different source that is not already held.
            current_sources = {
                bench.memories.get(x).source_class for x in allocation[aid]
            }
            candidates = [
                candidate.id
                for candidate in bench.memories
                if candidate.id not in allocation[aid]
                and (
                    not bench.config.one_per_source
                    or candidate.source_class not in current_sources
                )
                and not candidate.metadata.get("poison")
            ]
            candidates.sort(
                key=lambda cid: (
                    bench.memory_marginal_value(
                        aid, cid, frozenset(allocation[aid]), split=split
                    )
                    / max(0.01, bench.memories.get(cid).production_cost)
                ),
                reverse=True,
            )
            for cid in candidates:
                if len(allocation[aid]) < int(bench._agent(aid).max_memories):  # noqa: SLF001
                    if bench.memory_marginal_value(
                        aid, cid, frozenset(allocation[aid]), split=split
                    ) > 0:
                        allocation[aid].add(cid)
                        break

    return {aid: frozenset(bundle) for aid, bundle in allocation.items()}


# ----------------------------------------------------------------------
# Memory-system proxies
# ----------------------------------------------------------------------


def mem0_like(bench: MemoryBenchmark, split: str = "eval") -> Allocation:
    """Proxy: consolidate to one high-quality memory per (facet, source), then
    retrieve top-k by quality × task mass.  Not an official Mem0 reproduction."""

    representatives: dict[tuple[str, str], MemoryAsset] = {}
    for asset in bench.memories:
        key = (asset.facet, asset.source_class)
        current = representatives.get(key)
        if current is None or asset.quality > current.quality:
            representatives[key] = asset
    mass = _task_mass(bench, split)
    ranked = sorted(
        [asset.id for asset in representatives.values()],
        key=lambda mid: (
            bench.memories.get(mid).quality
            * (0.2 + mass.get(bench.memories.get(mid).facet, 0.0))
        ),
        reverse=True,
    )
    return {
        agent_id: _take_ranked(bench, agent_id, ranked)
        for agent_id in _agent_ids(bench)
    }


def memgpt_like(bench: MemoryBenchmark, split: str = "eval") -> Allocation:
    """Proxy: importance-ranked hierarchical memory (quality × task mass ×
    recency), retrieved per agent.  Not an official MemGPT reproduction."""

    ranked = _rank_by_score(bench, _importance_scores(bench, split))
    return {
        agent_id: _take_ranked(bench, agent_id, ranked)
        for agent_id in _agent_ids(bench)
    }


def raptor_like(bench: MemoryBenchmark, split: str = "eval") -> Allocation:
    """Proxy: cluster by facet, keep the highest-quality representative per
    facet, then retrieve per agent.  Not an official RAPTOR reproduction."""

    representatives: dict[str, MemoryAsset] = {}
    for asset in bench.memories:
        current = representatives.get(asset.facet)
        if current is None or asset.quality > current.quality:
            representatives[asset.facet] = asset
    mass = _task_mass(bench, split)
    ranked = sorted(
        [asset.id for asset in representatives.values()],
        key=lambda mid: (
            bench.memories.get(mid).quality
            * (0.2 + mass.get(bench.memories.get(mid).facet, 0.0))
        ),
        reverse=True,
    )
    return {
        agent_id: _take_ranked(bench, agent_id, ranked)
        for agent_id in _agent_ids(bench)
    }


def graphrag_like(bench: MemoryBenchmark, split: str = "eval") -> Allocation:
    """Proxy: greedy source/facet coverage, then per-agent retrieval.  Not an
    official GraphRAG reproduction."""

    mass = _task_mass(bench, split)
    uncovered = {(asset.facet, asset.source_class) for asset in bench.memories}
    representatives: list[str] = []
    remaining = list(bench.memories.ids())
    while remaining and uncovered:
        best_mid = max(
            remaining,
            key=lambda mid: (
                len(
                    {
                        (bench.memories.get(mid).facet, bench.memories.get(mid).source_class)
                    }
                    & uncovered
                )
                * (0.2 + mass.get(bench.memories.get(mid).facet, 0.0))
                * bench.memories.get(mid).quality
            ),
        )
        representatives.append(best_mid)
        uncovered.discard(
            (bench.memories.get(best_mid).facet, bench.memories.get(best_mid).source_class)
        )
        remaining.remove(best_mid)
    ranked = sorted(
        representatives,
        key=lambda mid: (
            bench.memories.get(mid).quality
            * (0.2 + mass.get(bench.memories.get(mid).facet, 0.0))
        ),
        reverse=True,
    )
    return {
        agent_id: _take_ranked(bench, agent_id, ranked)
        for agent_id in _agent_ids(bench)
    }


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def baseline_allocations(
    bench: MemoryBenchmark,
    *,
    seed: int = 0,
    include_proxies: bool = True,
    include_oracle: bool = True,
) -> dict[str, Allocation]:
    """Return all reproducible baselines for one benchmark instance."""

    allocations: dict[str, Allocation] = {
        "no_memory": no_memory(bench),
        "private": private(bench),
        "shared_all": shared_all(bench),
        "shared_budget": shared_budget(bench),
        "top_k": top_k(bench),
        "selective": selective(bench),
        "central_reported": central_planner(bench, seed=seed, reported=True),
        "central_oracle": central_planner(bench, seed=seed, reported=False),
        "composition": composition(bench),
    }
    if include_proxies:
        allocations.update(
            {
                "mem0_like": mem0_like(bench),
                "memgpt_like": memgpt_like(bench),
                "raptor_like": raptor_like(bench),
                "graphrag_like": graphrag_like(bench),
            }
        )
    if include_oracle:
        _, oracle_allocation = oracle_welfare(bench, solver="exact")
        allocations["oracle"] = {
            aid: frozenset(bundle) for aid, bundle in oracle_allocation.items()
        }
    return allocations


def baseline_metadata() -> dict[str, dict[str, str]]:
    """Human-readable labels for the report."""

    return {
        "no_memory": {"type": "allocation", "official": "yes", "note": "empty bundles"},
        "private": {"type": "allocation", "official": "yes", "note": "agent-owned memories only"},
        "shared_all": {"type": "allocation", "official": "yes", "note": "unconstrained over-sharing lower bound"},
        "shared_budget": {"type": "allocation", "official": "yes", "note": "global quality pool with per-agent budget"},
        "top_k": {"type": "allocation", "official": "yes", "note": "quality × task mass"},
        "selective": {"type": "allocation", "official": "yes", "note": "greedy per-agent marginal value"},
        "central_reported": {"type": "allocation", "official": "yes", "note": "central optimizer with noisy reported values"},
        "central_oracle": {"type": "allocation", "official": "yes", "note": "central optimizer with ground-truth values"},
        "composition": {"type": "composition", "official": "proxy", "note": "RCR + CAMA + GovMem + MemPrice proxy"},
        "mem0_like": {"type": "memory_proxy", "official": "proxy", "note": "consolidation + retrieval proxy"},
        "memgpt_like": {"type": "memory_proxy", "official": "proxy", "note": "importance-ranked hierarchy proxy"},
        "raptor_like": {"type": "memory_proxy", "official": "proxy", "note": "facet-cluster representative proxy"},
        "graphrag_like": {"type": "memory_proxy", "official": "proxy", "note": "source/facet coverage proxy"},
        "oracle": {"type": "upper_bound", "official": "yes", "note": "ground-truth welfare optimum"},
    }
