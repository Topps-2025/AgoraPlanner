"""Memory-asset auctions.

The auction allocates *licenses* to use memory assets.  Memory is non-rival:
several agents may hold the same asset, but

* the asset's production/verification cost is paid once if any agent uses it;
* sharing an asset across agents creates a common-mode externality, priced by
  ``common_mode_lambda * correlation_risk * C(k, 2)`` where ``k`` is the number
  of agents that hold it;
* each agent has a context budget and, by default, may hold at most one asset
  per source class (a partition matroid).

The mechanism is VCG on this welfare function.  It is dominant-strategy
incentive compatible for quasi-linear agents, allocates the welfare-maximizing
feasible bundle profile, and gives each agent non-negative utility.  When the
instance is too large, we fall back to beam search or a greedy allocator and
label the result accordingly; payments are then *approximate* VCG payments.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import BenchmarkConfig
from .memory import MemoryStore
from .valuation import Valuator


@dataclass
class Bid:
    agent_id: str
    bundle: frozenset[str]
    value: float
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "bundle": sorted(self.bundle),
            "value": round(self.value, 6),
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
        }


@dataclass
class AuctionResult:
    allocation: dict[str, frozenset[str]]
    welfare: float
    payments: dict[str, float]
    utilities: dict[str, float]
    prices: dict[str, float]
    admitted: frozenset[str]
    total_cost: float
    common_mode_penalty: float
    solver: str
    revenue: float
    surplus: float
    bids: list[Bid] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation": {
                agent_id: sorted(bundle)
                for agent_id, bundle in sorted(self.allocation.items())
            },
            "welfare": round(self.welfare, 6),
            "payments": {
                agent_id: round(value, 6)
                for agent_id, value in sorted(self.payments.items())
            },
            "utilities": {
                agent_id: round(value, 6)
                for agent_id, value in sorted(self.utilities.items())
            },
            "prices": {
                memory_id: round(value, 6)
                for memory_id, value in sorted(self.prices.items())
            },
            "admitted": sorted(self.admitted),
            "total_cost": round(self.total_cost, 6),
            "common_mode_penalty": round(self.common_mode_penalty, 6),
            "solver": self.solver,
            "revenue": round(self.revenue, 6),
            "surplus": round(self.surplus, 6),
            "metadata": self.metadata,
        }


class MemoryAuction:
    """VCG / beam / greedy allocator for memory licenses."""

    def __init__(
        self,
        memories: MemoryStore,
        agents: list[Any],
        valuator: Valuator,
        config: BenchmarkConfig | None = None,
    ):
        self.memories = memories
        self.agents = list(agents)
        self.valuator = valuator
        self.config = config or BenchmarkConfig()
        self._agent_by_id = {agent.id: agent for agent in self.agents}
        self._feasible_cache: dict[str, list[frozenset[str]]] = {}
        self._value_cache: dict[tuple[str, frozenset[str]], float] = {}
        self._memory_ids = tuple(sorted(self.memories.ids()))

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    def feasible_bundles(self, agent_id: str) -> list[frozenset[str]]:
        if agent_id in self._feasible_cache:
            return self._feasible_cache[agent_id]
        agent = self._agent_by_id[agent_id]
        budget = max(0, int(getattr(agent, "max_memories", 1)))
        bundles: list[frozenset[str]] = [frozenset()]
        for size in range(1, budget + 1):
            for combo in itertools.combinations(self._memory_ids, size):
                if self.config.one_per_source:
                    sources = [
                        self.memories.get(mid).source_class for mid in combo
                    ]
                    if len(set(sources)) != len(sources):
                        continue
                bundles.append(frozenset(combo))
        bundles.sort(key=lambda b: (len(b), tuple(sorted(b))))
        self._feasible_cache[agent_id] = bundles
        return bundles

    def _value(self, agent_id: str, bundle: frozenset[str], split: str = "eval") -> float:
        key = (agent_id, bundle)
        if key not in self._value_cache:
            self._value_cache[key] = float(
                self.valuator.value(agent_id, bundle, split=split)
            )
        return self._value_cache[key]

    def _cost(self, admitted: Iterable[str]) -> float:
        return float(
            sum(self.memories.get(mid).production_cost for mid in set(admitted))
        )

    def _common_mode(self, bundles: dict[str, frozenset[str]]) -> float:
        counts: dict[str, int] = {}
        for bundle in bundles.values():
            for mid in bundle:
                counts[mid] = counts.get(mid, 0) + 1
        penalty = 0.0
        for mid, count in counts.items():
            if count <= 1:
                continue
            asset = self.memories.get(mid)
            penalty += (
                self.config.common_mode_lambda
                * asset.correlation_risk
                * (count * (count - 1) / 2.0)
            )
        return float(penalty)

    def _welfare(self, bundles: dict[str, frozenset[str]], split: str = "eval") -> float:
        value = sum(
            self._value(agent_id, bundle, split) for agent_id, bundle in bundles.items()
        )
        admitted = set().union(*bundles.values()) if bundles else set()
        return float(value - self._cost(admitted) - self._common_mode(bundles))

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------
    def _enumerate_product(
        self, agent_ids: list[str], *, split: str = "eval"
    ) -> tuple[float, dict[str, frozenset[str]]] | None:
        feasible = [self.feasible_bundles(agent_id) for agent_id in agent_ids]
        total = 1
        for bundles in feasible:
            total *= len(bundles)
            if total > self.config.max_combinations:
                return None
        best_welfare = -math.inf
        best: dict[str, frozenset[str]] | None = None
        for combo in itertools.product(*feasible):
            bundles = dict(zip(agent_ids, combo))
            welfare = self._welfare(bundles, split)
            if welfare > best_welfare + 1e-12:
                best_welfare = welfare
                best = bundles
        return (best_welfare, best) if best is not None else None

    def _beam_search(
        self,
        agent_ids: list[str],
        *,
        split: str = "eval",
        width: int | None = None,
    ) -> tuple[float, dict[str, frozenset[str]]]:
        width = width or self.config.beam_width
        feasible = {aid: self.feasible_bundles(aid) for aid in agent_ids}
        max_value = {
            aid: max(self._value(aid, bundle, split) for bundle in feasible[aid])
            for aid in agent_ids
        }
        # State: (processed agents -> bundles, per-memory counts, value sum, cost sum)
        states: list[
            tuple[dict[str, frozenset[str]], dict[str, int], float, float]
        ] = [({}, {}, 0.0, 0.0)]
        for idx, agent_id in enumerate(agent_ids):
            remaining = agent_ids[idx + 1 :]
            optimistic = sum(max_value[aid] for aid in remaining)
            expanded: list[
                tuple[dict[str, frozenset[str]], dict[str, int], float, float]
            ] = []
            for bundles, counts, value_sum, cost_sum in states:
                for bundle in feasible[agent_id]:
                    new_counts = dict(counts)
                    new_cost = cost_sum
                    for mid in bundle:
                        old = new_counts.get(mid, 0)
                        if old == 0:
                            new_cost += self.memories.get(mid).production_cost
                        new_counts[mid] = old + 1
                    new_bundles = dict(bundles)
                    new_bundles[agent_id] = bundle
                    new_value = value_sum + self._value(agent_id, bundle, split)
                    expanded.append(
                        (new_bundles, new_counts, new_value, new_cost)
                    )
            # Rank by current welfare + optimistic completion bound.
            expanded.sort(
                key=lambda s: (
                    s[2]
                    - s[3]
                    - self._common_mode_from_counts(s[1])
                    + optimistic
                ),
                reverse=True,
            )
            states = expanded[:width]
        best = max(
            states,
            key=lambda s: s[2] - s[3] - self._common_mode_from_counts(s[1]),
        )
        bundles, counts, value_sum, cost_sum = best
        penalty = self._common_mode_from_counts(counts)
        return float(value_sum - cost_sum - penalty), bundles

    def _common_mode_from_counts(self, counts: dict[str, int]) -> float:
        penalty = 0.0
        for mid, count in counts.items():
            if count <= 1:
                continue
            asset = self.memories.get(mid)
            penalty += (
                self.config.common_mode_lambda
                * asset.correlation_risk
                * (count * (count - 1) / 2.0)
            )
        return float(penalty)

    def _greedy(
        self, agent_ids: list[str], *, split: str = "eval"
    ) -> tuple[float, dict[str, frozenset[str]]]:
        bundles: dict[str, frozenset[str]] = {aid: frozenset() for aid in agent_ids}
        counts: dict[str, int] = {}
        while True:
            best_gain = 0.0
            best_move: tuple[str, str] | None = None
            for agent_id in agent_ids:
                current = bundles[agent_id]
                for mid in self._memory_ids:
                    if mid in current:
                        continue
                    if len(current) >= self._agent_by_id[agent_id].max_memories:
                        continue
                    if self.config.one_per_source:
                        current_sources = {
                            self.memories.get(x).source_class for x in current
                        }
                        if current_sources and self.memories.get(mid).source_class in current_sources:
                            continue
                    new_bundle = current | {mid}
                    delta_value = self._value(agent_id, new_bundle, split) - self._value(
                        agent_id, current, split
                    )
                    old_count = counts.get(mid, 0)
                    delta_cost = (
                        0.0
                        if old_count > 0
                        else self.memories.get(mid).production_cost
                    )
                    delta_penalty = (
                        self.config.common_mode_lambda
                        * self.memories.get(mid).correlation_risk
                        * old_count
                    )
                    gain = delta_value - delta_cost - delta_penalty
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_move = (agent_id, mid)
            if best_move is None:
                break
            agent_id, mid = best_move
            bundles[agent_id] = bundles[agent_id] | {mid}
            counts[mid] = counts.get(mid, 0) + 1
        welfare = self._welfare(bundles, split)
        return welfare, bundles

    def solve(
        self,
        *,
        split: str = "eval",
        exclude_agent: str | None = None,
        solver: str | None = None,
    ) -> tuple[float, dict[str, frozenset[str]], str]:
        agent_ids = [
            agent.id for agent in self.agents if agent.id != exclude_agent
        ]
        solver = solver or self.config.solver
        if not agent_ids:
            return 0.0, {}, "empty"
        if solver in {"auto", "exact"}:
            enumerated = self._enumerate_product(agent_ids, split=split)
            if enumerated is not None:
                welfare, bundles = enumerated
                return welfare, bundles, "exact"
            if solver == "exact":
                solver = "beam"
        if solver == "beam":
            welfare, bundles = self._beam_search(agent_ids, split=split)
            return welfare, bundles, "beam"
        welfare, bundles = self._greedy(agent_ids, split=split)
        return welfare, bundles, "greedy"

    # ------------------------------------------------------------------
    # VCG
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        split: str = "eval",
        solver: str | None = None,
        collect_bids: bool = True,
    ) -> AuctionResult:
        welfare, allocation, solver = self.solve(split=split, solver=solver)
        payments: dict[str, float] = {}
        utilities: dict[str, float] = {}
        for agent in self.agents:
            agent_id = agent.id
            value_i = self._value(agent_id, allocation.get(agent_id, frozenset()), split)
            welfare_minus_i, _, _ = self.solve(
                split=split, exclude_agent=agent_id, solver=solver
            )
            payment = welfare_minus_i - (welfare - value_i)
            payments[agent_id] = float(payment)
            utilities[agent_id] = float(value_i - payment)

        admitted = set().union(*allocation.values()) if allocation else set()
        total_cost = self._cost(admitted)
        common_mode = self._common_mode(allocation)
        prices = self._compute_prices(allocation, split=split)
        revenue = float(sum(payments.values()))
        return AuctionResult(
            allocation={aid: frozenset(b) for aid, b in allocation.items()},
            welfare=float(welfare),
            payments=payments,
            utilities=utilities,
            prices=prices,
            admitted=frozenset(admitted),
            total_cost=total_cost,
            common_mode_penalty=common_mode,
            solver=solver,
            revenue=revenue,
            surplus=float(revenue - total_cost),
            bids=self.bid_profile(split=split) if collect_bids else [],
            metadata={"split": split, "collect_bids": bool(collect_bids)},
        )

    def _compute_prices(
        self, allocation: dict[str, frozenset[str]], *, split: str = "eval"
    ) -> dict[str, float]:
        counts: dict[str, int] = {}
        for bundle in allocation.values():
            for mid in bundle:
                counts[mid] = counts.get(mid, 0) + 1
        prices: dict[str, float] = {}
        for mid in sorted(counts):
            marginal_total = 0.0
            for agent_id, bundle in allocation.items():
                if mid not in bundle:
                    continue
                marginal_total += self._value(
                    agent_id, bundle, split
                ) - self._value(agent_id, bundle - {mid}, split)
            asset = self.memories.get(mid)
            common_mode_marginal = (
                self.config.common_mode_lambda
                * asset.correlation_risk
                * (counts[mid] - 1)
            )
            price = marginal_total - asset.production_cost - common_mode_marginal
            prices[mid] = float(max(0.0, price))
        return prices

    def bid_profile(self, *, split: str = "eval") -> list[Bid]:
        bids: list[Bid] = []
        for agent in self.agents:
            for bundle in self.feasible_bundles(agent.id):
                value = self._value(agent.id, bundle, split)
                bids.append(
                    Bid(
                        agent_id=agent.id,
                        bundle=bundle,
                        value=float(value),
                        confidence=1.0,
                        evidence={"source": "valuator"},
                    )
                )
        return bids

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def zero_price_memories(self, result: AuctionResult) -> list[str]:
        return [
            mid
            for mid, price in result.prices.items()
            if price <= 1e-9
        ]

    def marginal_values(
        self, result: AuctionResult, *, split: str = "eval"
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for mid in sorted(set().union(*result.allocation.values()) if result.allocation else set()):
            total = 0.0
            for agent_id, bundle in result.allocation.items():
                if mid in bundle:
                    total += self._value(
                        agent_id, bundle, split
                    ) - self._value(agent_id, bundle - {mid}, split)
            values[mid] = float(total)
        return values
