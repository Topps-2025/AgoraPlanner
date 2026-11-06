"""Bounded recursive improvement (RI) of the memory-market interaction protocol.

This module implements the v9 RSI specification.  It is deliberately
conservative about what counts as RSI:

* the **object layer** is the memory store and the allocation it induces;
* the **interaction protocol** ``P`` is the update operator: evidence rule,
  payment rule, access rule, verification rule, and memory-update rule;
* the **meta layer** revises ``P`` with a fixed meta-procedure ``M`` and a
  fixed external anchor (held-out ground-truth welfare);
* the **constitutional layer** (team objective, held-out anchor, safety and
  independence constraints, budget, human veto) is not editable.

Therefore the system can claim *bounded RI* (N1-N4), not open-ended RSI (N5).
The module also implements the negative controls required to make the claim
falsifiable: fixed protocol, scalar-only tuning, self-validated meta-loop,
no-anchor evaluator co-evolution, and a closed operator library.

Key design invariants
---------------------
1. A candidate protocol must differ from its parent.  A candidate that only
   changes scalar parameters is labelled ``is_structural=False`` and cannot be
   adopted in structural RI mode.
2. A candidate is accepted only if it improves held-out welfare with a positive
   lower confidence bound, under a matched compute budget, and without
   violating the constitutional constraints.
3. The adopted protocol is versioned, logged, and reused in later rounds; a
   revision that is not reused is not RI.
4. Self-validated and no-anchor loops are negative controls; they must never be
   reported as RSI evidence.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .auction import MemoryAuction
from .benchmark import MemoryBenchmark
from .config import BenchmarkConfig, DEFAULT_RESULTS_DIR, RSIConfig
from .memory import MemoryAsset, MemoryStore
from .metrics import summarize_allocation
from .rsi import MarketGroundedRSI
from .valuation import GroundTruthValuator, NoisyBidder, make_valuator

# ----------------------------------------------------------------------
# Protocol definition
# ----------------------------------------------------------------------

STRUCTURAL_FIELDS: tuple[str, ...] = (
    "evidence_rule",
    "payment_rule",
    "access_rule",
    "verification_rule",
    "pricing_rule",
    "update_rule",
)
PARAMETER_FIELDS: tuple[str, ...] = (
    "common_mode_lambda",
    "max_memories_per_agent",
    "one_per_source",
    "audit_prob",
    "review_quorum",
    "acceptance_delta",
    "validation_seeds",
    "max_inner_rounds",
    "posted_markup",
)

EVIDENCE_RULES = {"truthful", "noisy"}
PAYMENT_RULES = {"vcg", "first_price", "posted_price"}
ACCESS_RULES = {"budget_source", "budget_only", "open"}
VERIFICATION_RULES = {"none", "audit", "audit_review"}
PRICING_RULES = {"none", "common_mode"}
UPDATE_RULES = {"validated", "training_only", "retire_only", "none"}


@dataclass(frozen=True)
class Protocol:
    """A versioned interaction protocol ``P``.

    Structural components determine *what kind* of interaction happens; scalar
    parameters tune its intensity.  The distinction is what separates RI from
    hyperparameter tuning.
    """

    version: int
    name: str
    evidence_rule: str = "truthful"
    payment_rule: str = "vcg"
    access_rule: str = "budget_source"
    verification_rule: str = "none"
    pricing_rule: str = "none"
    update_rule: str = "validated"
    common_mode_lambda: float = 0.35
    max_memories_per_agent: int = 2
    one_per_source: bool = True
    audit_prob: float = 0.0
    review_quorum: int = 1
    acceptance_delta: float = 0.002
    validation_seeds: int = 2
    max_inner_rounds: int = 1
    posted_markup: float = 1.25
    parent_version: int | None = None
    source: str = "human"
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("version must be non-negative")
        if self.evidence_rule not in EVIDENCE_RULES:
            raise ValueError(f"unknown evidence_rule: {self.evidence_rule}")
        if self.payment_rule not in PAYMENT_RULES:
            raise ValueError(f"unknown payment_rule: {self.payment_rule}")
        if self.access_rule not in ACCESS_RULES:
            raise ValueError(f"unknown access_rule: {self.access_rule}")
        if self.verification_rule not in VERIFICATION_RULES:
            raise ValueError(f"unknown verification_rule: {self.verification_rule}")
        if self.pricing_rule not in PRICING_RULES:
            raise ValueError(f"unknown pricing_rule: {self.pricing_rule}")
        if self.update_rule not in UPDATE_RULES:
            raise ValueError(f"unknown update_rule: {self.update_rule}")
        if not (0.0 <= self.audit_prob <= 1.0):
            raise ValueError("audit_prob must be in [0, 1]")
        if self.review_quorum < 1:
            raise ValueError("review_quorum must be >= 1")
        if self.max_memories_per_agent < 1:
            raise ValueError("max_memories_per_agent must be >= 1")
        if self.common_mode_lambda < 0.0:
            raise ValueError("common_mode_lambda must be >= 0")
        if self.acceptance_delta < 0.0:
            raise ValueError("acceptance_delta must be >= 0")
        if self.validation_seeds < 1:
            raise ValueError("validation_seeds must be >= 1")
        if self.max_inner_rounds < 0:
            raise ValueError("max_inner_rounds must be >= 0")
        if self.posted_markup < 1.0:
            raise ValueError("posted_markup must be >= 1")
        if self.verification_rule != "none" and self.audit_prob <= 0.0:
            raise ValueError("audit rule requires audit_prob > 0")
        if self.access_rule == "open" and self.one_per_source:
            raise ValueError("open access requires one_per_source=False")

    def evolve(
        self,
        changes: dict[str, Any],
        *,
        version: int | None = None,
        source: str = "meta_proposal",
        rationale: str = "",
    ) -> "Protocol":
        """Return a successor protocol with ``changes`` applied."""

        data = asdict(self)
        data.update(changes)
        new_version = self.version + 1 if version is None else int(version)
        data["version"] = new_version
        data["parent_version"] = self.version
        data["source"] = source
        data["rationale"] = rationale
        data["name"] = str(changes.get("name", f"{self.name}_v{new_version}"))
        return Protocol(**data)

    def diff(self, other: "Protocol") -> "ProtocolDiff":
        return ProtocolDiff.from_protocols(self, other)

    def compute_budget(self) -> int:
        """A conservative proxy for replay compute, used for matched budgets."""

        return max(1, self.max_inner_rounds) * max(1, self.validation_seeds)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["structural_fields"] = list(STRUCTURAL_FIELDS)
        data["parameter_fields"] = list(PARAMETER_FIELDS)
        return data


@dataclass(frozen=True)
class ProtocolDiff:
    """The difference between a parent protocol and a candidate."""

    base_version: int
    candidate: Protocol
    structural_changes: dict[str, tuple[Any, Any]]
    parameter_changes: dict[str, tuple[Any, Any]]
    rationale: str = ""

    @classmethod
    def from_protocols(cls, base: Protocol, candidate: Protocol) -> "ProtocolDiff":
        base_data = asdict(base)
        cand_data = asdict(candidate)
        structural: dict[str, tuple[Any, Any]] = {}
        parameters: dict[str, tuple[Any, Any]] = {}
        for field_name in STRUCTURAL_FIELDS:
            if base_data[field_name] != cand_data[field_name]:
                structural[field_name] = (
                    base_data[field_name],
                    cand_data[field_name],
                )
        for field_name in PARAMETER_FIELDS:
            if base_data[field_name] != cand_data[field_name]:
                parameters[field_name] = (
                    base_data[field_name],
                    cand_data[field_name],
                )
        return cls(
            base_version=base.version,
            candidate=candidate,
            structural_changes=structural,
            parameter_changes=parameters,
            rationale=candidate.rationale,
        )

    @property
    def is_structural(self) -> bool:
        return bool(self.structural_changes)

    @property
    def is_empty(self) -> bool:
        return not self.structural_changes and not self.parameter_changes

    @property
    def changed_fields(self) -> list[str]:
        return sorted(list(self.structural_changes) + list(self.parameter_changes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_version": self.base_version,
            "candidate_version": self.candidate.version,
            "candidate_name": self.candidate.name,
            "is_structural": self.is_structural,
            "is_empty": self.is_empty,
            "structural_changes": {
                k: [v[0], v[1]] for k, v in self.structural_changes.items()
            },
            "parameter_changes": {
                k: [v[0], v[1]] for k, v in self.parameter_changes.items()
            },
            "changed_fields": self.changed_fields,
            "rationale": self.rationale,
        }


class ProtocolRegistry:
    """Versioned protocols plus an append-only adoption log."""

    def __init__(self, initial: Protocol | None = None):
        self._protocols: dict[int, Protocol] = {}
        self.adoptions: list[dict[str, Any]] = []
        if initial is not None:
            self.register(initial)

    def register(self, protocol: Protocol) -> None:
        existing = self._protocols.get(protocol.version)
        if existing is not None and existing != protocol:
            raise ValueError(
                f"protocol version {protocol.version} already registered with "
                "different content"
            )
        self._protocols[protocol.version] = protocol

    def get(self, version: int) -> Protocol:
        return self._protocols[version]

    def current(self) -> Protocol:
        if not self._protocols:
            raise ValueError("registry is empty")
        return self._protocols[max(self._protocols)]

    def all(self) -> list[Protocol]:
        return [self._protocols[v] for v in sorted(self._protocols)]

    def record_adoption(
        self,
        protocol: Protocol,
        *,
        round_idx: int,
        evidence: dict[str, Any],
    ) -> None:
        self.adoptions.append(
            {
                "round": round_idx,
                "version": protocol.version,
                "parent_version": protocol.parent_version,
                "source": protocol.source,
                "rationale": protocol.rationale,
                "evidence": evidence,
            }
        )

    def lineage(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for protocol in self.all():
            structural: bool | None = None
            if (
                protocol.parent_version is not None
                and protocol.parent_version in self._protocols
            ):
                parent = self._protocols[protocol.parent_version]
                structural = parent.diff(protocol).is_structural
            rows.append(
                {
                    "version": protocol.version,
                    "name": protocol.name,
                    "parent_version": protocol.parent_version,
                    "source": protocol.source,
                    "structural": structural,
                    "rationale": protocol.rationale,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocols": [p.to_dict() for p in self.all()],
            "adoptions": list(self.adoptions),
            "lineage": self.lineage(),
        }


# ----------------------------------------------------------------------
# Transactions and evidence
# ----------------------------------------------------------------------


@dataclass
class Transaction:
    round_idx: int
    protocol_version: int
    seed: int
    split: str
    gt_welfare: float
    task_success: float
    min_utility: float
    error_correlation: float
    memory_count: int
    total_cost: float
    audit_flags: int
    public_pool_size: int
    transactions: int
    common_mode_penalty: float = 0.0
    negative_edit_rate: float = 0.0
    structural_edits: int = 0
    parameter_edits: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data


class TransactionLog:
    """Append-only evidence log for S1 diagnosis."""

    def __init__(self) -> None:
        self.rows: list[Transaction] = []

    def append(self, transaction: Transaction) -> None:
        self.rows.append(transaction)

    def last(self, n: int | None = None) -> list[Transaction]:
        if n is None:
            return list(self.rows)
        return list(self.rows[-n:])

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {"n": 0}
        return {
            "n": len(self.rows),
            "mean_gt_welfare": float(mean(r.gt_welfare for r in self.rows)),
            "mean_task_success": float(mean(r.task_success for r in self.rows)),
            "mean_min_utility": float(mean(r.min_utility for r in self.rows)),
            "mean_error_correlation": float(
                mean(r.error_correlation for r in self.rows)
            ),
            "mean_negative_edit_rate": float(
                mean(r.negative_edit_rate for r in self.rows)
            ),
            "total_transactions": int(sum(r.transactions for r in self.rows)),
        }

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rows]


# ----------------------------------------------------------------------
# Protocol execution
# ----------------------------------------------------------------------


@dataclass
class ProtocolRun:
    protocol: Protocol
    allocation: dict[str, frozenset[str]]
    payments: dict[str, float]
    prices: dict[str, float]
    summary: dict[str, Any]
    audit_flags: int = 0
    audited: int = 0
    public_pool: list[str] = field(default_factory=list)
    transactions: int = 0
    negative_edit_rate: float = 0.0
    update_logs: list[dict[str, Any]] = field(default_factory=list)
    store: MemoryStore | None = None

    def to_dict(self, *, include_store: bool = False) -> dict[str, Any]:
        payload = {
            "protocol": self.protocol.to_dict(),
            "allocation": {
                aid: sorted(bundle) for aid, bundle in sorted(self.allocation.items())
            },
            "payments": {
                aid: round(float(v), 6) for aid, v in sorted(self.payments.items())
            },
            "prices": {
                mid: round(float(v), 6) for mid, v in sorted(self.prices.items())
            },
            "summary": dict(self.summary),
            "audit_flags": self.audit_flags,
            "audited": self.audited,
            "public_pool": list(self.public_pool),
            "transactions": self.transactions,
            "negative_edit_rate": self.negative_edit_rate,
            "update_logs": list(self.update_logs),
        }
        if include_store and self.store is not None:
            payload["store"] = self.store.to_dicts()
        return payload


class ProtocolRunner:
    """Run one protocol on one benchmark instance."""

    def __init__(
        self,
        benchmark: MemoryBenchmark,
        *,
        seed: int = 0,
        valuator_kind: str | None = None,
        evaluation_common_mode_lambda: float | None = None,
    ):
        self.benchmark = benchmark
        self.seed = int(seed)
        self.valuator_kind = valuator_kind
        # The *evaluator's* social cost of correlation is fixed by the
        # constitutional layer.  It is deliberately independent of the
        # protocol's own pricing rule: otherwise a protocol could improve its
        # score by turning off the penalty it is being judged against.
        self.evaluation_common_mode_lambda = (
            float(benchmark.config.common_mode_lambda)
            if evaluation_common_mode_lambda is None
            else float(evaluation_common_mode_lambda)
        )

    # ------------------------------------------------------------------
    # Config / valuator
    # ------------------------------------------------------------------
    @staticmethod
    def _effective_lambda(protocol: Protocol) -> float:
        return (
            float(protocol.common_mode_lambda)
            if protocol.pricing_rule == "common_mode"
            else 0.0
        )

    def _config(self, protocol: Protocol) -> BenchmarkConfig:
        open_access = protocol.access_rule == "open"
        one_per_source = (
            protocol.one_per_source if protocol.access_rule == "budget_source" else False
        )
        max_memories = protocol.max_memories_per_agent
        if open_access:
            # "Open" means no per-source cap and a larger context budget, not
            # an exponential blow-up of the auction.  Cap it at a small number
            # of slots so counterfactual replay stays tractable.
            max_memories = min(
                len(self.benchmark.memories.ids()),
                max(max_memories + 2, 4),
            )
        return replace(
            self.benchmark.config,
            common_mode_lambda=self._effective_lambda(protocol),
            max_memories_per_agent=int(max_memories),
            one_per_source=bool(one_per_source),
            valuation=("ground_truth" if protocol.evidence_rule == "truthful" else "additive_noisy"),
        )

    def _make_valuator(self, protocol: Protocol, config: BenchmarkConfig):
        if self.valuator_kind is not None:
            return make_valuator(
                self.valuator_kind,
                self.benchmark,
                config,
                seed=self.seed,
            )
        if protocol.evidence_rule == "truthful":
            return GroundTruthValuator(self.benchmark)
        return NoisyBidder(self.benchmark, config, seed=self.seed)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(
        self,
        protocol: Protocol,
        *,
        split: str = "eval",
        apply_update: bool = False,
    ) -> ProtocolRun:
        config = self._config(protocol)
        valuator = self._make_valuator(protocol, config)
        # The auction reads each agent's context budget from the agent object,
        # not from the config.  Apply the protocol's access rule explicitly.
        protocol_agents = [
            replace(agent, max_memories=int(config.max_memories_per_agent))
            for agent in self.benchmark.agents
        ]
        auction = MemoryAuction(
            self.benchmark.memories,
            protocol_agents,
            valuator,
            config,
        )
        result = auction.run(split=split, collect_bids=False)
        allocation: dict[str, frozenset[str]] = {
            aid: frozenset(bundle) for aid, bundle in result.allocation.items()
        }
        payments: dict[str, float] = dict(result.payments)
        prices: dict[str, float] = dict(result.prices)

        audit_flags = 0
        audited = 0
        if protocol.verification_rule in {"audit", "audit_review"}:
            allocation, payments, audit_flags, audited = self._audit(
                protocol, allocation, payments, valuator, split=split
            )

        if protocol.review_quorum > 1:
            allocation = self._apply_quorum(
                protocol, allocation, valuator, split=split
            )

        if protocol.payment_rule == "first_price":
            payments = {
                aid: max(0.0, float(valuator.value(aid, bundle, split=split)))
                for aid, bundle in allocation.items()
            }
        elif protocol.payment_rule == "posted_price":
            effective_lambda = self._effective_lambda(protocol)
            prices = {
                asset.id: float(
                    asset.production_cost * protocol.posted_markup
                    + effective_lambda * asset.correlation_risk
                )
                for asset in self.benchmark.memories
            }
            payments = {
                aid: float(sum(prices.get(mid, 0.0) for mid in bundle))
                for aid, bundle in allocation.items()
            }

        summary = summarize_allocation(
            self.benchmark,
            allocation,
            payments=payments,
            prices=prices,
            common_mode_lambda=self.evaluation_common_mode_lambda,
        )
        # The reported welfare is the mechanism's own signal (not the anchor).
        reported_values = sum(
            float(valuator.value(aid, bundle, split=split))
            for aid, bundle in allocation.items()
        )
        summary["reported_value_sum"] = float(reported_values)
        summary["reported_welfare"] = float(
            reported_values
            - float(summary.get("total_cost", 0.0))
            - float(summary.get("common_mode_penalty", 0.0))
        )
        summary["protocol_version"] = protocol.version
        summary["protocol_name"] = protocol.name
        summary["audit_flags"] = int(audit_flags)
        summary["audited"] = int(audited)

        transactions = int(
            sum(len(bundle) for bundle in allocation.values())
        )
        public_pool = self._public_pool(allocation, protocol)
        summary["public_pool_size"] = len(public_pool)
        summary["public_pool_source_diversity"] = self._source_diversity(public_pool)
        summary["transactions"] = transactions

        store: MemoryStore | None = None
        negative_rate = 0.0
        update_logs: list[dict[str, Any]] = []
        if apply_update and protocol.update_rule != "none":
            store, negative_rate, update_logs = self._apply_update(
                protocol, self.benchmark.memories
            )
            summary["negative_edit_rate"] = float(negative_rate)
            summary["post_update_memory_count"] = len(store)

        return ProtocolRun(
            protocol=protocol,
            allocation=allocation,
            payments=payments,
            prices=prices,
            summary=summary,
            audit_flags=int(audit_flags),
            audited=int(audited),
            public_pool=public_pool,
            transactions=transactions,
            negative_edit_rate=float(negative_rate),
            update_logs=update_logs,
            store=store,
        )

    # ------------------------------------------------------------------
    # Verification / access / payment
    # ------------------------------------------------------------------
    def _audit(
        self,
        protocol: Protocol,
        allocation: dict[str, frozenset[str]],
        payments: dict[str, float],
        valuator,
        *,
        split: str,
        tolerance: float = 0.05,
    ) -> tuple[dict[str, frozenset[str]], dict[str, float], int, int]:
        rng = random.Random(self.seed * 10007 + protocol.version * 7919)
        flags = 0
        audited = 0
        new_allocation = dict(allocation)
        new_payments = dict(payments)
        for agent in self.benchmark.agents:
            if rng.random() > protocol.audit_prob:
                continue
            audited += 1
            bundle = allocation.get(agent.id, frozenset())
            if not bundle:
                continue
            reported = float(valuator.value(agent.id, bundle, split=split))
            truth = float(
                self.benchmark.agent_value(agent.id, bundle, split=split)
            )
            if reported - truth > tolerance:
                flags += 1
                new_allocation[agent.id] = frozenset()
                new_payments[agent.id] = 0.0
        return new_allocation, new_payments, flags, audited

    def _apply_quorum(
        self,
        protocol: Protocol,
        allocation: dict[str, frozenset[str]],
        valuator,
        *,
        split: str,
    ) -> dict[str, frozenset[str]]:
        """If a memory is held by fewer than ``review_quorum`` agents, keep only
        the holder with the highest reported marginal value.  This is a
        structural public-admission rule, not just a metric."""

        holders: dict[str, list[str]] = {}
        for agent_id, bundle in allocation.items():
            for mid in bundle:
                holders.setdefault(mid, []).append(agent_id)
        keep: dict[str, set[str]] = {
            aid: set(bundle) for aid, bundle in allocation.items()
        }
        for mid, agents in holders.items():
            if len(agents) < 2 or len(agents) >= protocol.review_quorum:
                continue
            scored: list[tuple[float, str]] = []
            for aid in agents:
                bundle = allocation[aid]
                score = float(
                    valuator.marginal(aid, mid, bundle - {mid}, split=split)
                )
                scored.append((score, aid))
            scored.sort(reverse=True)
            for _, aid in scored[1:]:
                keep[aid].discard(mid)
        return {aid: frozenset(bundle) for aid, bundle in keep.items()}

    def _public_pool(
        self, allocation: dict[str, frozenset[str]], protocol: Protocol
    ) -> list[str]:
        counts: dict[str, int] = {}
        for bundle in allocation.values():
            for mid in bundle:
                counts[mid] = counts.get(mid, 0) + 1
        quorum = max(2, protocol.review_quorum)
        return sorted(mid for mid, count in counts.items() if count >= quorum)

    def _source_diversity(self, memory_ids: Iterable[str]) -> float:
        ids = list(memory_ids)
        if not ids:
            return 0.0
        sources = {
            self.benchmark.memories.get(mid).source_class
            for mid in ids
            if mid in self.benchmark.memories
        }
        return len(sources) / len(ids)

    # ------------------------------------------------------------------
    # Inner memory update (object-level loop)
    # ------------------------------------------------------------------
    def _apply_update(
        self, protocol: Protocol, store: MemoryStore
    ) -> tuple[MemoryStore, float, list[dict[str, Any]]]:
        allowed_ops = ("retire",) if protocol.update_rule == "retire_only" else ()
        rsi_config = RSIConfig(
            rounds=int(protocol.max_inner_rounds),
            validation_seeds=int(protocol.validation_seeds),
            delta=float(protocol.acceptance_delta),
            max_memories=max(len(store) + 4, 8),
            max_proposals=6,
            allowed_ops=allowed_ops,
        )
        rsi = MarketGroundedRSI(
            self.benchmark,
            rsi_config,
            auction_config=replace(
                self.benchmark.config,
                common_mode_lambda=self.evaluation_common_mode_lambda,
            ),
            valuator_kind="ground_truth",
            seed=self.seed,
        )
        if protocol.update_rule == "training_only":
            payload = rsi.run_self_validated(
                store, rounds=int(protocol.max_inner_rounds)
            )
        else:
            payload = rsi.run(store, rounds=int(protocol.max_inner_rounds))
        logs = list(payload.get("rounds", []))
        negative = sum(
            1
            for row in logs
            if row.get("accepted") and float(row.get("mean_gain", 0.0)) < 0.0
        )
        rate = negative / max(1, len(logs))
        new_store = MemoryStore(
            MemoryAsset.from_dict(item)
            for item in payload.get("final_store", [])
        )
        return new_store, float(rate), logs


# ----------------------------------------------------------------------
# Counterfactual replay
# ----------------------------------------------------------------------


@dataclass
class ReplayResult:
    base: Protocol
    candidate: Protocol
    seeds: list[int]
    base_metrics: list[dict[str, Any]]
    candidate_metrics: list[dict[str, Any]]
    mean_delta: float  # composite objective delta
    std_delta: float
    lcb: float
    mean_welfare_delta: float
    mean_min_utility_delta: float
    mean_error_corr_delta: float
    constraint_violations: list[str]
    compute_matched: bool
    accepted: bool
    budget: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def diff(self) -> ProtocolDiff:
        return self.base.diff(self.candidate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_version": self.base.version,
            "candidate_version": self.candidate.version,
            "candidate_name": self.candidate.name,
            "seeds": list(self.seeds),
            "diff": self.diff.to_dict(),
            "base_metrics": self.base_metrics,
            "candidate_metrics": self.candidate_metrics,
            "mean_delta": round(float(self.mean_delta), 6),
            "std_delta": round(float(self.std_delta), 6),
            "lcb": round(float(self.lcb), 6),
            "mean_welfare_delta": round(float(self.mean_welfare_delta), 6),
            "mean_min_utility_delta": round(float(self.mean_min_utility_delta), 6),
            "mean_error_corr_delta": round(float(self.mean_error_corr_delta), 6),
            "constraint_violations": list(self.constraint_violations),
            "compute_matched": bool(self.compute_matched),
            "accepted": bool(self.accepted),
            "budget": dict(self.budget),
            "metadata": dict(self.metadata),
        }


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def counterfactual_replay(
    benchmark_factory: Callable[[int], MemoryBenchmark],
    base: Protocol,
    candidate: Protocol,
    *,
    seeds: Sequence[int],
    split: str = "eval",
    apply_update: bool = False,
    delta: float = 0.002,
    min_utility_floor: float = -0.5,
    max_error_corr: float = 0.35,
    min_utility_tolerance: float = 0.1,
    error_corr_tolerance: float = 0.05,
    audit_floor: float = 0.0,
    welfare_weight: float = 1.0,
    min_utility_weight: float = 0.0,
    error_corr_weight: float = 2.0,
    evaluation_common_mode_lambda: float | None = None,
    n_boot: int = 0,
) -> ReplayResult:
    """Evaluate ``candidate`` against ``base`` on held-out seeds.

    Both protocols are run from fresh, identical benchmarks with the same task
    seed, the same number of inner rounds, and the same number of validation
    seeds.  The comparison is therefore matched in compute and in task
    distribution; the only difference is the protocol.
    """

    if not seeds:
        raise ValueError("counterfactual_replay requires at least one seed")
    diff = base.diff(candidate)
    if diff.is_empty:
        raise ValueError("candidate protocol is identical to the base protocol")
    if candidate.compute_budget() > base.compute_budget():
        compute_matched = False
    else:
        compute_matched = True

    base_rows: list[dict[str, Any]] = []
    cand_rows: list[dict[str, Any]] = []
    for seed in seeds:
        # Both protocols start from the same benchmark and task sample; the
        # benchmark's value cache is protocol-independent, so sharing it is
        # safe and makes matched-budget replay cheaper.
        bench = benchmark_factory(int(seed))
        base_run = ProtocolRunner(
            bench,
            seed=int(seed),
            evaluation_common_mode_lambda=evaluation_common_mode_lambda,
        ).run(base, split=split, apply_update=apply_update)
        cand_run = ProtocolRunner(
            bench,
            seed=int(seed),
            evaluation_common_mode_lambda=evaluation_common_mode_lambda,
        ).run(candidate, split=split, apply_update=apply_update)
        base_rows.append(base_run.summary)
        cand_rows.append(cand_run.summary)

    def objective(row: dict[str, Any]) -> float:
        return float(
            welfare_weight * float(row["gt_welfare"])
            + min_utility_weight * float(row["gt_min_utility"])
            - error_corr_weight * float(row["between_agent_error_corr"])
        )

    welfare_deltas = [
        float(c["gt_welfare"]) - float(b["gt_welfare"])
        for b, c in zip(base_rows, cand_rows)
    ]
    min_utility_deltas = [
        float(c["gt_min_utility"]) - float(b["gt_min_utility"])
        for b, c in zip(base_rows, cand_rows)
    ]
    error_corr_deltas = [
        float(c["between_agent_error_corr"])
        - float(b["between_agent_error_corr"])
        for b, c in zip(base_rows, cand_rows)
    ]
    deltas = [
        objective(c) - objective(b)
        for b, c in zip(base_rows, cand_rows)
    ]
    mean_delta = _mean(deltas)
    std_delta = _std(deltas)
    if len(deltas) > 1:
        lcb = mean_delta - 1.96 * std_delta / math.sqrt(len(deltas))
    else:
        lcb = mean_delta
    mean_welfare_delta = _mean(welfare_deltas)
    mean_min_utility_delta = _mean(min_utility_deltas)
    mean_error_corr_delta = _mean(error_corr_deltas)

    base_min_utility = _mean([float(b["gt_min_utility"]) for b in base_rows])
    cand_min_utility = _mean([float(c["gt_min_utility"]) for c in cand_rows])
    base_error_corr = _mean(
        [float(b["between_agent_error_corr"]) for b in base_rows]
    )
    cand_error_corr = _mean(
        [float(c["between_agent_error_corr"]) for c in cand_rows]
    )

    violations: list[str] = []
    if not compute_matched:
        violations.append("compute_budget_exceeds_base")
    # Absolute floor plus a no-regression check relative to the parent.  A
    # candidate from a bad baseline may improve the worst-off agent without
    # clearing the absolute floor; a candidate that makes it worse fails.
    if (
        cand_min_utility < min_utility_floor
        and cand_min_utility < base_min_utility - min_utility_tolerance
    ):
        violations.append("min_utility_floor")
    if (
        cand_error_corr > max_error_corr
        and cand_error_corr > base_error_corr + error_corr_tolerance
    ):
        violations.append("error_correlation_ceiling")
    if (
        base.verification_rule != "none"
        and candidate.verification_rule == "none"
        and candidate.audit_prob < audit_floor
    ):
        violations.append("audit_removed")
    if (
        base.update_rule == "validated"
        and candidate.update_rule == "training_only"
    ):
        violations.append("validation_removed")

    accepted = bool(
        mean_delta >= delta and lcb > 0.0 and not violations
    )
    return ReplayResult(
        base=base,
        candidate=candidate,
        seeds=[int(s) for s in seeds],
        base_metrics=base_rows,
        candidate_metrics=cand_rows,
        mean_delta=float(mean_delta),
        std_delta=float(std_delta),
        lcb=float(lcb),
        mean_welfare_delta=float(mean_welfare_delta),
        mean_min_utility_delta=float(mean_min_utility_delta),
        mean_error_corr_delta=float(mean_error_corr_delta),
        constraint_violations=violations,
        compute_matched=compute_matched,
        accepted=accepted,
        budget={
            "base_compute": base.compute_budget(),
            "candidate_compute": candidate.compute_budget(),
            "inner_rounds": int(base.max_inner_rounds),
            "validation_seeds": int(base.validation_seeds),
            "replay_seeds": len(seeds),
            "n_boot": n_boot,
        },
    )


# ----------------------------------------------------------------------
# Operator library and meta-controller
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolVariant:
    name: str
    apply: Callable[[Protocol], dict[str, Any]]
    tags: tuple[str, ...]
    source: str  # "human" | "open_source" | "closed"
    rationale: str = ""


@dataclass
class Diagnosis:
    round_idx: int
    tags: dict[str, float]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_idx,
            "tags": {k: round(v, 6) for k, v in self.tags.items()},
            "rationale": self.rationale,
        }


class OperatorLibrary:
    """A pool of protocol edits.

    ``open_source`` variants model reusable protocol components (VCG, audit,
    quorum, source caps).  The closed variant of the library keeps only
    human-authored edits; it is the attribution ablation for E4.
    """

    def __init__(
        self,
        variants: Sequence[ProtocolVariant] | None = None,
        *,
        include_open_source: bool = True,
    ):
        all_variants = list(default_protocol_variants())
        if variants is not None:
            all_variants = list(variants)
        self.include_open_source = bool(include_open_source)
        self.variants = [
            v
            for v in all_variants
            if include_open_source or v.source != "open_source"
        ]

    def candidates(
        self,
        protocol: Protocol,
        diagnosis: Diagnosis,
        *,
        structural_only: bool,
        max_candidates: int = 3,
    ) -> list[Protocol]:
        scored: list[tuple[float, ProtocolVariant, Protocol]] = []
        for variant in self.variants:
            changes = dict(variant.apply(protocol))
            if not changes:
                continue
            changes.setdefault(
                "name", f"{protocol.name}__{variant.name}_v{protocol.version + 1}"
            )
            candidate = protocol.evolve(
                changes,
                version=protocol.version + 1,
                source=variant.source,
                rationale=variant.rationale,
            )
            diff = protocol.diff(candidate)
            if diff.is_empty:
                continue
            if structural_only and not diff.is_structural:
                continue
            if not structural_only and diff.is_structural:
                # Parameter-only mode: the structural edit is not eligible.
                continue
            score = 0.0
            for tag in variant.tags:
                score += float(diagnosis.tags.get(tag, 0.0))
            # Deterministic tie-break: prefer structural edits and lower versions.
            score += 0.01 if diff.is_structural else 0.0
            scored.append((score, variant, candidate))
        scored.sort(key=lambda item: (-item[0], item[2].name))
        return [candidate for _, _, candidate in scored[:max_candidates]]


def default_protocol_variants() -> list[ProtocolVariant]:
    """The default operator library.

    Structural variants change one of the five protocol components; parameter
    variants only change scalar knobs and are used as the tuning control.
    """

    return [
        ProtocolVariant(
            name="add_audit",
            apply=lambda p: {"verification_rule": "audit", "audit_prob": 0.25},
            tags=("audit",),
            source="open_source",
            rationale="add a random audit of reported bundle values",
        ),
        ProtocolVariant(
            name="strengthen_audit",
            apply=lambda p: {
                "verification_rule": "audit",
                "audit_prob": min(0.9, max(0.35, p.audit_prob + 0.2)),
            },
            tags=("audit",),
            source="open_source",
            rationale="raise the audit probability to deter overconfident bids",
        ),
        ProtocolVariant(
            name="add_review_quorum",
            apply=lambda p: {
                "verification_rule": "audit_review",
                "audit_prob": max(0.2, p.audit_prob),
                "review_quorum": max(2, p.review_quorum),
            },
            tags=("quorum", "audit"),
            source="open_source",
            rationale="require a quorum before a memory is treated as public",
        ),
        ProtocolVariant(
            name="raise_review_quorum",
            apply=lambda p: {"review_quorum": min(3, p.review_quorum + 1)},
            tags=("quorum",),
            source="open_source",
            rationale="tighten the public-pool quorum",
        ),
        ProtocolVariant(
            name="switch_to_vcg",
            apply=lambda p: {"payment_rule": "vcg"},
            tags=("payment",),
            source="open_source",
            rationale="replace the current payment rule with VCG",
        ),
        ProtocolVariant(
            name="switch_to_posted_price",
            apply=lambda p: {"payment_rule": "posted_price"},
            tags=("payment",),
            source="open_source",
            rationale="use posted prices instead of VCG payments",
        ),
        ProtocolVariant(
            name="enforce_source_cap",
            apply=lambda p: {"access_rule": "budget_source", "one_per_source": True},
            tags=("access",),
            source="human",
            rationale="enforce the source matroid",
        ),
        ProtocolVariant(
            name="open_access",
            apply=lambda p: {"access_rule": "open", "one_per_source": False},
            tags=("access",),
            source="human",
            rationale="remove the source cap and allow open access",
        ),
        ProtocolVariant(
            name="switch_to_noisy_evidence",
            apply=lambda p: {"evidence_rule": "noisy"},
            tags=("evidence",),
            source="open_source",
            rationale="use noisy agent bids instead of oracle values",
        ),
        ProtocolVariant(
            name="switch_to_truthful_evidence",
            apply=lambda p: {"evidence_rule": "truthful"},
            tags=("evidence",),
            source="human",
            rationale="use truthful probe values",
        ),
        ProtocolVariant(
            name="validated_update",
            apply=lambda p: {"update_rule": "validated"},
            tags=("update",),
            source="open_source",
            rationale="use held-out validated memory updates",
        ),
        ProtocolVariant(
            name="retire_only_update",
            apply=lambda p: {"update_rule": "retire_only"},
            tags=("update",),
            source="human",
            rationale="restrict memory updates to retirement",
        ),
        ProtocolVariant(
            name="raise_common_mode",
            apply=lambda p: {
                "common_mode_lambda": round(min(1.0, p.common_mode_lambda + 0.15), 3)
            },
            tags=("common_mode",),
            source="open_source",
            rationale="raise the price of shared-source correlation",
        ),
        ProtocolVariant(
            name="lower_common_mode",
            apply=lambda p: {
                "common_mode_lambda": round(max(0.0, p.common_mode_lambda - 0.15), 3)
            },
            tags=("common_mode",),
            source="open_source",
            rationale="lower the price of shared-source correlation",
        ),
        ProtocolVariant(
            name="add_common_mode_pricing",
            apply=lambda p: {
                "pricing_rule": "common_mode",
                "common_mode_lambda": max(0.35, p.common_mode_lambda),
            },
            tags=("common_mode",),
            source="open_source",
            rationale="enable correlation pricing in the market objective",
        ),
        ProtocolVariant(
            name="remove_common_mode_pricing",
            apply=lambda p: {"pricing_rule": "none"},
            tags=("common_mode",),
            source="human",
            rationale="disable correlation pricing",
        ),
        ProtocolVariant(
            name="increase_memory_budget",
            apply=lambda p: {
                "max_memories_per_agent": min(4, p.max_memories_per_agent + 1)
            },
            tags=("access",),
            source="open_source",
            rationale="increase the per-agent context budget",
        ),
        ProtocolVariant(
            name="decrease_memory_budget",
            apply=lambda p: {
                "max_memories_per_agent": max(1, p.max_memories_per_agent - 1)
            },
            tags=("access",),
            source="open_source",
            rationale="decrease the per-agent context budget",
        ),
        ProtocolVariant(
            name="more_validation",
            apply=lambda p: {"validation_seeds": min(4, p.validation_seeds + 1)},
            tags=("update",),
            source="open_source",
            rationale="increase the number of held-out validation seeds",
        ),
    ]


class MetaController:
    """S1 diagnose -> S2 propose -> S3 evaluate -> S4 adopt."""

    def __init__(
        self,
        benchmark_factory: Callable[[int], MemoryBenchmark],
        library: OperatorLibrary,
        *,
        replay_seeds: Sequence[int],
        delta: float = 0.002,
        min_utility_floor: float = -0.5,
        max_error_corr: float = 0.35,
        max_candidates: int = 3,
        structural_only: bool = True,
        welfare_weight: float = 1.0,
        min_utility_weight: float = 0.0,
        error_corr_weight: float = 2.0,
        evaluation_common_mode_lambda: float | None = None,
    ):
        self.benchmark_factory = benchmark_factory
        self.library = library
        self.replay_seeds = [int(s) for s in replay_seeds]
        self.delta = float(delta)
        self.min_utility_floor = float(min_utility_floor)
        self.max_error_corr = float(max_error_corr)
        self.max_candidates = int(max_candidates)
        self.structural_only = bool(structural_only)
        self.welfare_weight = float(welfare_weight)
        self.min_utility_weight = float(min_utility_weight)
        self.error_corr_weight = float(error_corr_weight)
        self.evaluation_common_mode_lambda = evaluation_common_mode_lambda

    # ------------------------------------------------------------------
    # S1
    # ------------------------------------------------------------------
    def diagnose(self, log: TransactionLog, round_idx: int) -> Diagnosis:
        recent = log.last(1)
        if not recent:
            return Diagnosis(round_idx=round_idx, tags={}, rationale="no evidence")
        row = recent[0]
        tags: dict[str, float] = {}
        reasons: list[str] = []
        transaction_denominator = max(1, row.transactions)
        audit_rate = row.audit_flags / transaction_denominator
        if audit_rate > 0.02:
            tags["audit"] = min(1.0, audit_rate * 10.0)
            reasons.append(f"audit flags {row.audit_flags}/{row.transactions}")
        if row.common_mode_penalty > 0.5:
            tags["common_mode"] = max(
                tags.get("common_mode", 0.0),
                min(1.0, row.common_mode_penalty),
            )
            reasons.append(
                f"common-mode penalty {row.common_mode_penalty:.3f}"
            )
        if row.error_correlation > 0.05:
            tags["common_mode"] = max(
                tags.get("common_mode", 0.0),
                min(1.0, row.error_correlation * 5.0),
            )
            tags["quorum"] = min(1.0, row.error_correlation * 3.0)
            reasons.append(f"error correlation {row.error_correlation:.3f}")
        if row.min_utility < 0.0:
            tags["payment"] = min(1.0, abs(row.min_utility))
            reasons.append(f"negative min utility {row.min_utility:.3f}")
        if row.negative_edit_rate > 0.0:
            tags["update"] = min(1.0, row.negative_edit_rate * 5.0)
            reasons.append(f"negative edit rate {row.negative_edit_rate:.3f}")
        if row.public_pool_size <= 1:
            tags["quorum"] = max(tags.get("quorum", 0.0), 0.5)
            reasons.append("public pool too small")
        if row.memory_count <= 1:
            tags["access"] = 0.5
            reasons.append("memory count too small")
        if not tags:
            # Default pressure: try a structural improvement.
            tags["access"] = 0.2
            tags["payment"] = 0.2
            reasons.append("no failure signal; explore structural edits")
        return Diagnosis(
            round_idx=round_idx,
            tags=tags,
            rationale="; ".join(reasons),
        )

    # ------------------------------------------------------------------
    # S2 / S3 / S4
    # ------------------------------------------------------------------
    def step(
        self,
        protocol: Protocol,
        log: TransactionLog,
        round_idx: int,
    ) -> tuple[Protocol, Diagnosis, list[ReplayResult], ReplayResult | None]:
        diagnosis = self.diagnose(log, round_idx)
        candidates = self.library.candidates(
            protocol,
            diagnosis,
            structural_only=self.structural_only,
            max_candidates=self.max_candidates,
        )
        results: list[ReplayResult] = []
        accepted: list[ReplayResult] = []
        for candidate in candidates:
            replay = counterfactual_replay(
                self.benchmark_factory,
                protocol,
                candidate,
                seeds=self.replay_seeds,
                apply_update=False,
                delta=self.delta,
                min_utility_floor=self.min_utility_floor,
                max_error_corr=self.max_error_corr,
                welfare_weight=self.welfare_weight,
                min_utility_weight=self.min_utility_weight,
                error_corr_weight=self.error_corr_weight,
                evaluation_common_mode_lambda=self.evaluation_common_mode_lambda,
            )
            results.append(replay)
            if replay.accepted:
                accepted.append(replay)
        if not accepted:
            return protocol, diagnosis, results, None
        accepted.sort(
            key=lambda r: (
                r.mean_delta,
                r.lcb,
                1 if r.diff.is_structural else 0,
            ),
            reverse=True,
        )
        best = accepted[0]
        return best.candidate, diagnosis, results, best

    # ------------------------------------------------------------------
    # Negative controls
    # ------------------------------------------------------------------
    def self_validated_step(
        self,
        protocol: Protocol,
        log: TransactionLog,
        round_idx: int,
        *,
        training_benchmark_factory: Callable[[int], MemoryBenchmark],
        training_seed: int,
    ) -> tuple[Protocol, Diagnosis, list[ReplayResult], ReplayResult | None]:
        """Accept a candidate if it improves *training* welfare.

        This is the collapse control: no held-out anchor, no LCB, no constraint
        gate.  It must not be interpreted as RSI evidence.
        """

        diagnosis = self.diagnose(log, round_idx)
        candidates = self.library.candidates(
            protocol,
            diagnosis,
            structural_only=self.structural_only,
            max_candidates=self.max_candidates,
        )
        best: tuple[float, ReplayResult] | None = None
        results: list[ReplayResult] = []
        for candidate in candidates:
            replay = counterfactual_replay(
                training_benchmark_factory,
                protocol,
                candidate,
                seeds=[training_seed],
                apply_update=False,
                delta=0.0,
                min_utility_floor=-1e9,
                max_error_corr=1.0,
                welfare_weight=self.welfare_weight,
                min_utility_weight=self.min_utility_weight,
                error_corr_weight=self.error_corr_weight,
                evaluation_common_mode_lambda=self.evaluation_common_mode_lambda,
            )
            results.append(replay)
            if replay.mean_delta > 0.0 and (best is None or replay.mean_delta > best[0]):
                best = (replay.mean_delta, replay)
        if best is None:
            return protocol, diagnosis, results, None
        return best[1].candidate, diagnosis, results, best[1]

    def no_anchor_step(
        self,
        protocol: Protocol,
        log: TransactionLog,
        round_idx: int,
    ) -> tuple[Protocol, Diagnosis, list[ReplayResult], ReplayResult | None]:
        """Accept a candidate if its *reported* welfare improves.

        This is the evaluator-coevolution drift control: the mechanism's own
        reported values are used as the acceptance signal.  It is not an
        independent anchor.
        """

        diagnosis = self.diagnose(log, round_idx)
        candidates = self.library.candidates(
            protocol,
            diagnosis,
            structural_only=self.structural_only,
            max_candidates=self.max_candidates,
        )
        best: tuple[float, ReplayResult] | None = None
        results: list[ReplayResult] = []
        for candidate in candidates:
            replay = counterfactual_replay(
                self.benchmark_factory,
                protocol,
                candidate,
                seeds=self.replay_seeds,
                apply_update=False,
                delta=0.0,
                min_utility_floor=-1e9,
                max_error_corr=1.0,
                welfare_weight=self.welfare_weight,
                min_utility_weight=self.min_utility_weight,
                error_corr_weight=self.error_corr_weight,
                evaluation_common_mode_lambda=self.evaluation_common_mode_lambda,
            )
            results.append(replay)
            reported_delta = _mean(
                [
                    float(c["reported_welfare"]) - float(b["reported_welfare"])
                    for b, c in zip(replay.base_metrics, replay.candidate_metrics)
                ]
            )
            if reported_delta > 0.0 and (
                best is None or reported_delta > best[0]
            ):
                best = (reported_delta, replay)
        if best is None:
            return protocol, diagnosis, results, None
        return best[1].candidate, diagnosis, results, best[1]


# ----------------------------------------------------------------------
# Multi-seed bounded-RI experiment
# ----------------------------------------------------------------------


@dataclass
class ExperimentSettings:
    seeds: tuple[int, ...] = (0, 1, 2)
    meta_rounds: int = 3
    replay_seeds: int = 3
    candidates_per_round: int = 3
    delta: float = 0.002
    min_utility_floor: float = -0.5
    max_error_corr: float = 0.35
    welfare_weight: float = 1.0
    min_utility_weight: float = 0.0
    error_corr_weight: float = 2.0
    apply_update: bool = False
    n_agents: int = 3
    n_facets: int = 10
    n_memories: int = 12
    n_train_tasks: int = 32
    n_eval_tasks: int = 32
    max_memories_per_agent: int = 2
    redundancy_fraction: float = 0.30
    poison_fraction: float = 0.10
    common_mode_lambda: float = 0.35
    evaluation_common_mode_lambda: float = 0.35
    common_mode_risk_same_source: float = 0.85
    common_mode_risk_cross_source: float = 0.25
    bid_noise: float = 0.20
    bid_overconfidence: float = 0.35
    beam_width: int = 32
    solver: str = "greedy"
    output_dir: str = str(Path(DEFAULT_RESULTS_DIR) / "protocol_rsi_v9")

    def benchmark_config(self, seed: int) -> BenchmarkConfig:
        return BenchmarkConfig(
            seed=int(seed),
            n_agents=int(self.n_agents),
            n_facets=int(self.n_facets),
            n_memories=int(self.n_memories),
            n_train_tasks=int(self.n_train_tasks),
            n_eval_tasks=int(self.n_eval_tasks),
            max_memories_per_agent=int(self.max_memories_per_agent),
            redundancy_fraction=float(self.redundancy_fraction),
            poison_fraction=float(self.poison_fraction),
            common_mode_lambda=float(self.common_mode_lambda),
            common_mode_risk_same_source=float(self.common_mode_risk_same_source),
            common_mode_risk_cross_source=float(self.common_mode_risk_cross_source),
            bid_noise=float(self.bid_noise),
            bid_overconfidence=float(self.bid_overconfidence),
            beam_width=int(self.beam_width),
            solver=str(self.solver),
        )

    def benchmark_factory(self, seed: int) -> MemoryBenchmark:
        return MemoryBenchmark(self.benchmark_config(seed))

    def objective(self, summary: dict[str, Any]) -> float:
        return float(
            self.welfare_weight * float(summary["gt_welfare"])
            + self.min_utility_weight * float(summary["gt_min_utility"])
            - self.error_corr_weight * float(summary["between_agent_error_corr"])
        )


def initial_protocol(settings: ExperimentSettings) -> Protocol:
    """A deliberately weak baseline: noisy bids, posted prices, no audit, no
    source cap, and a training-only memory update.  The meta-level has something
    real to improve; a truthful VCG protocol would leave no room for structural
    RI on this benchmark."""

    return Protocol(
        version=0,
        name="p0_weak",
        evidence_rule="truthful",
        payment_rule="vcg",
        access_rule="open",
        verification_rule="none",
        pricing_rule="none",
        update_rule="validated",
        common_mode_lambda=0.0,
        max_memories_per_agent=max(3, int(settings.max_memories_per_agent) + 1),
        one_per_source=False,
        audit_prob=0.0,
        review_quorum=1,
        acceptance_delta=0.002,
        validation_seeds=2,
        max_inner_rounds=1,
        posted_markup=1.25,
        source="human",
        rationale="naive market baseline: open access, no source cap, no correlation pricing",
    )


class BoundedRIRunner:
    """Run fixed-P, meta-P, and negative controls over multiple seeds."""

    MODES = (
        "fixed",
        "meta_structural",
        "meta_scalar",
        "random_selection",
        "random_replay",
        "diagnosis_no_replay",
        "self_validated",
        "no_anchor",
        "closed_library",
    )

    def __init__(self, settings: ExperimentSettings | None = None):
        self.settings = settings or ExperimentSettings()

    # ------------------------------------------------------------------
    # One seed
    # ------------------------------------------------------------------
    def run_seed(self, seed: int, mode: str) -> dict[str, Any]:
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        settings = self.settings
        bench_factory = settings.benchmark_factory
        initial = initial_protocol(settings)
        registry = ProtocolRegistry(initial)
        log = TransactionLog()
        protocol = initial
        rounds: list[dict[str, Any]] = []
        all_replays: list[dict[str, Any]] = []

        library = OperatorLibrary(
            include_open_source=(mode != "closed_library")
        )
        controller = MetaController(
            bench_factory,
            library,
            replay_seeds=[
                seed + 100 + 37 * i for i in range(settings.replay_seeds)
            ],
            delta=settings.delta,
            min_utility_floor=settings.min_utility_floor,
            max_error_corr=settings.max_error_corr,
            max_candidates=settings.candidates_per_round,
            structural_only=(mode in {"meta_structural", "closed_library"}),
            welfare_weight=settings.welfare_weight,
            min_utility_weight=settings.min_utility_weight,
            error_corr_weight=settings.error_corr_weight,
            evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
        )

        for round_idx in range(settings.meta_rounds):
            bench = bench_factory(seed)
            run = ProtocolRunner(
                bench,
                seed=seed,
                evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
            ).run(protocol, apply_update=settings.apply_update)
            protocol_diff = (
                registry.get(protocol.parent_version).diff(protocol)
                if protocol.parent_version is not None
                else None
            )
            log.append(
                Transaction(
                    round_idx=round_idx,
                    protocol_version=protocol.version,
                    seed=seed,
                    split="eval",
                    gt_welfare=float(run.summary["gt_welfare"]),
                    task_success=float(run.summary["task_success"]),
                    min_utility=float(run.summary["gt_min_utility"]),
                    error_correlation=float(
                        run.summary["between_agent_error_corr"]
                    ),
                    memory_count=int(run.summary["memory_count"]),
                    total_cost=float(run.summary["total_cost"]),
                    audit_flags=int(run.audit_flags),
                    public_pool_size=int(run.summary["public_pool_size"]),
                    transactions=int(run.transactions),
                    common_mode_penalty=float(
                        run.summary["common_mode_penalty"]
                    ),
                    structural_edits=(
                        len(protocol_diff.structural_changes)
                        if protocol_diff is not None
                        else 0
                    ),
                    parameter_edits=(
                        len(protocol_diff.parameter_changes)
                        if protocol_diff is not None
                        else 0
                    ),
                    negative_edit_rate=float(run.negative_edit_rate),
                )
            )
            round_record: dict[str, Any] = {
                "round": round_idx,
                "protocol_version": protocol.version,
                "evaluated_protocol_version": protocol.version,
                "protocol_name": protocol.name,
                "welfare": float(run.summary["gt_welfare"]),
                "task_success": float(run.summary["task_success"]),
                "min_utility": float(run.summary["gt_min_utility"]),
                "error_correlation": float(
                    run.summary["between_agent_error_corr"]
                ),
                "memory_count": int(run.summary["memory_count"]),
                "audit_flags": int(run.audit_flags),
                "public_pool_size": int(run.summary["public_pool_size"]),
                "transactions": int(run.transactions),
                "negative_edit_rate": float(run.negative_edit_rate),
                "adopted": False,
                "diagnosis": None,
                "replay": None,
            }

            if mode != "fixed":
                if mode in {
                    "random_selection",
                    "random_replay",
                    "diagnosis_no_replay",
                }:
                    # Controls for the value of S1 diagnosis and S3 replay:
                    #  * random_selection: random structural edit, no replay;
                    #  * random_replay: random structural edit + replay;
                    #  * diagnosis_no_replay: diagnosis-ranked edit, no replay;
                    #  * meta_structural: diagnosis + replay.
                    # If meta_structural does not beat these controls, the gain
                    # is attributable to the library, not to RI.
                    diagnosis = controller.diagnose(log, round_idx)
                    candidates = controller.library.candidates(
                        protocol,
                        diagnosis,
                        structural_only=True,
                        max_candidates=settings.candidates_per_round,
                    )
                    round_record["diagnosis"] = diagnosis.to_dict()
                    round_record["replay_count"] = len(candidates)
                    chosen = None
                    if candidates:
                        if mode == "diagnosis_no_replay":
                            chosen = candidates[0]
                        else:
                            rng = random.Random(seed * 10007 + round_idx * 7919)
                            chosen = rng.choice(candidates)
                    if chosen is not None and mode == "random_replay":
                        replay = counterfactual_replay(
                            settings.benchmark_factory,
                            protocol,
                            chosen,
                            seeds=controller.replay_seeds,
                            delta=settings.delta,
                            min_utility_floor=settings.min_utility_floor,
                            max_error_corr=settings.max_error_corr,
                            welfare_weight=settings.welfare_weight,
                            min_utility_weight=settings.min_utility_weight,
                            error_corr_weight=settings.error_corr_weight,
                            evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
                        )
                        round_record["replay"] = replay.to_dict()
                        if not replay.accepted:
                            chosen = None
                    if chosen is not None:
                        registry.register(chosen)
                        registry.record_adoption(
                            chosen,
                            round_idx=round_idx,
                            evidence={"mode": mode},
                        )
                        selected_diff = protocol.diff(chosen)
                        protocol = chosen
                        round_record["adopted"] = True
                        round_record["adopted_protocol_version"] = protocol.version
                        round_record["structural_edits"] = len(
                            selected_diff.structural_changes
                        )
                        round_record["parameter_edits"] = len(
                            selected_diff.parameter_changes
                        )
                        round_record["changed_fields"] = selected_diff.changed_fields
                        if mode != "random_replay":
                            round_record["replay"] = {
                                "mode": mode,
                                "candidate": chosen.name,
                            }
                    rounds.append(round_record)
                    continue
                if mode == "self_validated":
                    new_protocol, diagnosis, replays, best = (
                        controller.self_validated_step(
                            protocol,
                            log,
                            round_idx,
                            training_benchmark_factory=bench_factory,
                            training_seed=seed,
                        )
                    )
                elif mode == "no_anchor":
                    new_protocol, diagnosis, replays, best = (
                        controller.no_anchor_step(protocol, log, round_idx)
                    )
                else:
                    new_protocol, diagnosis, replays, best = controller.step(
                        protocol, log, round_idx
                    )
                round_record["diagnosis"] = diagnosis.to_dict()
                round_record["replay"] = (
                    best.to_dict() if best is not None else None
                )
                round_record["replay_count"] = len(replays)
                if best is not None:
                    registry.register(new_protocol)
                    registry.record_adoption(
                        new_protocol,
                        round_idx=round_idx,
                        evidence=best.to_dict(),
                    )
                    protocol = new_protocol
                    round_record["adopted"] = True
                    round_record["adopted_protocol_version"] = protocol.version
                    round_record["structural_edits"] = len(
                        best.diff.structural_changes
                    )
                    round_record["parameter_edits"] = len(
                        best.diff.parameter_changes
                    )
                    round_record["changed_fields"] = best.diff.changed_fields
                all_replays.extend(r.to_dict() for r in replays)
            rounds.append(round_record)

        # Held-out test seed: the protocol is frozen before this evaluation.
        test_seed = seed + 900
        test_protocol = protocol
        test_bench = bench_factory(test_seed)
        test_run = ProtocolRunner(
            test_bench,
            seed=test_seed,
            evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
        ).run(test_protocol, apply_update=False)
        initial_bench = bench_factory(test_seed)
        initial_run = ProtocolRunner(
            initial_bench,
            seed=test_seed,
            evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
        ).run(initial, apply_update=False)
        # Transfer test: the same protocol on a shifted task distribution.  The
        # protocol was not selected on this distribution.
        shifted_bench = test_bench.clone_with_shifted_tasks(
            test_seed, shift_strength=0.7
        )
        shifted_initial = ProtocolRunner(
            shifted_bench,
            seed=test_seed,
            evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
        ).run(initial, apply_update=False)
        shifted_final = ProtocolRunner(
            shifted_bench,
            seed=test_seed,
            evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
        ).run(test_protocol, apply_update=False)
        shifted_gain = float(
            shifted_final.summary["gt_welfare"]
            - shifted_initial.summary["gt_welfare"]
        )
        shifted_objective_gain = float(
            settings.objective(shifted_final.summary)
            - settings.objective(shifted_initial.summary)
        )
        test_gain = float(
            test_run.summary["gt_welfare"] - initial_run.summary["gt_welfare"]
        )
        initial_objective = settings.objective(initial_run.summary)
        final_objective = settings.objective(test_run.summary)
        objective_gain = float(final_objective - initial_objective)
        total_transactions = int(
            sum(int(r["transactions"]) for r in rounds)
        )
        productivity = (
            float(objective_gain) / total_transactions
            if total_transactions
            else 0.0
        )
        return {
            "seed": int(seed),
            "mode": mode,
            "initial_protocol": initial.to_dict(),
            "final_protocol": test_protocol.to_dict(),
            "rounds": rounds,
            "registry": registry.to_dict(),
            "heldout_test": {
                "seed": test_seed,
                "initial_welfare": float(initial_run.summary["gt_welfare"]),
                "final_welfare": float(test_run.summary["gt_welfare"]),
                "gain": test_gain,
                "initial_min_utility": float(
                    initial_run.summary["gt_min_utility"]
                ),
                "final_min_utility": float(test_run.summary["gt_min_utility"]),
                "initial_error_corr": float(
                    initial_run.summary["between_agent_error_corr"]
                ),
                "final_error_corr": float(
                    test_run.summary["between_agent_error_corr"]
                ),
                "initial_objective": float(initial_objective),
                "final_objective": float(final_objective),
                "objective_gain": objective_gain,
                "ir_preserved": bool(
                    float(test_run.summary["gt_min_utility"]) >= 0.0
                ),
                "error_corr_preserved": bool(
                    float(test_run.summary["between_agent_error_corr"])
                    <= float(initial_run.summary["between_agent_error_corr"])
                    + 0.05
                ),
                "shifted_initial_welfare": float(
                    shifted_initial.summary["gt_welfare"]
                ),
                "shifted_final_welfare": float(
                    shifted_final.summary["gt_welfare"]
                ),
                "shifted_gain": shifted_gain,
                "shifted_objective_gain": shifted_objective_gain,
                "transactions": int(test_run.transactions),
            },
            "total_transactions": total_transactions,
            "improvement_productivity": float(productivity),
            "compounding": {
                "welfare_by_round": [
                    round(float(r["welfare"]), 6) for r in rounds
                ],
                "welfare_deltas": [
                    round(
                        float(rounds[i + 1]["welfare"])
                        - float(rounds[i]["welfare"]),
                        6,
                    )
                    for i in range(len(rounds) - 1)
                ],
                "positive_rounds": sum(
                    1
                    for i in range(len(rounds) - 1)
                    if float(rounds[i + 1]["welfare"])
                    > float(rounds[i]["welfare"])
                ),
                "adoptions": sum(1 for r in rounds if r.get("adopted")),
            },
            "reuse": {
                str(version): sum(
                    1
                    for r in rounds
                    if int(r["evaluated_protocol_version"]) == version
                )
                for version in sorted(
                    {
                        int(r["adopted_protocol_version"])
                        for r in rounds
                        if r.get("adopted")
                    }
                )
            },
            "replays": all_replays,
            "log": log.to_dicts(),
        }

    # ------------------------------------------------------------------
    # All modes / all seeds
    # ------------------------------------------------------------------
    def run(self, modes: Sequence[str] | None = None) -> dict[str, Any]:
        modes = list(modes or self.MODES)
        results: list[dict[str, Any]] = []
        for mode in modes:
            for seed in self.settings.seeds:
                results.append(self.run_seed(int(seed), mode))
        summary = self.aggregate(results)
        return {
            "settings": asdict(self.settings),
            "results": results,
            "summary": summary,
        }

    @staticmethod
    def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
        by_mode: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            by_mode.setdefault(result["mode"], []).append(result)
        summary: dict[str, Any] = {}
        for mode, rows in by_mode.items():
            gains = [float(r["heldout_test"]["gain"]) for r in rows]
            objective_gains = [
                float(r["heldout_test"]["objective_gain"]) for r in rows
            ]
            productivity = [float(r["improvement_productivity"]) for r in rows]
            adopted = [
                sum(1 for rd in r["rounds"] if rd.get("adopted")) for r in rows
            ]
            positive_rounds = [
                float(r.get("compounding", {}).get("positive_rounds", 0))
                for r in rows
            ]
            max_reuse = [
                float(max(r["reuse"].values())) if r.get("reuse") else 0.0
                for r in rows
            ]
            shifted_gains = [
                float(r["heldout_test"].get("shifted_gain", 0.0)) for r in rows
            ]
            ir_preserved = [
                1.0 if r["heldout_test"].get("ir_preserved", False) else 0.0
                for r in rows
            ]
            corr_preserved = [
                1.0
                if r["heldout_test"].get("error_corr_preserved", False)
                else 0.0
                for r in rows
            ]
            summary[mode] = {
                "n_seeds": len(rows),
                "mean_test_gain": _mean(gains),
                "std_test_gain": _std(gains),
                "mean_objective_gain": _mean(objective_gains),
                "std_objective_gain": _std(objective_gains),
                "mean_improvement_productivity": _mean(productivity),
                "std_improvement_productivity": _std(productivity),
                "mean_adoptions": _mean([float(x) for x in adopted]),
                "mean_positive_rounds": _mean(positive_rounds),
                "mean_max_reuse": _mean(max_reuse),
                "mean_shifted_gain": _mean(shifted_gains),
                "ir_preservation_rate": _mean(ir_preserved),
                "error_corr_preservation_rate": _mean(corr_preserved),
                "gains": [round(g, 6) for g in gains],
                "objective_gains": [round(g, 6) for g in objective_gains],
                "shifted_gains": [round(g, 6) for g in shifted_gains],
            }
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, payload: dict[str, Any], output_dir: str | Path | None = None) -> Path:
        out = Path(output_dir or self.settings.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "mode",
                    "seed",
                    "initial_welfare",
                    "final_welfare",
                    "test_gain",
                    "shifted_gain",
                    "improvement_productivity",
                    "adoptions",
                    "total_transactions",
                    "ir_preserved",
                    "error_corr_preserved",
                ]
            )
            for result in payload["results"]:
                writer.writerow(
                    [
                        result["mode"],
                        result["seed"],
                        result["heldout_test"]["initial_welfare"],
                        result["heldout_test"]["final_welfare"],
                        result["heldout_test"]["gain"],
                        result["heldout_test"].get("shifted_gain", 0.0),
                        result["improvement_productivity"],
                        sum(1 for r in result["rounds"] if r.get("adopted")),
                        result["total_transactions"],
                        result["heldout_test"].get("ir_preserved", False),
                        result["heldout_test"].get("error_corr_preserved", False),
                    ]
                )
        (out / "report.md").write_text(
            self.render_report(payload), encoding="utf-8"
        )
        return out

    @staticmethod
    def render_report(payload: dict[str, Any]) -> str:
        summary = payload["summary"]
        lines = [
            "# Protocol-RSI v9 experiment",
            "",
            "All numbers are held-out test results; gains are relative to the",
            "initial protocol on the same test seed. The test seed is not used",
            "for protocol selection. `test_gain` is ground-truth welfare;",
            "`improvement productivity` is the fixed composite objective",
            "J = W - 2 * error_corr, divided by the number of memory transactions.",
            "",
            "| mode | seeds | mean test gain | std | mean improvement productivity | mean adoptions | mean positive rounds | mean max reuse | mean shifted gain | IR preserved |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, row in sorted(summary.items()):
            lines.append(
                f"| {mode} | {row['n_seeds']} | "
                f"{row['mean_test_gain']:.6f} | "
                f"{row['std_test_gain']:.6f} | "
                f"{row['mean_improvement_productivity']:.6f} | "
                f"{row['mean_adoptions']:.2f} | "
                f"{row.get('mean_positive_rounds', 0.0):.2f} | "
                f"{row.get('mean_max_reuse', 0.0):.2f} | "
                f"{row.get('mean_shifted_gain', 0.0):.6f} | "
                f"{row.get('ir_preservation_rate', 0.0):.2f} |"
            )
        lines.extend(
            [
                "",
                "## How to read this",
                "",
                "* `fixed` is the no-RI control: the protocol never changes.",
                "* `meta_structural` is the bounded-RI condition: only structural",
                "  protocol edits are eligible; adoption requires a positive",
                "  lower confidence bound on held-out welfare.",
                "* `meta_scalar` is the tuning control: only scalar edits are",
                "  eligible. If it matches `meta_structural`, the gains are",
                "  hyperparameter tuning, not RI.",
                "* `self_validated` accepts edits that improve the training seed;",
                "  it has no held-out anchor and is a negative control.",
                "* `no_anchor` accepts edits based on the mechanism's own reported",
                "  welfare; it is an evaluator-coevolution drift control.",
                "* `closed_library` repeats structural RI without open-source",
                "  protocol variants; it is the attribution ablation.",
                "",
            ]
        )
        return "\n".join(lines)
