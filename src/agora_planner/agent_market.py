"""Agent-side bidding, central regulation, and consensus metrics (v6 scaffold).

This module is the interface layer for the v6 narrative:

* agents submit bids **with evidence**, not central probe values;
* a regulator checks budgets, consistency, and a random audit sample;
* consensus metrics connect memory allocation to team agreement/error
  correlation.

The implementation is deliberately small: it provides the contracts and
testable primitives; the LLM probe/self-evolution loop is layered on top of a
``Valuator`` (e.g. :class:`agora_planner.valuation.LLMValuator`).
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from .auction import Bid
from .valuation import Valuator


class ValueModel(Protocol):
    """Anything that can price a bundle for an agent."""

    def value(
        self, agent_id: str, bundle: frozenset[str], *, split: str = "eval"
    ) -> float: ...

    def marginal(
        self,
        agent_id: str,
        memory_id: str,
        bundle: frozenset[str],
        *,
        split: str = "eval",
    ) -> float: ...


@dataclass
class BidEvidence:
    """Evidence packet attached to an agent bid."""

    probe_trace: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    rationale: str = ""
    history: list[float] = field(default_factory=list)
    uncertainty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_trace": self.probe_trace,
            "sources": self.sources,
            "rationale": self.rationale,
            "history": self.history,
            "uncertainty": round(float(self.uncertainty), 6),
        }


@dataclass
class AgentBid:
    """A bid plus the evidence the regulator can audit."""

    agent_id: str
    bundle: frozenset[str]
    value: float
    evidence: BidEvidence
    confidence: float = 1.0

    def to_bid(self) -> Bid:
        return Bid(
            agent_id=self.agent_id,
            bundle=self.bundle,
            value=float(self.value),
            confidence=float(self.confidence),
            evidence=self.evidence.to_dict(),
        )


class BidBook:
    """Collects agent bids so the auction can clear on agent-reported values."""

    def __init__(self) -> None:
        self._bids: dict[tuple[str, frozenset[str]], AgentBid] = {}

    def submit(self, bid: AgentBid) -> None:
        self._bids[(bid.agent_id, frozenset(bid.bundle))] = bid

    def get(self, agent_id: str, bundle: frozenset[str]) -> AgentBid | None:
        return self._bids.get((agent_id, frozenset(bundle)))

    def value(self, agent_id: str, bundle: frozenset[str], default: float = 0.0) -> float:
        bid = self.get(agent_id, bundle)
        return float(bid.value) if bid is not None else float(default)

    def all_bids(self) -> list[AgentBid]:
        return list(self._bids.values())

    def by_agent(self, agent_id: str) -> list[AgentBid]:
        return [bid for bid in self._bids.values() if bid.agent_id == agent_id]

    def __len__(self) -> int:
        return len(self._bids)


class BidBookValuator:
    """A ``Valuator`` backed by agent-submitted bids.

    Missing bundles fall back to an optional valuator.  In a real system the
    fallback should be the agent's own prior/estimate, not ground truth.
    """

    def __init__(self, book: BidBook, fallback: Valuator | None = None):
        self.book = book
        self.fallback = fallback

    def value(
        self, agent_id: str, bundle: frozenset[str], *, split: str = "eval"
    ) -> float:
        bid = self.book.get(agent_id, bundle)
        if bid is not None:
            return float(bid.value)
        if self.fallback is not None:
            return float(self.fallback.value(agent_id, bundle, split=split))
        return 0.0

    def marginal(
        self,
        agent_id: str,
        memory_id: str,
        bundle: frozenset[str],
        *,
        split: str = "eval",
    ) -> float:
        return self.value(agent_id, bundle | {memory_id}, split=split) - self.value(
            agent_id, bundle, split=split
        )

    def bundle_values(
        self,
        agent_id: str,
        bundles: list[frozenset[str]],
        *,
        split: str = "eval",
    ) -> dict[frozenset[str], float]:
        return {bundle: self.value(agent_id, bundle, split=split) for bundle in bundles}


@dataclass
class CalibratedValueModel:
    """A lightweight calibrated wrapper around an agent's base value model.

    The base model can be an LLM probe, an additive noisy bidder, or a learned
    value function.  The wrapper fits an affine calibration on observed
    outcomes; this is the scaffold for the v6 "valuation is a learning problem"
    claim.
    """

    base: ValueModel
    slope: float = 1.0
    intercept: float = 0.0
    observations: list[tuple[float, float]] = field(default_factory=list)

    def predict(
        self, agent_id: str, bundle: frozenset[str], *, split: str = "eval"
    ) -> float:
        raw = float(self.base.value(agent_id, bundle, split=split))
        return max(0.0, self.slope * raw + self.intercept)

    def value(
        self, agent_id: str, bundle: frozenset[str], *, split: str = "eval"
    ) -> float:
        return self.predict(agent_id, bundle, split=split)

    def marginal(
        self,
        agent_id: str,
        memory_id: str,
        bundle: frozenset[str],
        *,
        split: str = "eval",
    ) -> float:
        return self.value(agent_id, bundle | {memory_id}, split=split) - self.value(
            agent_id, bundle, split=split
        )

    def update(self, predicted: float, observed: float) -> None:
        """Add one outcome and refit the affine calibration."""

        self.observations.append((float(predicted), float(observed)))
        if len(self.observations) < 2:
            return
        xs = np.asarray([x for x, _ in self.observations], dtype=float)
        ys = np.asarray([y for _, y in self.observations], dtype=float)
        if float(np.std(xs)) < 1e-9:
            self.slope = 1.0
            self.intercept = float(np.mean(ys) - np.mean(xs))
            return
        slope, intercept = np.polyfit(xs, ys, 1)
        self.slope = float(slope)
        self.intercept = float(intercept)

    def evidence(
        self,
        agent_id: str,
        bundle: frozenset[str],
        *,
        rationale: str = "",
        split: str = "eval",
    ) -> BidEvidence:
        raw = float(self.base.value(agent_id, bundle, split=split))
        calibrated = self.predict(agent_id, bundle, split=split)
        return BidEvidence(
            probe_trace=[{"raw": round(raw, 6), "calibrated": round(calibrated, 6)}],
            rationale=rationale,
            history=[round(y, 6) for _, y in self.observations[-10:]],
            uncertainty=max(0.0, abs(calibrated - raw)),
        )


@dataclass
class AuditReport:
    agent_id: str
    bundle: frozenset[str]
    bid: float
    audited_value: float
    inflated_by: float
    flagged: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "bundle": sorted(self.bundle),
            "bid": round(self.bid, 6),
            "audited_value": round(self.audited_value, 6),
            "inflated_by": round(self.inflated_by, 6),
            "flagged": self.flagged,
            "reason": self.reason,
        }


class MarketRegulator:
    """Budget verification, consistency checks, and random audits."""

    def __init__(
        self,
        budgets: dict[str, float] | None = None,
        *,
        tolerance: float = 0.05,
        audit_prob: float = 0.2,
        penalty: float = 1.0,
        seed: int = 0,
    ):
        self.budgets = dict(budgets or {})
        self.tolerance = float(tolerance)
        self.audit_prob = float(audit_prob)
        self.penalty = float(penalty)
        self.rng = random.Random(seed)

    def verify_budget(self, bid: AgentBid) -> tuple[bool, str]:
        budget = self.budgets.get(bid.agent_id)
        if budget is None:
            return True, ""
        if bid.value > budget + 1e-9:
            return False, f"bid {bid.value:.3f} exceeds budget {budget:.3f}"
        return True, ""

    def check_consistency(self, bids: Iterable[AgentBid]) -> list[str]:
        """Check free-disposal/monotonicity within each agent's bid book."""

        violations: list[str] = []
        by_agent: dict[str, list[AgentBid]] = {}
        for bid in bids:
            by_agent.setdefault(bid.agent_id, []).append(bid)
        for agent_id, agent_bids in by_agent.items():
            for left, right in itertools.combinations(agent_bids, 2):
                if left.bundle < right.bundle and left.value > right.value + self.tolerance:
                    violations.append(
                        f"{agent_id}: {sorted(left.bundle)} has value {left.value:.3f} "
                        f"> superset {sorted(right.bundle)} value {right.value:.3f}"
                    )
                elif right.bundle < left.bundle and right.value > left.value + self.tolerance:
                    violations.append(
                        f"{agent_id}: {sorted(right.bundle)} has value {right.value:.3f} "
                        f"> superset {sorted(left.bundle)} value {left.value:.3f}"
                    )
        return violations

    def audit(
        self,
        bid: AgentBid,
        true_value_fn: Callable[[str, frozenset[str]], float] | None = None,
    ) -> AuditReport:
        if true_value_fn is not None:
            audited = float(true_value_fn(bid.agent_id, bid.bundle))
            reason = "re-probe"
        else:
            audited = float(bid.evidence.probe_trace[-1].get("calibrated", bid.value)) if bid.evidence.probe_trace else float(bid.value)
            reason = "evidence"
        inflated = float(bid.value) - audited
        flagged = inflated > self.tolerance
        return AuditReport(
            agent_id=bid.agent_id,
            bundle=bid.bundle,
            bid=float(bid.value),
            audited_value=audited,
            inflated_by=inflated,
            flagged=flagged,
            reason=reason,
        )

    def sample_audits(
        self,
        bids: Iterable[AgentBid],
        true_value_fn: Callable[[str, frozenset[str]], float] | None = None,
    ) -> list[AuditReport]:
        reports: list[AuditReport] = []
        for bid in bids:
            if self.rng.random() <= self.audit_prob:
                reports.append(self.audit(bid, true_value_fn))
        return reports

    def penalty_for(self, report: AuditReport) -> float:
        return self.penalty * max(0.0, report.inflated_by) if report.flagged else 0.0


@dataclass
class ConsensusMetrics:
    agreement: float
    error_correlation: float
    consensus_quality: float
    common_mode_risk: float

    def to_dict(self) -> dict[str, float]:
        return {
            "agreement": round(self.agreement, 6),
            "error_correlation": round(self.error_correlation, 6),
            "consensus_quality": round(self.consensus_quality, 6),
            "common_mode_risk": round(self.common_mode_risk, 6),
        }


def _pairwise_mean(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.mean(values))


def agreement(answers: dict[str, np.ndarray]) -> float:
    """Mean pairwise agreement for binary/real answer vectors in [0,1]."""

    arrays = [np.asarray(v, dtype=float) for v in answers.values()]
    if len(arrays) < 2:
        return 0.0
    scores: list[float] = []
    for left, right in itertools.combinations(arrays, 2):
        denom = max(1.0, float(np.mean(np.abs(left) + np.abs(right))))
        scores.append(1.0 - float(np.mean(np.abs(left - right))) / denom)
    return float(np.clip(_pairwise_mean(scores), 0.0, 1.0))


def error_correlation(errors: np.ndarray) -> float:
    """Mean off-diagonal Pearson correlation of agent error vectors."""

    mat = np.asarray(errors, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 2:
        return 0.0
    corr = np.corrcoef(mat)
    if corr.ndim != 2 or corr.shape[0] < 2:
        return 0.0
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    off = off[np.isfinite(off)]
    return float(np.mean(off)) if off.size else 0.0


def common_mode_risk(shared_counts: dict[str, int], rho: dict[str, float]) -> float:
    return float(
        sum(float(rho.get(mid, 0.0)) * math.comb(max(0, int(k)), 2) for mid, k in shared_counts.items())
    )


def compute_consensus_metrics(
    answers: dict[str, np.ndarray],
    errors: np.ndarray,
    shared_counts: dict[str, int] | None = None,
    rho: dict[str, float] | None = None,
) -> ConsensusMetrics:
    agr = agreement(answers)
    corr = error_correlation(errors)
    cm = common_mode_risk(shared_counts or {}, rho or {})
    quality = float(np.clip(agr * (1.0 - max(0.0, corr)), 0.0, 1.0))
    return ConsensusMetrics(
        agreement=agr,
        error_correlation=corr,
        consensus_quality=quality,
        common_mode_risk=cm,
    )
