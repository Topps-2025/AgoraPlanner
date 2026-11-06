"""Agent-level bounded RSI for a two-sided memory market.

This module implements the construction discussed on 2026-11-02:

* a two-sided market: buyers bid from demand-based valuations, sellers ask from
  existing-memory opportunity costs;
* self-evolution: each side updates its valuation and quote rules from the
  outcome of a transaction;
* RSI: the **update rule itself** is represented as an editable program of
  operators, and the meta-level *generates* new update programs by structural
  mutations (insert / delete / replace / swap / compose), rather than selecting
  from a fixed list of pre-defined update rules;
* bounded RI: the meta-procedure, the meta-objective, and the held-out
  meta-episode protocol are fixed; only the update program is revised and
  reused.

The design deliberately separates three things that are often conflated:

1. **valuation error** (predicted value vs realized value),
2. **quote regret** (realized price vs a reference price),
3. **learning-rule quality** (how fast/robustly the agent reduces 1 and 2).

Self-evolution improves (1) and (2) under a fixed update rule.  RSI revises the
update rule that governs how (1) and (2) are improved.  The meta-level therefore
does not optimize the same transaction utility as the inner loop; it optimizes a
fixed learning-performance objective on held-out meta-episodes.
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

from .config import DEFAULT_RESULTS_DIR

# ----------------------------------------------------------------------
# Market configuration and state
# ----------------------------------------------------------------------


@dataclass
class MarketConfig:
    seed: int = 0
    n_buyers: int = 3
    n_sellers: int = 3
    n_memories: int = 6
    feature_dim: int = 3
    n_steps: int = 40
    noise: float = 0.05
    price_rule: str = "midpoint"  # midpoint | buyer | seller
    reference_noise: float = 0.05
    shift: float = 0.0  # distribution shift for held-out test markets

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transaction:
    step: int
    buyer_id: str
    seller_id: str
    memory_id: str
    features: np.ndarray
    predicted_value: float
    predicted_cost: float
    bid: float
    ask: float
    traded: bool
    price: float
    realized_value: float
    realized_cost: float
    reference_price: float
    buyer_utility: float
    seller_utility: float
    surplus: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "memory_id": self.memory_id,
            "features": [round(float(x), 6) for x in self.features],
            "predicted_value": round(float(self.predicted_value), 6),
            "predicted_cost": round(float(self.predicted_cost), 6),
            "bid": round(float(self.bid), 6),
            "ask": round(float(self.ask), 6),
            "traded": bool(self.traded),
            "price": round(float(self.price), 6),
            "realized_value": round(float(self.realized_value), 6),
            "realized_cost": round(float(self.realized_cost), 6),
            "reference_price": round(float(self.reference_price), 6),
            "buyer_utility": round(float(self.buyer_utility), 6),
            "seller_utility": round(float(self.seller_utility), 6),
            "surplus": round(float(self.surplus), 6),
        }


@dataclass
class BuyerState:
    id: str
    theta_intercept: float
    theta: np.ndarray
    w: np.ndarray
    bias: float = 0.0
    shading: float = 0.0
    buffer: list[tuple[np.ndarray, float]] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def predict(self, x: np.ndarray) -> float:
        return float(self.w @ x + self.bias)

    def true_value(self, x: np.ndarray) -> float:
        return float(self.theta_intercept + self.theta @ x)

    def bid(self) -> float:
        return max(0.0, self.predict_last - self.shading)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "w": [round(float(v), 6) for v in self.w],
            "bias": round(float(self.bias), 6),
            "shading": round(float(self.shading), 6),
        }


@dataclass
class SellerState:
    id: str
    gamma_intercept: float
    gamma: np.ndarray
    z: np.ndarray
    bias: float = 0.0
    markup: float = 0.0
    buffer: list[tuple[np.ndarray, float]] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def predict(self, x: np.ndarray) -> float:
        return float(self.z @ x + self.bias)

    def true_cost(self, x: np.ndarray) -> float:
        return float(self.gamma_intercept + self.gamma @ x)

    def ask(self) -> float:
        return max(0.0, self.predict_last + self.markup)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "z": [round(float(v), 6) for v in self.z],
            "bias": round(float(self.bias), 6),
            "markup": round(float(self.markup), 6),
        }


def _predict_last(self, x: np.ndarray) -> float:
    return float(self.w @ x + self.bias)


# ``bid`` / ``ask`` need the features of the current memory; the market sets
# ``predict_last`` before calling them.  This is a small convenience attribute,
# not part of the public state.
BuyerState.predict_last = 0.0  # type: ignore[attr-defined]
SellerState.predict_last = 0.0  # type: ignore[attr-defined]


@dataclass
class MemoryItem:
    id: str
    seller_id: str
    features: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seller_id": self.seller_id,
            "features": [round(float(x), 6) for x in self.features],
        }


class TwoSidedMarket:
    """A small two-sided market with private buyer values and seller costs."""

    def __init__(self, config: MarketConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.features = self._make_features()
        self.memories = self._make_memories()
        self.buyers = self._make_buyers()
        self.sellers = self._make_sellers()
        self.step_idx = 0
        self.last_transaction: Transaction | None = None

    def _make_features(self) -> np.ndarray:
        # No constant column: the intercept is a separate bias that the
        # calibration operators are responsible for learning.
        return self.rng.uniform(0.1, 1.0, size=(self.config.n_memories, self.config.feature_dim))

    def _make_memories(self) -> list[MemoryItem]:
        memories: list[MemoryItem] = []
        for idx in range(self.config.n_memories):
            memories.append(
                MemoryItem(
                    id=f"m{idx:02d}",
                    seller_id=f"seller{idx % self.config.n_sellers}",
                    features=self.features[idx],
                )
            )
        return memories

    def _make_buyers(self) -> list[BuyerState]:
        buyers: list[BuyerState] = []
        shift = float(self.config.shift)
        for idx in range(self.config.n_buyers):
            theta_intercept = float(self.rng.uniform(0.6, 1.2)) * (1.0 + shift)
            theta = self.rng.uniform(0.4, 1.2, size=self.config.feature_dim) * (
                1.0 + 0.5 * shift
            )
            buyers.append(
                BuyerState(
                    id=f"buyer{idx}",
                    theta_intercept=theta_intercept,
                    theta=theta,
                    w=np.zeros(self.config.feature_dim),
                    rng=np.random.default_rng(self.config.seed + 1000 + idx),
                )
            )
        return buyers

    def _make_sellers(self) -> list[SellerState]:
        sellers: list[SellerState] = []
        shift = float(self.config.shift)
        for idx in range(self.config.n_sellers):
            gamma_intercept = float(self.rng.uniform(0.1, 0.4)) * (
                1.0 - 0.3 * shift
            )
            gamma = self.rng.uniform(0.1, 0.5, size=self.config.feature_dim) * (
                1.0 - 0.2 * shift
            )
            sellers.append(
                SellerState(
                    id=f"seller{idx}",
                    gamma_intercept=gamma_intercept,
                    gamma=gamma,
                    z=np.zeros(self.config.feature_dim),
                    rng=np.random.default_rng(self.config.seed + 2000 + idx),
                )
            )
        return sellers

    def _sample_triple(self) -> tuple[BuyerState, SellerState, MemoryItem]:
        buyer = self.buyers[int(self.rng.integers(0, len(self.buyers)))]
        memory = self.memories[int(self.rng.integers(0, len(self.memories)))]
        seller = next(s for s in self.sellers if s.id == memory.seller_id)
        return buyer, seller, memory

    def _price(self, bid: float, ask: float) -> float:
        if self.config.price_rule == "buyer":
            return bid
        if self.config.price_rule == "seller":
            return ask
        return 0.5 * (bid + ask)

    def step(self, program: "UpdateProgram") -> Transaction:
        buyer, seller, memory = self._sample_triple()
        x = memory.features
        predicted_value = buyer.predict(x)
        predicted_cost = seller.predict(x)
        buyer.predict_last = predicted_value  # type: ignore[attr-defined]
        seller.predict_last = predicted_cost  # type: ignore[attr-defined]
        bid = buyer.bid()
        ask = seller.ask()
        traded = bid >= ask
        price = self._price(bid, ask) if traded else 0.0

        buyer_noise = float(self.rng.normal(0.0, self.config.noise))
        seller_noise = float(self.rng.normal(0.0, self.config.noise))
        realized_value = buyer.true_value(x) + buyer_noise
        realized_cost = seller.true_cost(x) + seller_noise
        reference_noise = float(self.rng.normal(0.0, self.config.reference_noise))
        reference_price = 0.5 * (realized_value + realized_cost) + reference_noise

        buyer_utility = realized_value - price if traded else 0.0
        seller_utility = price - realized_cost if traded else 0.0
        surplus = realized_value - realized_cost if traded else 0.0
        transaction = Transaction(
            step=self.step_idx,
            buyer_id=buyer.id,
            seller_id=seller.id,
            memory_id=memory.id,
            features=x,
            predicted_value=predicted_value,
            predicted_cost=predicted_cost,
            bid=bid,
            ask=ask,
            traded=traded,
            price=price,
            realized_value=realized_value,
            realized_cost=realized_cost,
            reference_price=reference_price,
            buyer_utility=buyer_utility,
            seller_utility=seller_utility,
            surplus=surplus,
        )
        program.apply(self, buyer, seller, transaction)
        self.last_transaction = transaction
        self.step_idx += 1
        return transaction

    def run(self, program: "UpdateProgram", steps: int | None = None) -> list[Transaction]:
        steps = self.config.n_steps if steps is None else steps
        return [self.step(program) for _ in range(steps)]


# ----------------------------------------------------------------------
# Update programs: the object of RSI
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateOp:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}


@dataclass
class UpdateProgram:
    version: int
    name: str
    ops: list[UpdateOp]
    parent_version: int | None = None
    source: str = "human"
    rationale: str = ""
    lineage: list[str] = field(default_factory=list)

    def apply(
        self,
        market: TwoSidedMarket,
        buyer: BuyerState,
        seller: SellerState,
        tx: Transaction,
    ) -> None:
        for op in self.ops:
            apply_op(op, market, buyer, seller, tx)

    def copy_with(
        self,
        ops: list[UpdateOp],
        *,
        version: int,
        name: str | None = None,
        source: str = "meta_mutation",
        rationale: str = "",
        lineage: list[str] | None = None,
    ) -> "UpdateProgram":
        return UpdateProgram(
            version=version,
            name=name or f"{self.name}_v{version}",
            ops=list(ops),
            parent_version=self.version,
            source=source,
            rationale=rationale,
            lineage=list(self.lineage) + list(lineage or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "parent_version": self.parent_version,
            "source": self.source,
            "rationale": self.rationale,
            "ops": [op.to_dict() for op in self.ops],
            "lineage": list(self.lineage),
        }


@dataclass(frozen=True)
class ProgramDiff:
    base_version: int
    candidate_version: int
    added: tuple[str, ...]
    removed: tuple[str, ...]
    replaced: tuple[str, ...]
    parameters_changed: tuple[str, ...]
    mutation: str

    @property
    def is_structural(self) -> bool:
        return bool(self.added or self.removed or self.replaced)

    @property
    def is_empty(self) -> bool:
        return not self.is_structural and not self.parameters_changed

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_version": self.base_version,
            "candidate_version": self.candidate_version,
            "added": list(self.added),
            "removed": list(self.removed),
            "replaced": list(self.replaced),
            "parameters_changed": list(self.parameters_changed),
            "mutation": self.mutation,
            "is_structural": self.is_structural,
        }


@dataclass(frozen=True)
class MutationRecord:
    kind: str
    position: int
    op_kind: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    structural: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "position": self.position,
            "op_kind": self.op_kind,
            "params": dict(self.params),
            "description": self.description,
            "structural": self.structural,
        }


# ----------------------------------------------------------------------
# Operator implementations
# ----------------------------------------------------------------------


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _update_weights(
    weights: np.ndarray,
    bias: float,
    x: np.ndarray,
    residual: float,
    *,
    rule: str,
    lr: float,
) -> tuple[np.ndarray, float]:
    if rule == "normalized":
        denom = float(x @ x) + 1.0
        return weights + lr * residual * x / denom, bias
    if rule == "sign":
        return weights + lr * math.copysign(1.0, residual) * x, bias
    if rule == "bayes":
        denom = 1.0 + float(x @ x)
        return weights + lr * residual * x / denom, bias + lr * residual / denom
    # sgd
    return weights + lr * residual * x, bias + 0.25 * lr * residual


def apply_op(
    op: UpdateOp,
    market: TwoSidedMarket,
    buyer: BuyerState,
    seller: SellerState,
    tx: Transaction,
) -> None:
    kind = op.kind
    params = op.params
    x = tx.features

    if kind == "update_buyer_valuation":
        predicted = buyer.predict(x)
        residual = tx.realized_value - predicted
        buyer.w, buyer.bias = _update_weights(
            buyer.w,
            buyer.bias,
            x,
            residual,
            rule=str(params.get("rule", "sgd")),
            lr=float(params.get("lr", 0.02)),
        )
        buyer.buffer.append((x.copy(), tx.realized_value))
        if len(buyer.buffer) > 128:
            buyer.buffer.pop(0)
        return

    if kind == "update_seller_valuation":
        predicted = seller.predict(x)
        residual = tx.realized_cost - predicted
        seller.z, seller.bias = _update_weights(
            seller.z,
            seller.bias,
            x,
            residual,
            rule=str(params.get("rule", "sgd")),
            lr=float(params.get("lr", 0.02)),
        )
        seller.buffer.append((x.copy(), tx.realized_cost))
        if len(seller.buffer) > 128:
            seller.buffer.pop(0)
        return

    if kind == "calibrate_buyer":
        if not buyer.buffer:
            return
        window = int(params.get("window", 8))
        recent = buyer.buffer[-window:]
        residuals = [
            y - (buyer.w @ xb + buyer.bias) for xb, y in recent
        ]
        buyer.bias += float(params.get("lr", 0.1)) * float(np.mean(residuals))
        return

    if kind == "calibrate_seller":
        if not seller.buffer:
            return
        window = int(params.get("window", 8))
        recent = seller.buffer[-window:]
        residuals = [
            c - (seller.z @ xb + seller.bias) for xb, c in recent
        ]
        seller.bias += float(params.get("lr", 0.1)) * float(np.mean(residuals))
        return

    if kind == "update_buyer_quote":
        lr = float(params.get("lr", 0.02))
        if tx.traded:
            regret = tx.reference_price - tx.price
            buyer.shading = max(0.0, buyer.shading + lr * math.copysign(1.0, regret))
        else:
            buyer.shading = max(0.0, buyer.shading - lr)
        return

    if kind == "update_seller_quote":
        lr = float(params.get("lr", 0.02))
        if tx.traded:
            regret = tx.price - tx.reference_price
            seller.markup = max(0.0, seller.markup + lr * math.copysign(1.0, regret))
        else:
            seller.markup = max(0.0, seller.markup - lr)
        return

    if kind == "replay_buyer":
        if not buyer.buffer:
            return
        window = int(params.get("window", 8))
        sample = buyer.buffer[-window:]
        lr = float(params.get("lr", 0.01))
        for xb, y in sample:
            residual = y - buyer.predict(xb)
            buyer.w, buyer.bias = _update_weights(
                buyer.w, buyer.bias, xb, residual, rule="sgd", lr=lr
            )
        return

    if kind == "replay_seller":
        if not seller.buffer:
            return
        window = int(params.get("window", 8))
        sample = seller.buffer[-window:]
        lr = float(params.get("lr", 0.01))
        for xb, c in sample:
            residual = c - seller.predict(xb)
            seller.z, seller.bias = _update_weights(
                seller.z, seller.bias, xb, residual, rule="sgd", lr=lr
            )
        return

    if kind == "explore_buyer":
        eps = float(params.get("eps", 0.1))
        if buyer.rng.random() < eps:
            buyer.shading = max(
                0.0, buyer.shading + float(buyer.rng.normal(0.0, 0.02))
            )
        return

    if kind == "explore_seller":
        eps = float(params.get("eps", 0.1))
        if seller.rng.random() < eps:
            seller.markup = max(
                0.0, seller.markup + float(seller.rng.normal(0.0, 0.02))
            )
        return

    if kind == "normalize_buyer":
        buyer.w = np.clip(buyer.w, -5.0, 5.0)
        buyer.bias = _clip(buyer.bias, -5.0, 5.0)
        return

    if kind == "normalize_seller":
        seller.z = np.clip(seller.z, -5.0, 5.0)
        seller.bias = _clip(seller.bias, -5.0, 5.0)
        return

    if kind == "clamp_quotes":
        buyer.shading = max(0.0, buyer.shading)
        seller.markup = max(0.0, seller.markup)
        return

    raise ValueError(f"unknown update op: {kind}")


# ----------------------------------------------------------------------
# Operator templates and mutation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class OpTemplate:
    kind: str
    params: dict[str, Any]
    tags: tuple[str, ...]


def default_op_templates() -> list[OpTemplate]:
    return [
        OpTemplate(
            "update_buyer_valuation",
            {"rule": "sgd", "lr": 0.02},
            ("buyer_valuation",),
        ),
        OpTemplate(
            "update_buyer_valuation",
            {"rule": "normalized", "lr": 0.04},
            ("buyer_valuation", "stability"),
        ),
        OpTemplate(
            "update_buyer_valuation",
            {"rule": "sign", "lr": 0.01},
            ("buyer_valuation", "stability"),
        ),
        OpTemplate(
            "update_seller_valuation",
            {"rule": "sgd", "lr": 0.02},
            ("seller_valuation",),
        ),
        OpTemplate(
            "update_seller_valuation",
            {"rule": "normalized", "lr": 0.04},
            ("seller_valuation", "stability"),
        ),
        OpTemplate(
            "update_seller_valuation",
            {"rule": "sign", "lr": 0.01},
            ("seller_valuation", "stability"),
        ),
        OpTemplate("calibrate_buyer", {"window": 8, "lr": 0.1}, ("calibration", "buyer_valuation")),
        OpTemplate("calibrate_seller", {"window": 8, "lr": 0.1}, ("calibration", "seller_valuation")),
        OpTemplate("update_buyer_quote", {"lr": 0.02}, ("quote", "buyer_valuation")),
        OpTemplate("update_seller_quote", {"lr": 0.02}, ("quote", "seller_valuation")),
        OpTemplate("replay_buyer", {"window": 8, "lr": 0.01}, ("replay", "stability", "buyer_valuation")),
        OpTemplate("replay_seller", {"window": 8, "lr": 0.01}, ("replay", "stability", "seller_valuation")),
        OpTemplate("explore_buyer", {"eps": 0.1}, ("exploration", "buyer_valuation")),
        OpTemplate("explore_seller", {"eps": 0.1}, ("exploration", "seller_valuation")),
        OpTemplate("normalize_buyer", {}, ("stability", "buyer_valuation")),
        OpTemplate("normalize_seller", {}, ("stability", "seller_valuation")),
        OpTemplate("clamp_quotes", {}, ("stability",)),
    ]


class ProgramMutator:
    """Generate new update programs by structural edits.

    There is no fixed list of candidate programs: each candidate is constructed
    by applying an edit to the current program.  The edit operators are the
    "edit language"; the resulting program may be a composition that was never
    pre-enumerated.
    """

    def __init__(self, seed: int = 0, *, allowed_kinds: Sequence[str] | None = None):
        self.rng = random.Random(seed)
        self.templates = default_op_templates()
        if allowed_kinds is not None:
            allowed = set(allowed_kinds)
            self.templates = [t for t in self.templates if t.kind in allowed]

    def _choose(
        self,
        templates: Sequence[OpTemplate],
        diagnosis: "LearningDiagnosis",
    ) -> OpTemplate:
        if not templates:
            raise ValueError("no templates available")
        scores: list[float] = []
        for template in templates:
            score = 1.0
            for tag in template.tags:
                score += float(diagnosis.tags.get(tag, 0.0)) * 2.0
            scores.append(score)
        total = sum(scores)
        pick = self.rng.random() * total
        acc = 0.0
        for template, score in zip(templates, scores):
            acc += score
            if pick <= acc:
                return template
        return templates[-1]

    def _template(self, diagnosis: "LearningDiagnosis") -> OpTemplate:
        return self._choose(self.templates, diagnosis)

    def mutate(
        self,
        program: UpdateProgram,
        diagnosis: "LearningDiagnosis",
        *,
        structural: bool = True,
    ) -> tuple[UpdateProgram, MutationRecord]:
        if not program.ops:
            template = self._template(diagnosis)
            op = UpdateOp(template.kind, dict(template.params))
            record = MutationRecord(
                kind="insert",
                position=0,
                op_kind=op.kind,
                params=dict(op.params),
                description=f"insert {op.kind} at 0",
                structural=True,
            )
            return (
                program.copy_with(
                    [op],
                    version=program.version + 1,
                    rationale=record.description,
                    lineage=[record.description],
                ),
                record,
            )

        ops = list(program.ops)
        if structural:
            eligible = ["insert", "replace", "wrap", "duplicate"]
            if len(ops) > 1:
                eligible += ["delete", "swap"]
            mutation_kind = self.rng.choice(eligible)
        else:
            mutation_kind = "param"

        position = self.rng.randrange(len(ops))

        if mutation_kind == "insert":
            template = self._template(diagnosis)
            op = UpdateOp(template.kind, dict(template.params))
            ops.insert(position, op)
            record = MutationRecord(
                kind="insert",
                position=position,
                op_kind=op.kind,
                params=dict(op.params),
                description=f"insert {op.kind} at {position}",
                structural=True,
            )
        elif mutation_kind == "delete":
            removed = ops.pop(position)
            record = MutationRecord(
                kind="delete",
                position=position,
                op_kind=removed.kind,
                params=dict(removed.params),
                description=f"delete {removed.kind} at {position}",
                structural=True,
            )
        elif mutation_kind == "replace":
            old = ops[position]
            alternatives = [t for t in self.templates if t.kind != old.kind]
            if not alternatives:
                alternatives = self.templates
            template = self.rng.choice(alternatives)
            new_op = UpdateOp(template.kind, dict(template.params))
            ops[position] = new_op
            record = MutationRecord(
                kind="replace",
                position=position,
                op_kind=new_op.kind,
                params=dict(new_op.params),
                description=f"replace {old.kind} with {new_op.kind} at {position}",
                structural=True,
            )
        elif mutation_kind == "swap":
            other = self.rng.randrange(len(ops))
            while other == position:
                other = self.rng.randrange(len(ops))
            ops[position], ops[other] = ops[other], ops[position]
            record = MutationRecord(
                kind="swap",
                position=position,
                op_kind=ops[position].kind,
                params=dict(ops[position].params),
                description=f"swap positions {position} and {other}",
                structural=True,
            )
        elif mutation_kind == "wrap":
            wrap_kinds = {
                "calibrate_buyer",
                "calibrate_seller",
                "replay_buyer",
                "replay_seller",
            }
            wrap_templates = [t for t in self.templates if t.kind in wrap_kinds]
            if not wrap_templates:
                wrap_templates = self.templates
            template = self._choose(wrap_templates, diagnosis)
            op = UpdateOp(template.kind, dict(template.params))
            ops.insert(position, op)
            record = MutationRecord(
                kind="wrap",
                position=position,
                op_kind=op.kind,
                params=dict(op.params),
                description=f"wrap {ops[position + 1].kind if position + 1 < len(ops) else 'end'} with {op.kind}",
                structural=True,
            )
        elif mutation_kind == "duplicate":
            op = ops[position]
            ops.insert(position, UpdateOp(op.kind, dict(op.params)))
            record = MutationRecord(
                kind="duplicate",
                position=position,
                op_kind=op.kind,
                params=dict(op.params),
                description=f"duplicate {op.kind} at {position}",
                structural=True,
            )
        else:
            # Parameter-only mutation.
            op = ops[position]
            params = dict(op.params)
            changed: list[str] = []
            for key, value in list(params.items()):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    factor = self.rng.choice([0.5, 0.75, 1.25, 1.5])
                    new_value = value * factor
                    if isinstance(value, int):
                        new_value = max(1, int(round(new_value)))
                    params[key] = new_value
                    changed.append(f"{key}={new_value}")
            ops[position] = UpdateOp(op.kind, params)
            record = MutationRecord(
                kind="param",
                position=position,
                op_kind=op.kind,
                params=params,
                description=f"tune {op.kind} params: {', '.join(changed) or 'none'}",
                structural=False,
            )

        return (
            program.copy_with(
                ops,
                version=program.version + 1,
                rationale=record.description,
                lineage=[record.description],
            ),
            record,
        )

    def random_mutate(
        self, program: UpdateProgram
    ) -> tuple[UpdateProgram, MutationRecord]:
        """A control mutation that ignores the diagnosis.

        The structural mutation is still generated, but the template and edit
        kind are drawn uniformly.  This isolates the value of S1 diagnosis from
        the value of structural edits per se.
        """

        neutral = LearningDiagnosis(
            round_idx=-1,
            tags={},
            rationale="random control: no diagnosis signal",
        )
        return self.mutate(program, neutral, structural=True)


# ----------------------------------------------------------------------
# Meta-episode metrics
# ----------------------------------------------------------------------


@dataclass
class EpisodeMetrics:
    buyer_mse: float
    seller_mse: float
    buyer_mse_rel: float
    seller_mse_rel: float
    regret: float
    trade_rate: float
    buyer_utility: float
    seller_utility: float
    surplus: float
    instability: float
    sample_efficiency: float
    learning_curve: list[float]
    objective: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "buyer_mse": round(float(self.buyer_mse), 6),
            "seller_mse": round(float(self.seller_mse), 6),
            "buyer_mse_rel": round(float(self.buyer_mse_rel), 6),
            "seller_mse_rel": round(float(self.seller_mse_rel), 6),
            "regret": round(float(self.regret), 6),
            "trade_rate": round(float(self.trade_rate), 6),
            "buyer_utility": round(float(self.buyer_utility), 6),
            "seller_utility": round(float(self.seller_utility), 6),
            "surplus": round(float(self.surplus), 6),
            "instability": round(float(self.instability), 6),
            "sample_efficiency": round(float(self.sample_efficiency), 6),
            "learning_curve": [round(float(v), 6) for v in self.learning_curve],
            "objective": round(float(self.objective), 6),
        }


def episode_objective(
    *,
    buyer_mse: float,
    seller_mse: float,
    regret: float,
    surplus: float,
    instability: float,
) -> float:
    # Fixed meta-objective.  No protocol/agent can change these weights.
    return float(
        -(buyer_mse + seller_mse)
        - 0.5 * regret
        + 0.5 * surplus
        - 0.1 * instability
    )


def run_episode(
    program: UpdateProgram,
    config: MarketConfig,
    *,
    steps: int | None = None,
) -> tuple[EpisodeMetrics, list[Transaction]]:
    market = TwoSidedMarket(config)
    transactions = market.run(program, steps=steps)
    buyer_sq: list[float] = []
    seller_sq: list[float] = []
    regrets: list[float] = []
    traded = 0
    buyer_utility: list[float] = []
    seller_utility: list[float] = []
    surplus: list[float] = []
    learning_curve: list[float] = []
    for tx in transactions:
        b_sq = (tx.realized_value - tx.predicted_value) ** 2
        s_sq = (tx.realized_cost - tx.predicted_cost) ** 2
        buyer_sq.append(b_sq)
        seller_sq.append(s_sq)
        learning_curve.append(0.5 * (b_sq + s_sq))
        if tx.traded:
            traded += 1
            regrets.append(abs(tx.price - tx.reference_price))
            buyer_utility.append(tx.buyer_utility)
            seller_utility.append(tx.seller_utility)
            surplus.append(tx.surplus)
    buyer_mse = float(mean(buyer_sq)) if buyer_sq else 0.0
    seller_mse = float(mean(seller_sq)) if seller_sq else 0.0
    true_values = [tx.realized_value for tx in transactions]
    true_costs = [tx.realized_cost for tx in transactions]
    buyer_scale = float(np.var(true_values)) + 1e-6
    seller_scale = float(np.var(true_costs)) + 1e-6
    buyer_mse_rel = buyer_mse / buyer_scale
    seller_mse_rel = seller_mse / seller_scale
    regret = float(mean(regrets)) if regrets else 0.0
    trade_rate = traded / len(transactions) if transactions else 0.0
    mean_buyer_utility = float(mean(buyer_utility)) if buyer_utility else 0.0
    mean_seller_utility = float(mean(seller_utility)) if seller_utility else 0.0
    mean_surplus = float(mean(surplus)) if surplus else 0.0
    if len(learning_curve) > 1:
        deltas = np.diff(learning_curve)
        instability = float(np.std(deltas))
    else:
        instability = 0.0
    final_error = learning_curve[-1] if learning_curve else 0.0
    initial_error = learning_curve[0] if learning_curve else 0.0
    if initial_error > 1e-9:
        sample_efficiency = float(1.0 - final_error / initial_error)
    else:
        sample_efficiency = 0.0
    objective = episode_objective(
        buyer_mse=buyer_mse_rel,
        seller_mse=seller_mse_rel,
        regret=regret,
        surplus=mean_surplus,
        instability=instability,
    )
    metrics = EpisodeMetrics(
        buyer_mse=buyer_mse,
        seller_mse=seller_mse,
        buyer_mse_rel=buyer_mse_rel,
        seller_mse_rel=seller_mse_rel,
        regret=regret,
        trade_rate=trade_rate,
        buyer_utility=mean_buyer_utility,
        seller_utility=mean_seller_utility,
        surplus=mean_surplus,
        instability=instability,
        sample_efficiency=sample_efficiency,
        learning_curve=learning_curve,
        objective=objective,
    )
    return metrics, transactions


# ----------------------------------------------------------------------
# Diagnosis
# ----------------------------------------------------------------------


@dataclass
class LearningDiagnosis:
    round_idx: int
    tags: dict[str, float]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_idx,
            "tags": {k: round(float(v), 6) for k, v in self.tags.items()},
            "rationale": self.rationale,
        }


def diagnose(metrics: EpisodeMetrics, round_idx: int) -> LearningDiagnosis:
    tags: dict[str, float] = {}
    reasons: list[str] = []
    if metrics.buyer_mse_rel > 0.10:
        tags["buyer_valuation"] = min(5.0, metrics.buyer_mse_rel)
        reasons.append(f"buyer value rel-MSE {metrics.buyer_mse_rel:.3f}")
    if metrics.seller_mse_rel > 0.10:
        tags["seller_valuation"] = min(5.0, metrics.seller_mse_rel)
        reasons.append(f"seller cost rel-MSE {metrics.seller_mse_rel:.3f}")
    if metrics.regret > 0.05:
        tags["quote"] = min(1.0, metrics.regret * 5.0)
        reasons.append(f"quote regret {metrics.regret:.3f}")
    if metrics.trade_rate < 0.5:
        tags["exploration"] = min(1.0, (0.5 - metrics.trade_rate) * 2.0)
        reasons.append(f"low trade rate {metrics.trade_rate:.3f}")
    if metrics.instability > 0.02:
        tags["stability"] = min(1.0, metrics.instability * 20.0)
        reasons.append(f"learning instability {metrics.instability:.3f}")
    if metrics.sample_efficiency < 0.2:
        tags["replay"] = 0.5
        reasons.append(f"low sample efficiency {metrics.sample_efficiency:.3f}")
    if not tags:
        tags["quote"] = 0.2
        tags["calibration"] = 0.2
        reasons.append("no strong failure; explore calibration/quote edits")
    return LearningDiagnosis(
        round_idx=round_idx,
        tags=tags,
        rationale="; ".join(reasons),
    )


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


class ProgramRegistry:
    def __init__(self, initial: UpdateProgram | None = None):
        self._programs: dict[int, UpdateProgram] = {}
        self.adoptions: list[dict[str, Any]] = []
        if initial is not None:
            self.register(initial)

    def register(self, program: UpdateProgram) -> None:
        existing = self._programs.get(program.version)
        if existing is not None and existing != program:
            raise ValueError(
                f"program version {program.version} already registered with "
                "different content"
            )
        self._programs[program.version] = program

    def get(self, version: int) -> UpdateProgram:
        return self._programs[version]

    def current(self) -> UpdateProgram:
        if not self._programs:
            raise ValueError("registry is empty")
        return self._programs[max(self._programs)]

    def all(self) -> list[UpdateProgram]:
        return [self._programs[v] for v in sorted(self._programs)]

    def record_adoption(
        self, program: UpdateProgram, *, round_idx: int, evidence: dict[str, Any]
    ) -> None:
        self.adoptions.append(
            {
                "round": round_idx,
                "version": program.version,
                "parent_version": program.parent_version,
                "rationale": program.rationale,
                "lineage": list(program.lineage),
                "evidence": evidence,
            }
        )

    def lineage(self) -> list[dict[str, Any]]:
        return [
            {
                "version": p.version,
                "name": p.name,
                "parent_version": p.parent_version,
                "source": p.source,
                "ops": [op.kind for op in p.ops],
                "lineage": list(p.lineage),
            }
            for p in self.all()
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "programs": [p.to_dict() for p in self.all()],
            "adoptions": list(self.adoptions),
            "lineage": self.lineage(),
        }


# ----------------------------------------------------------------------
# Meta controller
# ----------------------------------------------------------------------


@dataclass
class EvaluationResult:
    program: UpdateProgram
    seeds: list[int]
    metrics: list[EpisodeMetrics]
    mean_objective: float
    std_objective: float
    lcb: float
    mean_buyer_mse: float
    mean_seller_mse: float
    mean_buyer_mse_rel: float
    mean_seller_mse_rel: float
    mean_regret: float
    mean_trade_rate: float
    mean_surplus: float
    mean_buyer_utility: float
    mean_seller_utility: float
    mean_sample_efficiency: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_version": self.program.version,
            "program_name": self.program.name,
            "program_ops": [op.to_dict() for op in self.program.ops],
            "seeds": list(self.seeds),
            "mean_objective": round(float(self.mean_objective), 6),
            "std_objective": round(float(self.std_objective), 6),
            "lcb": round(float(self.lcb), 6),
            "mean_buyer_mse": round(float(self.mean_buyer_mse), 6),
            "mean_seller_mse": round(float(self.mean_seller_mse), 6),
            "mean_buyer_mse_rel": round(float(self.mean_buyer_mse_rel), 6),
            "mean_seller_mse_rel": round(float(self.mean_seller_mse_rel), 6),
            "mean_regret": round(float(self.mean_regret), 6),
            "mean_trade_rate": round(float(self.mean_trade_rate), 6),
            "mean_surplus": round(float(self.mean_surplus), 6),
            "mean_buyer_utility": round(float(self.mean_buyer_utility), 6),
            "mean_seller_utility": round(float(self.mean_seller_utility), 6),
            "mean_sample_efficiency": round(float(self.mean_sample_efficiency), 6),
            "metadata": dict(self.metadata),
        }


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def evaluate_program(
    program: UpdateProgram,
    *,
    seeds: Sequence[int],
    config: MarketConfig,
    steps: int,
) -> EvaluationResult:
    metrics: list[EpisodeMetrics] = []
    for seed in seeds:
        episode_config = replace(config, seed=int(seed))
        episode_metrics, _ = run_episode(program, episode_config, steps=steps)
        metrics.append(episode_metrics)
    objectives = [m.objective for m in metrics]
    mean_obj = _mean(objectives)
    std_obj = _std(objectives)
    lcb = (
        mean_obj - 1.96 * std_obj / math.sqrt(len(objectives))
        if len(objectives) > 1
        else mean_obj
    )
    return EvaluationResult(
        program=program,
        seeds=[int(s) for s in seeds],
        metrics=metrics,
        mean_objective=mean_obj,
        std_objective=std_obj,
        lcb=lcb,
        mean_buyer_mse=_mean([m.buyer_mse for m in metrics]),
        mean_seller_mse=_mean([m.seller_mse for m in metrics]),
        mean_buyer_mse_rel=_mean([m.buyer_mse_rel for m in metrics]),
        mean_seller_mse_rel=_mean([m.seller_mse_rel for m in metrics]),
        mean_regret=_mean([m.regret for m in metrics]),
        mean_trade_rate=_mean([m.trade_rate for m in metrics]),
        mean_surplus=_mean([m.surplus for m in metrics]),
        mean_buyer_utility=_mean([m.buyer_utility for m in metrics]),
        mean_seller_utility=_mean([m.seller_utility for m in metrics]),
        mean_sample_efficiency=_mean([m.sample_efficiency for m in metrics]),
    )


@dataclass
class MutationResult:
    mutation: MutationRecord
    candidate: UpdateProgram
    evaluation: EvaluationResult
    delta: float
    accepted: bool
    constraint_violations: list[str]
    gain_lcb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation": self.mutation.to_dict(),
            "candidate": self.candidate.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "delta": round(float(self.delta), 6),
            "gain_lcb": round(float(self.gain_lcb), 6),
            "accepted": bool(self.accepted),
            "constraint_violations": list(self.constraint_violations),
        }


class LearningRSIController:
    """S1 diagnose -> S2 mutate (generate) -> S3 meta-replay -> S4 adopt."""

    def __init__(
        self,
        config: MarketConfig,
        *,
        meta_seeds: Sequence[int],
        meta_steps: int,
        candidates: int = 4,
        delta: float = 0.001,
        max_buyer_mse: float = 5.0,
        max_seller_mse: float = 5.0,
        min_trade_rate: float = 0.0,
        mutator_seed: int = 0,
        allowed_kinds: Sequence[str] | None = None,
        structural_only: bool = True,
    ):
        self.config = config
        self.meta_seeds = [int(s) for s in meta_seeds]
        self.meta_steps = int(meta_steps)
        self.candidates = int(candidates)
        self.delta = float(delta)
        self.max_buyer_mse = float(max_buyer_mse)
        self.max_seller_mse = float(max_seller_mse)
        self.min_trade_rate = float(min_trade_rate)
        self.mutator = ProgramMutator(mutator_seed, allowed_kinds=allowed_kinds)
        self.structural_only = bool(structural_only)

    def step(
        self,
        program: UpdateProgram,
        round_idx: int,
    ) -> tuple[UpdateProgram, LearningDiagnosis, list[MutationResult], MutationResult | None]:
        base_eval = evaluate_program(
            program,
            seeds=self.meta_seeds,
            config=self.config,
            steps=self.meta_steps,
        )
        diagnosis = diagnose(base_eval.metrics[0], round_idx)
        results: list[MutationResult] = []
        accepted: list[MutationResult] = []
        for _ in range(self.candidates):
            candidate, mutation = self.mutator.mutate(
                program, diagnosis, structural=self.structural_only
            )
            evaluation = evaluate_program(
                candidate,
                seeds=self.meta_seeds,
                config=self.config,
                steps=self.meta_steps,
            )
            gains = [
                c.objective - b.objective
                for c, b in zip(evaluation.metrics, base_eval.metrics)
            ]
            delta = _mean(gains)
            gain_std = _std(gains)
            gain_lcb = (
                delta - 1.96 * gain_std / math.sqrt(len(gains))
                if len(gains) > 1
                else delta
            )
            violations: list[str] = []
            if evaluation.mean_buyer_mse > self.max_buyer_mse:
                violations.append("buyer_mse_ceiling")
            if evaluation.mean_seller_mse > self.max_seller_mse:
                violations.append("seller_mse_ceiling")
            if evaluation.mean_trade_rate < self.min_trade_rate:
                violations.append("trade_rate_floor")
            accepted_flag = bool(gain_lcb >= self.delta and not violations)
            result = MutationResult(
                mutation=mutation,
                candidate=candidate,
                evaluation=evaluation,
                delta=delta,
                accepted=accepted_flag,
                constraint_violations=violations,
                gain_lcb=gain_lcb,
            )
            results.append(result)
            if accepted_flag:
                accepted.append(result)
        if not accepted:
            return program, diagnosis, results, None
        accepted.sort(key=lambda r: (r.gain_lcb, r.delta), reverse=True)
        best = accepted[0]
        return best.candidate, diagnosis, results, best

    def self_validated_step(
        self,
        program: UpdateProgram,
        round_idx: int,
        *,
        training_seed: int,
    ) -> tuple[UpdateProgram, LearningDiagnosis, list[MutationResult], MutationResult | None]:
        base_eval = evaluate_program(
            program,
            seeds=[training_seed],
            config=self.config,
            steps=self.meta_steps,
        )
        diagnosis = diagnose(base_eval.metrics[0], round_idx)
        base_utility = base_eval.mean_buyer_utility + base_eval.mean_seller_utility
        results: list[MutationResult] = []
        best: MutationResult | None = None
        best_utility_delta: float | None = None
        for _ in range(self.candidates):
            candidate, mutation = self.mutator.mutate(
                program, diagnosis, structural=self.structural_only
            )
            evaluation = evaluate_program(
                candidate,
                seeds=[training_seed],
                config=self.config,
                steps=self.meta_steps,
            )
            # The self-validated control accepts on the utility of the single
            # training market, not on the held-out meta-objective.  This is the
            # "no independent anchor" ablation: it can learn to exploit that
            # market's price noise and may fail under shift.
            utility_delta = (
                evaluation.mean_buyer_utility
                + evaluation.mean_seller_utility
                - base_utility
            )
            objective_delta = evaluation.mean_objective - base_eval.mean_objective
            result = MutationResult(
                mutation=mutation,
                candidate=candidate,
                evaluation=evaluation,
                delta=objective_delta,
                accepted=utility_delta > 0.0,
                constraint_violations=[],
                gain_lcb=utility_delta,
            )
            results.append(result)
            if utility_delta > 0.0 and (
                best_utility_delta is None or utility_delta > best_utility_delta
            ):
                best = result
                best_utility_delta = utility_delta
        if best is None:
            return program, diagnosis, results, None
        return best.candidate, diagnosis, results, best


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


@dataclass
class LearningRSISettings:
    seeds: tuple[int, ...] = (0, 1, 2)
    meta_rounds: int = 3
    meta_steps: int = 40
    meta_seeds: int = 3
    candidates: int = 4
    delta: float = 0.001
    max_buyer_mse: float = 5.0
    max_seller_mse: float = 5.0
    min_trade_rate: float = 0.0
    test_steps: int = 40
    test_seeds: int = 2
    test_shift: float = 0.5
    n_buyers: int = 3
    n_sellers: int = 3
    n_memories: int = 6
    feature_dim: int = 3
    output_dir: str = str(Path(DEFAULT_RESULTS_DIR) / "bilateral_rsi_v2")

    def market_config(self, seed: int) -> MarketConfig:
        return MarketConfig(
            seed=int(seed),
            n_buyers=int(self.n_buyers),
            n_sellers=int(self.n_sellers),
            n_memories=int(self.n_memories),
            feature_dim=int(self.feature_dim),
            n_steps=int(self.meta_steps),
        )

    def benchmark_factory(self, seed: int) -> MarketConfig:
        return self.market_config(seed)


def seed_program() -> UpdateProgram:
    """A deliberately incomplete update program.

    It updates only the buyer's value weights and nothing else: no seller-side
    cost learning, no calibration of the intercept, no quote update, no replay,
    and no exploration.  Structural mutations can add these components; this is
    what makes the RSI object genuinely editable rather than a fixed choice
    among complete programs.
    """

    return UpdateProgram(
        version=0,
        name="seed_buyer_value_only",
        ops=[
            UpdateOp(
                "update_buyer_valuation",
                {"rule": "sgd", "lr": 0.02},
            )
        ],
        parent_version=None,
        source="human",
        rationale="seed program: only buyer value weights; seller side and calibration are missing",
        lineage=[],
    )


class AgentRSIRunner:
    MODES = (
        "fixed",
        "meta_structural",
        "meta_param",
        "random_mutation",
        "self_validated",
        "closed_library",
    )

    def __init__(self, settings: LearningRSISettings | None = None):
        self.settings = settings or LearningRSISettings()

    def _controller(
        self,
        *,
        mode: str,
        seed: int,
        structural_only: bool,
        allowed_kinds: Sequence[str] | None,
    ) -> LearningRSIController:
        return LearningRSIController(
            self.settings.market_config(seed),
            meta_seeds=[seed + 100 + 37 * i for i in range(self.settings.meta_seeds)],
            meta_steps=self.settings.meta_steps,
            candidates=self.settings.candidates,
            delta=self.settings.delta,
            max_buyer_mse=self.settings.max_buyer_mse,
            max_seller_mse=self.settings.max_seller_mse,
            min_trade_rate=self.settings.min_trade_rate,
            mutator_seed=seed * 10007 + 17,
            allowed_kinds=allowed_kinds,
            structural_only=structural_only,
        )

    def run_seed(self, seed: int, mode: str) -> dict[str, Any]:
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        settings = self.settings
        program = seed_program()
        registry = ProgramRegistry(program)
        rounds: list[dict[str, Any]] = []
        controller: LearningRSIController | None = None
        # The closed-library control is deliberately not an open RSI language:
        # it can only edit the buyer side, so it cannot generate the seller-side
        # / calibration composition that the open language can.
        open_language = mode != "closed_library"
        if mode != "fixed":
            structural_only = mode in {"meta_structural", "closed_library", "self_validated", "random_mutation"}
            allowed_kinds: Sequence[str] | None = None
            if mode == "closed_library":
                # A deliberately small, single-sided edit language: the closed
                # library can improve the buyer's value model but cannot add
                # seller-side cost learning, calibration, replay, or
                # exploration.  This isolates the value of the open,
                # composable operator language.
                allowed_kinds = (
                    "update_buyer_valuation",
                    "update_buyer_quote",
                    "normalize_buyer",
                    "clamp_quotes",
                )
            controller = self._controller(
                mode=mode,
                seed=seed,
                structural_only=structural_only,
                allowed_kinds=allowed_kinds,
            )

        for round_idx in range(settings.meta_rounds):
            base_metrics, _ = run_episode(
                program,
                settings.market_config(seed),
                steps=settings.meta_steps,
            )
            # A fixed held-out meta panel, disjoint from both the selection
            # panel and the final test panel.  It measures whether each adopted
            # update program actually makes the *learner* better, rather than
            # only improving the single training market.
            round_eval_seeds = [
                seed + 500 + 11 * i for i in range(max(2, settings.meta_seeds))
            ]
            round_eval = evaluate_program(
                program,
                seeds=round_eval_seeds,
                config=settings.market_config(seed),
                steps=settings.meta_steps,
            )
            round_record: dict[str, Any] = {
                "round": round_idx,
                "program_version": program.version,
                "evaluated_program_version": program.version,
                "program_name": program.name,
                "ops": [op.kind for op in program.ops],
                "base_objective": float(base_metrics.objective),
                "base_buyer_mse": float(base_metrics.buyer_mse),
                "base_seller_mse": float(base_metrics.seller_mse),
                "base_regret": float(base_metrics.regret),
                "base_trade_rate": float(base_metrics.trade_rate),
                "round_eval_seeds": round_eval_seeds,
                "round_heldout_objective": float(round_eval.mean_objective),
                "round_heldout_buyer_mse": float(round_eval.mean_buyer_mse),
                "round_heldout_seller_mse": float(round_eval.mean_seller_mse),
                "round_heldout_sample_efficiency": float(
                    round_eval.mean_sample_efficiency
                ),
                "adopted": False,
                "diagnosis": None,
                "mutation": None,
                "candidate_results": [],
            }
            if controller is not None and mode != "fixed":
                if mode == "self_validated":
                    new_program, diagnosis, results, best = controller.self_validated_step(
                        program, round_idx, training_seed=seed
                    )
                elif mode == "random_mutation":
                    diagnosis = diagnose(base_metrics, round_idx)
                    # No diagnosis signal: draw a uniformly random structural
                    # edit and adopt it without held-out validation.
                    candidate, mutation = controller.mutator.random_mutate(program)
                    evaluation = evaluate_program(
                        candidate,
                        seeds=controller.meta_seeds,
                        config=controller.config,
                        steps=controller.meta_steps,
                    )
                    best = MutationResult(
                        mutation=mutation,
                        candidate=candidate,
                        evaluation=evaluation,
                        delta=evaluation.mean_objective - base_metrics.objective,
                        accepted=True,
                        constraint_violations=[],
                        gain_lcb=evaluation.mean_objective - base_metrics.objective,
                    )
                    results = [best]
                    new_program = candidate
                    diagnosis = diagnosis
                else:
                    new_program, diagnosis, results, best = controller.step(
                        program, round_idx
                    )
                round_record["diagnosis"] = diagnosis.to_dict()
                round_record["candidate_results"] = [r.to_dict() for r in results]
                if best is not None:
                    registry.register(new_program)
                    registry.record_adoption(
                        new_program,
                        round_idx=round_idx,
                        evidence=best.to_dict(),
                    )
                    program = new_program
                    round_record["adopted"] = True
                    round_record["adopted_program_version"] = program.version
                    round_record["adopted_program_name"] = program.name
                    round_record["adopted_ops"] = [op.kind for op in program.ops]
                    round_record["mutation"] = best.mutation.to_dict()
            rounds.append(round_record)

        # Held-out test seeds, not used for selection.
        test_seeds = [
            seed + 900 + 13 * i for i in range(settings.test_seeds)
        ]
        test_config = replace(
            settings.market_config(seed), shift=float(settings.test_shift)
        )
        initial = seed_program()
        initial_eval = evaluate_program(
            initial,
            seeds=test_seeds,
            config=test_config,
            steps=settings.test_steps,
        )
        final_eval = evaluate_program(
            program,
            seeds=test_seeds,
            config=test_config,
            steps=settings.test_steps,
        )
        total_steps = int(settings.meta_rounds * settings.meta_steps)
        gains = [
            float(f.objective) - float(i.objective)
            for i, f in zip(initial_eval.metrics, final_eval.metrics)
        ]
        objective_gain = _mean(gains)
        gain_std = _std(gains)
        objective_gain_lcb = (
            objective_gain - 1.96 * gain_std / math.sqrt(len(gains))
            if len(gains) > 1
            else objective_gain
        )
        productivity = objective_gain / total_steps if total_steps else 0.0

        # Held-out meta panel after the last adoption, used for the
        # round-by-round productivity / compounding audit.
        round_eval_seeds = [
            seed + 500 + 11 * i for i in range(max(2, settings.meta_seeds))
        ]
        final_round_eval = evaluate_program(
            program,
            seeds=round_eval_seeds,
            config=settings.market_config(seed),
            steps=settings.meta_steps,
        )
        round_objectives = [
            float(r["round_heldout_objective"]) for r in rounds
        ] + [float(final_round_eval.mean_objective)]
        round_gains = [
            round_objectives[i + 1] - round_objectives[i]
            for i in range(len(round_objectives) - 1)
        ]
        round_productivities = [
            gain / settings.meta_steps for gain in round_gains
        ]
        if len(round_productivities) > 1:
            compounding_slope = float(
                np.polyfit(
                    np.arange(len(round_productivities), dtype=float),
                    np.asarray(round_productivities, dtype=float),
                    1,
                )[0]
            )
            half = max(1, len(round_productivities) // 2)
            compounding_delta = float(
                _mean(round_productivities[-half:])
                - _mean(round_productivities[:half])
            )
        else:
            compounding_slope = 0.0
            compounding_delta = 0.0

        # Composition synergy: compare the whole program's held-out gain to
        # the sum of the same operators applied *alone* from the seed program.
        # Positive synergy means the composed update rule is more than the sum
        # of its parts; zero/negative means the operators are redundant.
        solo_contributions: list[float] = []
        if len(program.ops) > 1:
            for idx, op in enumerate(program.ops):
                solo = program.copy_with(
                    [op],
                    version=program.version + 1000 + idx,
                    name=f"{program.name}_solo_{idx}",
                    source="leave_one_in",
                    rationale=f"solo operator: {op.kind}",
                )
                solo_eval = evaluate_program(
                    solo,
                    seeds=test_seeds,
                    config=test_config,
                    steps=settings.test_steps,
                )
                solo_contributions.append(
                    float(solo_eval.mean_objective)
                    - float(initial_eval.mean_objective)
                )
        sum_solo = sum(solo_contributions)
        synergy = objective_gain - sum_solo

        structural_edits = sum(
            1
            for r in rounds
            if r.get("adopted")
            and isinstance(r.get("mutation"), dict)
            and r["mutation"].get("structural")
        )
        parameter_edits = sum(
            1
            for r in rounds
            if r.get("adopted")
            and isinstance(r.get("mutation"), dict)
            and not r["mutation"].get("structural")
        )
        generated_structure = bool(
            structural_edits >= 2 and len(program.ops) >= 2
        )
        reuse = {
            str(v): sum(
                1
                for r in rounds
                if int(r["evaluated_program_version"]) == v
            )
            for v in sorted(
                {
                    int(r["adopted_program_version"])
                    for r in rounds
                    if r.get("adopted")
                }
            )
        }
        sample_efficiency_gain = float(
            final_eval.mean_sample_efficiency
            - initial_eval.mean_sample_efficiency
        )
        certificate = {
            "n1_persistence": bool(
                len(registry.all()) >= 2 and objective_gain > 0.0
            ),
            "n2_meta_level_generated": bool(
                open_language and generated_structure and structural_edits >= 2
            ),
            "n3_independent_validation": bool(objective_gain_lcb > 0.0),
            "n4_compounding": bool(
                synergy > 0.0
                or (
                    sample_efficiency_gain > 0.2
                    and objective_gain_lcb > 0.0
                )
            ),
            "n5_autonomy_over_criteria": False,
            "bounded": True,
            "open_language": bool(open_language),
            "structural_edits": int(structural_edits),
            "parameter_edits": int(parameter_edits),
            "generated_structure": generated_structure,
            "reuse": reuse,
            "composition_synergy": float(synergy),
            "compounding_slope": float(compounding_slope),
            "compounding_delta": float(compounding_delta),
            "sample_efficiency_gain": float(sample_efficiency_gain),
            "objective_gain_lcb": float(objective_gain_lcb),
        }
        certificate["passed"] = bool(
            certificate["n1_persistence"]
            and certificate["n2_meta_level_generated"]
            and certificate["n3_independent_validation"]
            and certificate["n4_compounding"]
        )
        certificate["rsi_level"] = (
            "bounded_rsi" if certificate["passed"] else "bounded_adaptation"
        )
        program_signature = ">".join(op.kind for op in program.ops)
        return {
            "seed": int(seed),
            "mode": mode,
            "initial_program": initial.to_dict(),
            "final_program": program.to_dict(),
            "program_signature": program_signature,
            "rounds": rounds,
            "registry": registry.to_dict(),
            "heldout_test": {
                "seeds": test_seeds,
                "test_shift": float(settings.test_shift),
                "initial_objective": float(initial_eval.mean_objective),
                "final_objective": float(final_eval.mean_objective),
                "objective_gain": float(objective_gain),
                "objective_gain_std": float(gain_std),
                "objective_gain_lcb": float(objective_gain_lcb),
                "initial_buyer_mse": float(initial_eval.mean_buyer_mse),
                "final_buyer_mse": float(final_eval.mean_buyer_mse),
                "initial_seller_mse": float(initial_eval.mean_seller_mse),
                "final_seller_mse": float(final_eval.mean_seller_mse),
                "initial_regret": float(initial_eval.mean_regret),
                "final_regret": float(final_eval.mean_regret),
                "initial_trade_rate": float(initial_eval.mean_trade_rate),
                "final_trade_rate": float(final_eval.mean_trade_rate),
                "final_surplus": float(final_eval.mean_surplus),
                "initial_buyer_utility": float(initial_eval.mean_buyer_utility),
                "final_buyer_utility": float(final_eval.mean_buyer_utility),
                "initial_seller_utility": float(initial_eval.mean_seller_utility),
                "final_seller_utility": float(final_eval.mean_seller_utility),
                "initial_sample_efficiency": float(
                    initial_eval.mean_sample_efficiency
                ),
                "final_sample_efficiency": float(
                    final_eval.mean_sample_efficiency
                ),
                "sample_efficiency_gain": float(sample_efficiency_gain),
            },
            "total_meta_steps": total_steps,
            "meta_improvement_productivity": float(productivity),
            "round_objectives": [float(x) for x in round_objectives],
            "round_productivities": [float(x) for x in round_productivities],
            "compounding_slope": float(compounding_slope),
            "compounding_delta": float(compounding_delta),
            "composition_synergy": float(synergy),
            "solo_contributions": [float(x) for x in solo_contributions],
            "reuse": reuse,
            "rsi_certificate": certificate,
        }

    def run(self, modes: Sequence[str] | None = None) -> dict[str, Any]:
        modes = list(modes or self.MODES)
        results = [
            self.run_seed(int(seed), mode)
            for mode in modes
            for seed in self.settings.seeds
        ]
        summary = self.aggregate(results)
        return {
            "settings": asdict(self.settings),
            "results": results,
            "summary": summary,
            "rsi_evidence": self.rsi_evidence(summary),
        }

    @staticmethod
    def rsi_evidence(summary: dict[str, Any]) -> dict[str, Any]:
        """Aggregate the per-seed certificates into one bounded-RSI claim.

        A per-seed certificate can be passed by a lucky random edit.  The
        aggregate claim therefore also requires a margin over every control.
        """

        meta = summary.get("meta_structural", {})
        controls = {
            name: summary.get(name, {})
            for name in (
                "fixed",
                "meta_param",
                "random_mutation",
                "self_validated",
                "closed_library",
            )
        }
        meta_gain = float(meta.get("mean_objective_gain", 0.0))
        margins = {
            name: meta_gain - float(row.get("mean_objective_gain", 0.0))
            for name, row in controls.items()
        }
        checks = {
            "controls_present": all(
                name in summary
                for name in (
                    "fixed",
                    "meta_param",
                    "random_mutation",
                    "self_validated",
                    "closed_library",
                )
            ),
            "meta_pass_rate": float(meta.get("rsi_pass_rate", 0.0)) >= 0.8,
            "meta_gain_positive": meta_gain > 0.0,
            "meta_lcb_positive": float(meta.get("mean_gain_lcb", 0.0)) > 0.0,
            "beats_fixed": margins.get("fixed", 0.0) > 0.0,
            "beats_meta_param": margins.get("meta_param", 0.0) > 0.0,
            "beats_random_mutation": margins.get("random_mutation", 0.0) > 0.0,
            "beats_self_validated": margins.get("self_validated", 0.0) > 0.0,
            "beats_closed_library": margins.get("closed_library", 0.0) > 0.0,
        }
        claim = "bounded_rsi" if all(checks.values()) else "bounded_adaptation"
        return {
            "claim": claim,
            "checks": checks,
            "meta_pass_rate": float(meta.get("rsi_pass_rate", 0.0)),
            "meta_mean_gain": meta_gain,
            "meta_mean_gain_lcb": float(meta.get("mean_gain_lcb", 0.0)),
            "margins": {k: round(v, 6) for k, v in margins.items()},
            "control_pass_rates": {
                name: float(row.get("rsi_pass_rate", 0.0))
                for name, row in controls.items()
            },
            "note": (
                "bounded_rsi requires N1-N4 per seed in >=80% of seeds and a "
                "mean held-out gain margin over every control; N5 is not claimed."
            ),
        }

    @staticmethod
    def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
        by_mode: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            by_mode.setdefault(result["mode"], []).append(result)
        summary: dict[str, Any] = {}
        for mode, rows in by_mode.items():
            gains = [float(r["heldout_test"]["objective_gain"]) for r in rows]
            productivity = [
                float(r["meta_improvement_productivity"]) for r in rows
            ]
            adoptions = [
                sum(1 for rd in r["rounds"] if rd.get("adopted")) for r in rows
            ]
            lcbs = [
                float(r["heldout_test"]["objective_gain_lcb"]) for r in rows
            ]
            synergy = [float(r["composition_synergy"]) for r in rows]
            compounding = [float(r["compounding_delta"]) for r in rows]
            sample_eff = [
                float(r["heldout_test"]["sample_efficiency_gain"]) for r in rows
            ]
            pass_rate = [
                1.0 if r["rsi_certificate"]["passed"] else 0.0 for r in rows
            ]
            summary[mode] = {
                "n_seeds": len(rows),
                "mean_objective_gain": _mean(gains),
                "std_objective_gain": _std(gains),
                "mean_gain_lcb": _mean(lcbs),
                "mean_productivity": _mean(productivity),
                "std_productivity": _std(productivity),
                "mean_adoptions": _mean([float(x) for x in adoptions]),
                "mean_composition_synergy": _mean(synergy),
                "mean_compounding_delta": _mean(compounding),
                "mean_sample_efficiency_gain": _mean(sample_eff),
                "rsi_pass_rate": _mean(pass_rate),
                "gains": [round(g, 6) for g in gains],
                "lcb": [round(x, 6) for x in lcbs],
                "synergy": [round(x, 6) for x in synergy],
                "compounding": [round(x, 6) for x in compounding],
            }
        return summary

    def save(self, payload: dict[str, Any], output_dir: str | Path | None = None) -> Path:
        out = Path(output_dir or self.settings.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "mode",
                    "seed",
                    "initial_objective",
                    "final_objective",
                    "objective_gain",
                    "objective_gain_lcb",
                    "productivity",
                    "adoptions",
                    "composition_synergy",
                    "compounding_delta",
                    "sample_efficiency_gain",
                    "rsi_level",
                ]
            )
            for result in payload["results"]:
                writer.writerow(
                    [
                        result["mode"],
                        result["seed"],
                        result["heldout_test"]["initial_objective"],
                        result["heldout_test"]["final_objective"],
                        result["heldout_test"]["objective_gain"],
                        result["heldout_test"]["objective_gain_lcb"],
                        result["meta_improvement_productivity"],
                        sum(1 for r in result["rounds"] if r.get("adopted")),
                        result["composition_synergy"],
                        result["compounding_delta"],
                        result["heldout_test"]["sample_efficiency_gain"],
                        result["rsi_certificate"]["rsi_level"],
                    ]
                )
        (out / "report.md").write_text(self.render_report(payload), encoding="utf-8")
        return out

    @staticmethod
    def render_report(payload: dict[str, Any]) -> str:
        lines = [
            "# Bilateral buyer-seller bounded RSI experiment",
            "",
            "Buyers learn demand-based valuations and bids; sellers learn",
            "cost-based valuations and asks; the inner loop updates both sides from",
            "realized transaction outcomes.  The update rule is a program; the",
            "meta-level generates new programs by structural mutations, validates",
            "them on held-out meta-episodes, and versions/reuses the adopted",
            "program.  Test seeds are not used for selection.",
            "",
            "| mode | seeds | mean gain | gain LCB | mean synergy | mean comp. delta | mean sample-eff. gain | RSI pass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, row in sorted(payload["summary"].items()):
            lines.append(
                f"| {mode} | {row['n_seeds']} | "
                f"{row['mean_objective_gain']:.4f} | "
                f"{row['mean_gain_lcb']:.4f} | "
                f"{row['mean_composition_synergy']:.4f} | "
                f"{row['mean_compounding_delta']:.4f} | "
                f"{row['mean_sample_efficiency_gain']:.4f} | "
                f"{row['rsi_pass_rate']:.2f} |"
            )
        evidence = payload.get("rsi_evidence", {})
        lines.extend(
            [
                "",
                "## Aggregate RSI certificate",
                "",
                f"Claim: **{evidence.get('claim', 'unknown')}**",
                "",
                "| check | passed |",
                "|---|---:|",
            ]
        )
        for name, value in evidence.get("checks", {}).items():
            lines.append(f"| {name} | {value} |")
        lines.extend(
            [
                "",
                f"Meta pass rate: {evidence.get('meta_pass_rate', 0.0):.2f}; "
                f"mean gain: {evidence.get('meta_mean_gain', 0.0):.4f}; "
                f"LCB: {evidence.get('meta_mean_gain_lcb', 0.0):.4f}.",
                "",
                "## Meta-structural per-seed programs",
                "",
                "| seed | final program | gain | gain LCB | synergy | sample-eff. | certificate |",
                "|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for result in payload["results"]:
            if result["mode"] != "meta_structural":
                continue
            cert = result["rsi_certificate"]
            lines.append(
                f"| {result['seed']} | `{result['program_signature']}` | "
                f"{result['heldout_test']['objective_gain']:.4f} | "
                f"{result['heldout_test']['objective_gain_lcb']:.4f} | "
                f"{cert['composition_synergy']:.4f} | "
                f"{cert['sample_efficiency_gain']:.4f} | "
                f"{cert['rsi_level']} |"
            )
        lines.extend(
            [
                "",
                "## Versioning and reuse",
                "",
                "| seed | structural edits | parameter edits | reuse counts | N1 |",
                "|---:|---:|---:|---|---|",
            ]
        )
        for result in payload["results"]:
            if result["mode"] != "meta_structural":
                continue
            cert = result["rsi_certificate"]
            reuse = ", ".join(
                f"v{k}:{v}" for k, v in cert["reuse"].items()
            ) or "none"
            lines.append(
                f"| {result['seed']} | {cert['structural_edits']} | "
                f"{cert['parameter_edits']} | {reuse} | "
                f"{cert['n1_persistence']} |"
            )
        lines.extend(
            [
                "",
                "## How to read this",
                "",
                "* `fixed`: the seed update program never changes.",
                "* `meta_structural`: the meta-level mutates the update program",
                "  (insert/delete/replace/swap/wrap/duplicate) and adopts a mutation",
                "  only if it improves the held-out meta-objective with a positive LCB.",
                "* `meta_param`: only numeric parameters are mutated; the program",
                "  structure is fixed. This is the tuning control.",
                "* `random_mutation`: a structural mutation is adopted without",
                "  meta-validation; it tests whether the meta-level matters.",
                "* `self_validated`: a mutation is accepted on the single training",
                "  market's transaction utility; there is no held-out anchor.",
                "* `closed_library`: the edit language is restricted to buyer-side",
                "  operators; it cannot generate the seller/calibration composition.",
                "",
                "Per-seed N1-N4: N1 versioned and transferred to held-out tasks;",
                "N2 open-language generated multi-op structure (>=2 structural edits);",
                "N3 held-out gain LCB > 0; N4 composition synergy > 0 or a",
                "sample-efficiency gain > 0.2 with a positive LCB. N5 is not claimed.",
                "",
            ]
        )
        return "\n".join(lines)
