"""LLM-backed bilateral buyer--seller market with bounded RSI.

This module implements the construction discussed on 2026-11-02/03, with the
buyer's valuation actually produced by an LLM:

    demand d
      -> LLM valuation operator V_b(d, memory, history; theta)
      -> quote operator B_b(V_b, history; phi)
      -> bilateral bid/ask with a seller
      -> transaction / no-trade + realized value + realized cost + price
      -> update operators U_V, U_B (self-evolution)
      -> meta-level generates new update programs U' (RSI)

The key three layers are kept separate:

1. **Valuation/quote layer**: `BuyerState.valuation` and `.quote`,
   `SellerState.valuation` and `.quote`.
2. **Self-evolution layer**: `UpdateProgram` operators update the parameters
   of the valuation/quote functions from transaction outcomes.
3. **RSI layer**: `ProgramMutator` and `LLMMetaProposer` generate *new*
   update programs (including new valuation-prompt addenda), validate them on
   held-out meta-episodes, and version/reuse the adopted program.

The LLM is used for the semantic part of valuation and for generating new
valuation-prompt operators.  The numeric calibration / demand weights / quote
parameters are updated by explicit operators, so the learning signal is
auditable.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Protocol, Sequence

import numpy as np

from .config import DEFAULT_RESULTS_DIR, LLMConfig
from .llm import DeepSeekClient, LLMError

FACETS = ("scheduling", "pricing", "compliance", "supply", "customer", "risk")
SOURCES = ("vendor", "internal", "market", "regulator")


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Demand:
    id: str
    question: str
    required_facets: tuple[str, ...]
    facet_weights: dict[str, float]
    budget: float
    source_limit: int = 1
    deadline: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "required_facets": list(self.required_facets),
            "facet_weights": {k: round(v, 4) for k, v in self.facet_weights.items()},
            "budget": self.budget,
            "source_limit": self.source_limit,
        }


@dataclass(frozen=True)
class MemoryItem:
    id: str
    seller_id: str
    content: str
    facets: tuple[str, ...]
    source: str
    quality: float
    production_cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seller_id": self.seller_id,
            "content": self.content,
            "facets": list(self.facets),
            "source": self.source,
            "quality": round(self.quality, 4),
            "production_cost": round(self.production_cost, 4),
        }


@dataclass
class LLMValue:
    semantic_value: float
    confidence: float
    matched_facets: tuple[str, ...]
    redundancy_risk: float
    rationale: str = ""
    used_llm: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_value": round(float(self.semantic_value), 4),
            "confidence": round(float(self.confidence), 4),
            "matched_facets": list(self.matched_facets),
            "redundancy_risk": round(float(self.redundancy_risk), 4),
            "rationale": self.rationale[:300],
            "used_llm": self.used_llm,
        }


@dataclass
class LLMCost:
    cost_estimate: float
    scarcity: float
    quality: float
    rationale: str = ""
    used_llm: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_estimate": round(float(self.cost_estimate), 4),
            "scarcity": round(float(self.scarcity), 4),
            "quality": round(float(self.quality), 4),
            "rationale": self.rationale[:300],
            "used_llm": self.used_llm,
        }


@dataclass
class ValuationRecord:
    semantic: float
    matched_facets: tuple[str, ...]
    redundancy: float
    realized_value: float


@dataclass
class BuyerState:
    id: str
    demand: Demand
    scale: float = 1.8
    bias: float = 2.2
    demand_weights: dict[str, float] = field(default_factory=dict)
    source_penalty: float = 0.5
    shading: float = 0.15
    bundle: list[str] = field(default_factory=list)
    buffer: list[ValuationRecord] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for facet in self.demand.required_facets:
            self.demand_weights.setdefault(facet, 0.2)

    def valuation(self, llm: LLMValue) -> float:
        facet_bonus = sum(
            self.demand_weights.get(f, 0.0)
            for f in llm.matched_facets
            if f in self.demand.required_facets
        )
        value = (
            self.scale * llm.semantic_value
            + self.bias
            + facet_bonus
            - self.source_penalty * llm.redundancy_risk
        )
        return float(np.clip(value, 0.0, self.demand.budget))

    def quote(self, value: float) -> float:
        return float(np.clip(value * (1.0 - self.shading), 0.0, self.demand.budget))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "demand": self.demand.to_dict(),
            "scale": round(self.scale, 4),
            "bias": round(self.bias, 4),
            "demand_weights": {k: round(v, 4) for k, v in self.demand_weights.items()},
            "source_penalty": round(self.source_penalty, 4),
            "shading": round(self.shading, 4),
            "bundle": list(self.bundle),
        }


@dataclass
class SellerState:
    id: str
    scale: float = 1.3
    bias: float = 0.8
    scarcity_bonus: float = 0.5
    markup: float = 0.2
    sold: dict[str, int] = field(default_factory=dict)
    buffer: list[tuple[float, float]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def valuation(self, llm: LLMCost) -> float:
        value = (
            self.scale * llm.cost_estimate
            + self.bias
            + self.scarcity_bonus * llm.scarcity
        )
        return float(max(0.0, value))

    def quote(self, value: float) -> float:
        return float(max(0.0, value * (1.0 + self.markup)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scale": round(self.scale, 4),
            "bias": round(self.bias, 4),
            "scarcity_bonus": round(self.scarcity_bonus, 4),
            "markup": round(self.markup, 4),
            "sold": dict(self.sold),
        }


@dataclass
class LLMMarketConfig:
    seed: int = 0
    n_buyers: int = 2
    n_sellers: int = 2
    n_memories: int = 6
    n_steps: int = 6
    noise: float = 0.05
    reference_noise: float = 0.05
    price_rule: str = "midpoint"  # midpoint | buyer | seller
    shift: float = 0.0
    max_llm_tokens: int = 600
    temperature: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transaction:
    step: int
    buyer_id: str
    seller_id: str
    memory_id: str
    demand_id: str
    memory_content: str
    buyer_llm: LLMValue
    seller_llm: LLMCost
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
    buyer_quote_regret: float
    seller_quote_regret: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "memory_id": self.memory_id,
            "demand_id": self.demand_id,
            "memory_content": self.memory_content[:300],
            "buyer_llm": self.buyer_llm.to_dict(),
            "seller_llm": self.seller_llm.to_dict(),
            "predicted_value": round(float(self.predicted_value), 4),
            "predicted_cost": round(float(self.predicted_cost), 4),
            "bid": round(float(self.bid), 4),
            "ask": round(float(self.ask), 4),
            "traded": bool(self.traded),
            "price": round(float(self.price), 4),
            "realized_value": round(float(self.realized_value), 4),
            "realized_cost": round(float(self.realized_cost), 4),
            "reference_price": round(float(self.reference_price), 4),
            "buyer_utility": round(float(self.buyer_utility), 4),
            "seller_utility": round(float(self.seller_utility), 4),
            "surplus": round(float(self.surplus), 4),
            "buyer_quote_regret": round(float(self.buyer_quote_regret), 4),
            "seller_quote_regret": round(float(self.seller_quote_regret), 4),
        }


# ----------------------------------------------------------------------
# LLM client wrapper
# ----------------------------------------------------------------------


class JSONClient(Protocol):
    def json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]: ...


class LLMBilateralClient:
    """Prompts for buyer value, seller cost, and meta-level prompt revision."""

    def __init__(
        self,
        client: JSONClient | None = None,
        *,
        config: LLMConfig | None = None,
        max_tokens: int = 600,
        temperature: float = 0.0,
    ):
        self.client = client or DeepSeekClient(config)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.calls = 0
        self.failures = 0
        self.cache: dict[str, dict[str, Any]] = {}

    @property
    def available(self) -> bool:
        return bool(getattr(self.client, "available", True))

    def _json(self, messages: list[dict[str, str]], key: str) -> dict[str, Any]:
        cache_key = json.dumps(
            {"messages": messages, "max_tokens": self.max_tokens}, sort_keys=True
        )
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            data = self.client.json(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                use_cache=True,
            )
            self.calls += 1
        except (LLMError, ValueError, TypeError) as exc:
            self.failures += 1
            raise LLMError(f"LLM bilateral call failed: {exc}") from exc
        if not isinstance(data, dict):
            self.failures += 1
            raise LLMError(f"LLM returned non-object for {key}")
        self.cache[cache_key] = data
        return data

    def buyer_value(
        self,
        buyer: BuyerState,
        memory: MemoryItem,
        *,
        held: Sequence[str],
        addendum: str = "",
    ) -> LLMValue:
        history = "\n".join(
            f"- memory={h.get('memory_id')} value={h.get('predicted_value')} "
            f"realized={h.get('realized_value')} price={h.get('price')} "
            f"traded={h.get('traded')}"
            for h in buyer.history[-4:]
        ) or "- none"
        prompt = f"""You are buyer agent {buyer.id} in a bilateral memory market.

Demand:
- question: {buyer.demand.question}
- required facets: {list(buyer.demand.required_facets)}
- budget: {buyer.demand.budget:.2f}
- source limit: at most {buyer.demand.source_limit} memory from one source

Memory for sale:
- content: {memory.content}
- facets: {list(memory.facets)}
- source: {memory.source}
- quality hint (noisy): {memory.quality:.2f}

Already held: {list(held)}
Recent outcomes:
{history}

{addendum}

Estimate the raw semantic usefulness of this memory for the demand.  Do not
use outside knowledge and do not apply any calibration, budget, or quote rule;
the controller applies those after your response.  Return ONLY JSON:
{{"semantic_value": 0.0-10.0, "confidence": 0.0-1.0,
  "matched_facets": ["..."], "redundancy_risk": 0.0-1.0,
  "rationale": "one sentence"}}"""
        try:
            data = self._json(
                [
                    {
                        "role": "system",
                        "content": "You are a calibrated buyer. Return only JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "buyer_value",
            )
            semantic = float(np.clip(float(data.get("semantic_value", 0.0)), 0.0, 10.0))
            confidence = float(np.clip(float(data.get("confidence", 0.5)), 0.0, 1.0))
            matched = tuple(
                f for f in data.get("matched_facets", []) if f in FACETS
            )
            redundancy = float(
                np.clip(float(data.get("redundancy_risk", 0.0)), 0.0, 1.0)
            )
            return LLMValue(
                semantic_value=semantic,
                confidence=confidence,
                matched_facets=matched,
                redundancy_risk=redundancy,
                rationale=str(data.get("rationale", ""))[:300],
                used_llm=True,
            )
        except (LLMError, ValueError, TypeError):
            # Deterministic fallback so the market can still finish.
            overlap = len(set(buyer.demand.required_facets) & set(memory.facets))
            coverage = overlap / max(1, len(buyer.demand.required_facets))
            return LLMValue(
                semantic_value=5.0 * coverage + 2.0 * memory.quality,
                confidence=0.3,
                matched_facets=tuple(
                    f for f in memory.facets if f in buyer.demand.required_facets
                ),
                redundancy_risk=min(1.0, len(held) / 3.0),
                rationale="fallback heuristic",
                used_llm=False,
            )

    def seller_value(
        self,
        seller: SellerState,
        memory: MemoryItem,
        *,
        addendum: str = "",
    ) -> LLMCost:
        history = "\n".join(
            f"- memory={h.get('memory_id')} predicted_cost={h.get('predicted_cost')} "
            f"realized_cost={h.get('realized_cost')} price={h.get('price')} "
            f"traded={h.get('traded')}"
            for h in seller.history[-4:]
        ) or "- none"
        prompt = f"""You are seller agent {seller.id} in a bilateral memory market.

Memory asset:
- content: {memory.content}
- facets: {list(memory.facets)}
- source: {memory.source}
- production cost (known to you): {memory.production_cost:.2f}
- times sold: {seller.sold.get(memory.id, 0)}

Recent outcomes:
{history}

{addendum}

Estimate the raw production/opportunity cost of this memory.  Do not apply
any calibration or markup rule; the controller applies those after your
response.  Return ONLY JSON:
{{"cost_estimate": 0.0-10.0, "scarcity": 0.0-1.0,
  "quality": 0.0-1.0, "rationale": "one sentence"}}"""
        try:
            data = self._json(
                [
                    {
                        "role": "system",
                        "content": "You are a calibrated seller. Return only JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "seller_value",
            )
            cost = float(np.clip(float(data.get("cost_estimate", 0.0)), 0.0, 10.0))
            scarcity = float(np.clip(float(data.get("scarcity", 0.0)), 0.0, 1.0))
            quality = float(np.clip(float(data.get("quality", 0.5)), 0.0, 1.0))
            return LLMCost(
                cost_estimate=cost,
                scarcity=scarcity,
                quality=quality,
                rationale=str(data.get("rationale", ""))[:300],
                used_llm=True,
            )
        except (LLMError, ValueError, TypeError):
            sold = seller.sold.get(memory.id, 0)
            return LLMCost(
                cost_estimate=memory.production_cost + 0.4 * memory.quality,
                scarcity=min(1.0, sold / 3.0),
                quality=memory.quality,
                rationale="fallback heuristic",
                used_llm=False,
            )

    def propose_prompt_addendum(
        self,
        *,
        side: str,
        diagnosis: str,
        current_addendum: str,
    ) -> str:
        prompt = f"""You are the meta-level of a bilateral memory market.
Rewrite the {side} valuation operator instruction so the next inner-learning
episode reduces valuation error and quote regret.

Diagnosis: {diagnosis}
Current instruction: {current_addendum or "(empty)"}

Return ONLY JSON: {{"addendum": "one short imperative instruction, <= 40 words"}}.
Do not mention the evaluation metric or hidden ground truth."""
        try:
            data = self._json(
                [
                    {
                        "role": "system",
                        "content": "You write concise, testable valuation instructions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "prompt_addendum",
            )
            text = str(data.get("addendum", "")).strip()
            return text[:400]
        except (LLMError, ValueError, TypeError):
            return current_addendum


# ----------------------------------------------------------------------
# Market
# ----------------------------------------------------------------------


class LLMBilateralMarket:
    def __init__(self, config: LLMMarketConfig, llm: LLMBilateralClient):
        self.config = config
        self.llm = llm
        self.rng = np.random.default_rng(config.seed)
        self.demands = self._make_demands()
        self.memories = self._make_memories()
        self.buyers = self._make_buyers()
        self.sellers = self._make_sellers()
        self.step_idx = 0

    def _make_demands(self) -> list[Demand]:
        shift = float(self.config.shift)
        templates = [
            (
                "Plan a compliant supplier switch under a tight deadline.",
                ("compliance", "supply", "risk"),
                {"compliance": 1.2, "supply": 1.0, "risk": 0.8},
                6.0,
            ),
            (
                "Price a new customer contract without violating policy.",
                ("pricing", "customer", "compliance"),
                {"pricing": 1.3, "customer": 0.9, "compliance": 0.7},
                7.0,
            ),
            (
                "Resolve a scheduling conflict across two vendors.",
                ("scheduling", "supply", "risk"),
                {"scheduling": 1.2, "supply": 0.9, "risk": 0.9},
                5.5,
            ),
        ]
        demands: list[Demand] = []
        for idx in range(self.config.n_buyers):
            question, required, weights, budget = templates[idx % len(templates)]
            adjusted = {
                k: round(v * (1.0 + 0.15 * shift * (1 if i % 2 == 0 else -1)), 3)
                for i, (k, v) in enumerate(weights.items())
            }
            demands.append(
                Demand(
                    id=f"d{idx}",
                    question=question,
                    required_facets=required,
                    facet_weights=adjusted,
                    budget=budget,
                    source_limit=1,
                )
            )
        return demands

    def _make_memories(self) -> list[MemoryItem]:
        contents = [
            "Vendor V7 has a compliant audit trail but a two-week lead time; its risk note is stale.",
            "Internal policy 4.2 requires dual approval for supplier switches above 50k.",
            "Market index shows component prices fell 8% this quarter; supply is tight for one part.",
            "Customer segment A is price-sensitive; discounting above 12% needs compliance review.",
            "Scheduling rule: cross-vendor handoffs need a 48-hour buffer and a named owner.",
            "Risk memo: the regulator changed reporting scope; old compliance checklists are incomplete.",
        ]
        sources = ["vendor", "internal", "market", "regulator", "internal", "regulator"]
        facets = [
            ("compliance", "supply", "risk"),
            ("compliance",),
            ("pricing", "supply", "risk"),
            ("pricing", "customer"),
            ("scheduling", "supply"),
            ("compliance", "risk"),
        ]
        memories: list[MemoryItem] = []
        for idx in range(self.config.n_memories):
            memories.append(
                MemoryItem(
                    id=f"m{idx}",
                    seller_id=f"seller{idx % self.config.n_sellers}",
                    content=contents[idx % len(contents)],
                    facets=facets[idx % len(facets)],
                    source=sources[idx % len(sources)],
                    quality=float(np.clip(0.55 + 0.35 * ((idx * 7) % 5) / 4, 0.0, 1.0)),
                    production_cost=float(0.8 + 0.4 * (idx % 3)),
                )
            )
        return memories

    def _make_buyers(self) -> list[BuyerState]:
        return [BuyerState(id=f"buyer{i}", demand=self.demands[i]) for i in range(len(self.demands))]

    def _make_sellers(self) -> list[SellerState]:
        return [SellerState(id=f"seller{i}") for i in range(self.config.n_sellers)]

    def _seller_for(self, memory: MemoryItem) -> SellerState:
        return next(s for s in self.sellers if s.id == memory.seller_id)

    def true_value(self, buyer: BuyerState, memory: MemoryItem) -> float:
        match = sum(
            buyer.demand.facet_weights.get(f, 0.0)
            for f in buyer.demand.required_facets
            if f in memory.facets
        )
        coverage = len(
            set(buyer.demand.required_facets) & set(memory.facets)
        ) / max(1, len(buyer.demand.required_facets))
        base = 10.0 * match * memory.quality * (0.6 + 0.4 * coverage)
        same_source = sum(
            1
            for mid in buyer.bundle
            if any(m.source == memory.source for m in self.memories if m.id == mid)
        )
        penalty = 0.6 * max(0, same_source - buyer.demand.source_limit + 1)
        return float(np.clip(base - penalty, 0.0, buyer.demand.budget))

    def true_cost(self, seller: SellerState, memory: MemoryItem) -> float:
        scarcity = 1.0 / (1.0 + seller.sold.get(memory.id, 0))
        return float(
            memory.production_cost + 0.4 * memory.quality + 0.5 * (1.0 - scarcity)
        )

    def _sample(self) -> tuple[BuyerState, SellerState, MemoryItem]:
        buyer = self.buyers[int(self.rng.integers(0, len(self.buyers)))]
        available = [m for m in self.memories if m.id not in buyer.bundle]
        if not available:
            buyer.bundle.clear()
            available = list(self.memories)
        memory = available[int(self.rng.integers(0, len(available)))]
        return buyer, self._seller_for(memory), memory

    def _price(self, bid: float, ask: float) -> float:
        if self.config.price_rule == "buyer":
            return bid
        if self.config.price_rule == "seller":
            return ask
        return 0.5 * (bid + ask)

    def step(self, program: "UpdateProgram") -> Transaction:
        buyer, seller, memory = self._sample()
        buyer_llm = self.llm.buyer_value(
            buyer,
            memory,
            held=buyer.bundle,
            addendum=program.buyer_prompt_addendum,
        )
        seller_llm = self.llm.seller_value(
            seller,
            memory,
            addendum=program.seller_prompt_addendum,
        )
        predicted_value = buyer.valuation(buyer_llm)
        predicted_cost = seller.valuation(seller_llm)
        bid = buyer.quote(predicted_value)
        ask = seller.quote(predicted_cost)
        traded = bid >= ask
        price = self._price(bid, ask) if traded else 0.0

        value_noise = float(self.rng.normal(0.0, self.config.noise))
        cost_noise = float(self.rng.normal(0.0, self.config.noise))
        realized_value = max(0.0, self.true_value(buyer, memory) + value_noise)
        realized_cost = max(0.0, self.true_cost(seller, memory) + cost_noise)
        reference_noise = float(self.rng.normal(0.0, self.config.reference_noise))
        reference_price = 0.5 * (realized_value + realized_cost) + reference_noise

        buyer_utility = realized_value - price if traded else 0.0
        seller_utility = price - realized_cost if traded else 0.0
        surplus = realized_value - realized_cost if traded else 0.0
        buyer_quote_regret = max(0.0, price - reference_price) if traded else max(
            0.0, bid - reference_price
        )
        seller_quote_regret = max(0.0, reference_price - price) if traded else max(
            0.0, reference_price - ask
        )

        tx = Transaction(
            step=self.step_idx,
            buyer_id=buyer.id,
            seller_id=seller.id,
            memory_id=memory.id,
            demand_id=buyer.demand.id,
            memory_content=memory.content,
            buyer_llm=buyer_llm,
            seller_llm=seller_llm,
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
            buyer_quote_regret=buyer_quote_regret,
            seller_quote_regret=seller_quote_regret,
        )
        if traded:
            buyer.bundle.append(memory.id)
            seller.sold[memory.id] = seller.sold.get(memory.id, 0) + 1
        buyer.buffer.append(
            ValuationRecord(
                semantic=buyer_llm.semantic_value,
                matched_facets=buyer_llm.matched_facets,
                redundancy=buyer_llm.redundancy_risk,
                realized_value=realized_value,
            )
        )
        buyer.buffer = buyer.buffer[-64:]
        seller.buffer.append((seller_llm.cost_estimate, realized_cost))
        seller.buffer = seller.buffer[-64:]
        buyer.history.append(tx.to_dict())
        seller.history.append(tx.to_dict())
        buyer.history = buyer.history[-12:]
        seller.history = seller.history[-12:]
        program.apply(self, buyer, seller, tx)
        self.step_idx += 1
        return tx

    def run(self, program: "UpdateProgram", steps: int | None = None) -> list[Transaction]:
        steps = self.config.n_steps if steps is None else steps
        return [self.step(program) for _ in range(steps)]


# ----------------------------------------------------------------------
# Update programs
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
    buyer_prompt_addendum: str = ""
    seller_prompt_addendum: str = ""
    parent_version: int | None = None
    source: str = "human"
    rationale: str = ""
    lineage: list[str] = field(default_factory=list)

    def apply(
        self,
        market: LLMBilateralMarket,
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
        buyer_prompt_addendum: str | None = None,
        seller_prompt_addendum: str | None = None,
        lineage: list[str] | None = None,
    ) -> "UpdateProgram":
        return UpdateProgram(
            version=version,
            name=name or f"{self.name}_v{version}",
            ops=list(ops),
            buyer_prompt_addendum=(
                self.buyer_prompt_addendum
                if buyer_prompt_addendum is None
                else buyer_prompt_addendum
            ),
            seller_prompt_addendum=(
                self.seller_prompt_addendum
                if seller_prompt_addendum is None
                else seller_prompt_addendum
            ),
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
            "buyer_prompt_addendum": self.buyer_prompt_addendum,
            "seller_prompt_addendum": self.seller_prompt_addendum,
            "lineage": list(self.lineage),
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


def apply_op(
    op: UpdateOp,
    market: LLMBilateralMarket,
    buyer: BuyerState,
    seller: SellerState,
    tx: Transaction,
) -> None:
    kind = op.kind
    params = op.params
    lr = float(params.get("lr", 0.05))

    if kind == "calibrate_buyer":
        residual = tx.realized_value - tx.predicted_value
        semantic = max(0.1, float(tx.buyer_llm.semantic_value))
        buyer.scale = _clip(
            buyer.scale + lr * residual * semantic / (semantic * semantic + 1.0),
            0.2,
            3.0,
        )
        buyer.bias = _clip(buyer.bias + lr * residual, -5.0, 5.0)
        return

    if kind == "update_buyer_demand_weights":
        residual = tx.realized_value - tx.predicted_value
        for facet in tx.buyer_llm.matched_facets:
            if facet in buyer.demand.required_facets:
                buyer.demand_weights[facet] = _clip(
                    buyer.demand_weights.get(facet, 0.2)
                    + lr * math.copysign(1.0, residual),
                    0.0,
                    2.0,
                )
        return

    if kind == "update_buyer_source_penalty":
        if tx.buyer_llm.redundancy_risk > 0.4 and tx.realized_value < tx.predicted_value:
            buyer.source_penalty = _clip(buyer.source_penalty + lr, 0.0, 2.0)
        else:
            buyer.source_penalty = _clip(buyer.source_penalty - 0.25 * lr, 0.0, 2.0)
        return

    if kind == "update_buyer_quote":
        if tx.traded:
            regret = tx.reference_price - tx.price
            buyer.shading = _clip(
                buyer.shading + lr * math.copysign(1.0, regret), 0.0, 0.5
            )
        else:
            buyer.shading = _clip(buyer.shading - lr, 0.0, 0.5)
        return

    if kind == "replay_buyer":
        if len(buyer.buffer) < 3:
            return
        recent = buyer.buffer[-int(params.get("window", 12)) :]
        X = []
        y = []
        for rec in recent:
            facet_bonus = sum(
                buyer.demand_weights.get(f, 0.0)
                for f in rec.matched_facets
                if f in buyer.demand.required_facets
            )
            X.append([rec.semantic, 1.0])
            y.append(
                rec.realized_value
                + buyer.source_penalty * rec.redundancy
                - facet_bonus
            )
        try:
            coef, *_ = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)
            buyer.scale = _clip(float(coef[0]), 0.2, 3.0)
            buyer.bias = _clip(float(coef[1]), -5.0, 5.0)
        except np.linalg.LinAlgError:
            return
        return

    if kind == "explore_buyer_quote":
        eps = float(params.get("eps", 0.1))
        if buyer.history and random.random() < eps:
            buyer.shading = _clip(
                buyer.shading + float(np.random.normal(0.0, 0.01)), 0.0, 0.5
            )
        return

    if kind == "calibrate_seller":
        residual = tx.realized_cost - tx.predicted_cost
        cost = max(0.1, float(tx.seller_llm.cost_estimate))
        seller.scale = _clip(
            seller.scale + lr * residual * cost / (cost * cost + 1.0),
            0.2,
            3.0,
        )
        seller.bias = _clip(seller.bias + lr * residual, -5.0, 5.0)
        return

    if kind == "update_seller_scarcity":
        residual = tx.realized_cost - tx.predicted_cost
        seller.scarcity_bonus = _clip(
            seller.scarcity_bonus
            + lr * math.copysign(1.0, residual) * tx.seller_llm.scarcity,
            0.0,
            2.0,
        )
        return

    if kind == "update_seller_quote":
        if tx.traded:
            regret = tx.price - tx.reference_price
            seller.markup = _clip(
                seller.markup + lr * math.copysign(1.0, regret), 0.0, 0.5
            )
        else:
            seller.markup = _clip(seller.markup - lr, 0.0, 0.5)
        return

    if kind == "replay_seller":
        if len(seller.buffer) < 3:
            return
        recent = seller.buffer[-int(params.get("window", 12)) :]
        X = np.asarray([[c, 1.0] for c, _ in recent])
        y = np.asarray([v for _, v in recent])
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            seller.scale = _clip(float(coef[0]), 0.2, 3.0)
            seller.bias = _clip(float(coef[1]), -5.0, 5.0)
        except np.linalg.LinAlgError:
            return
        return

    if kind == "explore_seller_quote":
        eps = float(params.get("eps", 0.1))
        if seller.history and random.random() < eps:
            seller.markup = _clip(
                seller.markup + float(np.random.normal(0.0, 0.01)), 0.0, 0.5
            )
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
        OpTemplate("calibrate_buyer", {"lr": 0.05}, ("buyer_valuation", "calibration")),
        OpTemplate("update_buyer_demand_weights", {"lr": 0.03}, ("buyer_valuation", "demand")),
        OpTemplate("update_buyer_source_penalty", {"lr": 0.03}, ("buyer_valuation", "source")),
        OpTemplate("update_buyer_quote", {"lr": 0.03}, ("quote",)),
        OpTemplate("replay_buyer", {"window": 12}, ("replay", "buyer_valuation")),
        OpTemplate("explore_buyer_quote", {"eps": 0.1}, ("exploration", "quote")),
        OpTemplate("calibrate_seller", {"lr": 0.05}, ("seller_valuation", "calibration")),
        OpTemplate("update_seller_scarcity", {"lr": 0.03}, ("seller_valuation",)),
        OpTemplate("update_seller_quote", {"lr": 0.03}, ("quote",)),
        OpTemplate("replay_seller", {"window": 12}, ("replay", "seller_valuation")),
        OpTemplate("explore_seller_quote", {"eps": 0.1}, ("exploration", "quote")),
    ]


class ProgramMutator:
    def __init__(self, seed: int = 0, *, allowed_kinds: Sequence[str] | None = None):
        self.rng = random.Random(seed)
        self.templates = default_op_templates()
        if allowed_kinds is not None:
            allowed = set(allowed_kinds)
            self.templates = [t for t in self.templates if t.kind in allowed]

    def _choose(self, templates: Sequence[OpTemplate], diagnosis: "Diagnosis") -> OpTemplate:
        scores = []
        for template in templates:
            score = 1.0
            for tag in template.tags:
                score += 2.0 * float(diagnosis.tags.get(tag, 0.0))
            scores.append(score)
        total = sum(scores)
        pick = self.rng.random() * total
        acc = 0.0
        for template, score in zip(templates, scores):
            acc += score
            if pick <= acc:
                return template
        return templates[-1]

    def mutate(
        self,
        program: UpdateProgram,
        diagnosis: "Diagnosis",
        *,
        structural: bool = True,
    ) -> tuple[UpdateProgram, MutationRecord]:
        if not program.ops:
            template = self._choose(self.templates, diagnosis)
            op = UpdateOp(template.kind, dict(template.params))
            record = MutationRecord("insert", 0, op.kind, dict(op.params), f"insert {op.kind} at 0")
            return program.copy_with([op], version=program.version + 1, rationale=record.description, lineage=[record.description]), record

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
            template = self._choose(self.templates, diagnosis)
            op = UpdateOp(template.kind, dict(template.params))
            ops.insert(position, op)
            record = MutationRecord("insert", position, op.kind, dict(op.params), f"insert {op.kind} at {position}")
        elif mutation_kind == "delete":
            removed = ops.pop(position)
            record = MutationRecord("delete", position, removed.kind, dict(removed.params), f"delete {removed.kind} at {position}")
        elif mutation_kind == "replace":
            old = ops[position]
            alternatives = [t for t in self.templates if t.kind != old.kind]
            template = self.rng.choice(alternatives or self.templates)
            new_op = UpdateOp(template.kind, dict(template.params))
            ops[position] = new_op
            record = MutationRecord("replace", position, new_op.kind, dict(new_op.params), f"replace {old.kind} with {new_op.kind} at {position}")
        elif mutation_kind == "swap":
            other = self.rng.randrange(len(ops))
            while other == position:
                other = self.rng.randrange(len(ops))
            ops[position], ops[other] = ops[other], ops[position]
            record = MutationRecord("swap", position, ops[position].kind, dict(ops[position].params), f"swap positions {position} and {other}")
        elif mutation_kind == "wrap":
            wrap_kinds = {"calibrate_buyer", "calibrate_seller", "replay_buyer", "replay_seller"}
            wrap_templates = [t for t in self.templates if t.kind in wrap_kinds]
            template = self._choose(wrap_templates or self.templates, diagnosis)
            op = UpdateOp(template.kind, dict(template.params))
            ops.insert(position, op)
            record = MutationRecord("wrap", position, op.kind, dict(op.params), f"wrap with {op.kind} at {position}")
        elif mutation_kind == "duplicate":
            op = ops[position]
            ops.insert(position, UpdateOp(op.kind, dict(op.params)))
            record = MutationRecord("duplicate", position, op.kind, dict(op.params), f"duplicate {op.kind} at {position}")
        else:
            op = ops[position]
            params = dict(op.params)
            changed = []
            for key, value in list(params.items()):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    factor = self.rng.choice([0.5, 0.75, 1.25, 1.5])
                    new_value = value * factor
                    if isinstance(value, int):
                        new_value = max(1, int(round(new_value)))
                    params[key] = new_value
                    changed.append(f"{key}={new_value}")
            ops[position] = UpdateOp(op.kind, params)
            record = MutationRecord("param", position, op.kind, params, f"tune {op.kind}: {changed}", structural=False)

        return program.copy_with(ops, version=program.version + 1, rationale=record.description, lineage=[record.description]), record

    def random_mutate(self, program: UpdateProgram) -> tuple[UpdateProgram, MutationRecord]:
        neutral = Diagnosis(round_idx=-1, tags={}, rationale="random control")
        return self.mutate(program, neutral, structural=True)

    def _template_score(self, template: OpTemplate, diagnosis: "Diagnosis") -> float:
        return 1.0 + 2.0 * sum(
            float(diagnosis.tags.get(tag, 0.0)) for tag in template.tags
        )

    def portfolio(
        self,
        program: UpdateProgram,
        diagnosis: "Diagnosis",
        limit: int = 6,
    ) -> list[tuple[UpdateProgram, MutationRecord]]:
        """A diagnosis-ranked portfolio of structural edits.

        This is still *generation* (each candidate is a new program), but it
        avoids wasting the candidate budget on duplicate random draws.  The
        portfolio covers every operator family the diagnosis points at.
        """

        ranked = sorted(
            self.templates,
            key=lambda t: self._template_score(t, diagnosis),
            reverse=True,
        )
        out: list[tuple[UpdateProgram, MutationRecord]] = []
        seen: set[tuple[str, ...]] = set()
        existing = {op.kind for op in program.ops}
        # Coverage-first: take the best available operator for each diagnosis
        # tag, so a dominant seller-side failure cannot crowd out the buyer
        # valuation operator.  Then fill with the remaining highest scores.
        selected: list[OpTemplate] = []
        selected_kinds: set[str] = set()
        for tag in sorted(diagnosis.tags, key=lambda k: diagnosis.tags[k], reverse=True):
            options = [
                t for t in ranked
                if tag in t.tags and t.kind not in existing and t.kind not in selected_kinds
            ]
            if options:
                selected.append(options[0])
                selected_kinds.add(options[0].kind)
        # Always cover both sides, even if the diagnosis is one-sided.
        for prefix in ("buyer", "seller"):
            if not any(t.kind.startswith(prefix) for t in selected):
                options = [
                    t for t in ranked
                    if t.kind.startswith(prefix)
                    and t.kind not in existing
                    and t.kind not in selected_kinds
                ]
                if options:
                    selected.append(options[0])
                    selected_kinds.add(options[0].kind)
        for template in ranked:
            if len(selected) >= max(limit, len(selected)):
                break
            if template.kind in existing or template.kind in selected_kinds:
                continue
            selected.append(template)
            selected_kinds.add(template.kind)
        # Duplicates / existing kinds are a fallback if coverage is exhausted.
        for template in ranked:
            if template.kind not in selected_kinds:
                selected.append(template)
                selected_kinds.add(template.kind)
        for template in selected:
            op = UpdateOp(template.kind, dict(template.params))
            ops = list(program.ops) + [op]
            signature = tuple(o.kind for o in ops)
            if signature in seen:
                continue
            seen.add(signature)
            record = MutationRecord(
                "insert",
                len(ops) - 1,
                op.kind,
                dict(op.params),
                f"portfolio insert {op.kind} at end",
                structural=True,
            )
            out.append(
                (
                    program.copy_with(
                        ops,
                        version=program.version + 1,
                        rationale=record.description,
                        lineage=[record.description],
                    ),
                    record,
                )
            )
            if len(out) >= limit:
                break

        # Add a wrap variant around the first op if there is room.
        if out and len(out) < limit and program.ops:
            wrap_kinds = {
                "calibrate_buyer",
                "calibrate_seller",
                "replay_buyer",
                "replay_seller",
            }
            wrap_templates = [t for t in self.templates if t.kind in wrap_kinds]
            if wrap_templates:
                template = max(
                    wrap_templates,
                    key=lambda t: self._template_score(t, diagnosis),
                )
                op = UpdateOp(template.kind, dict(template.params))
                ops = [op] + list(program.ops)
                signature = tuple(o.kind for o in ops)
                if signature not in seen:
                    record = MutationRecord(
                        "wrap",
                        0,
                        op.kind,
                        dict(op.params),
                        f"portfolio wrap with {op.kind} at 0",
                        structural=True,
                    )
                    out.append(
                        (
                            program.copy_with(
                                ops,
                                version=program.version + 1,
                                rationale=record.description,
                                lineage=[record.description],
                            ),
                            record,
                        )
                    )

        # A multi-operator bundle: the meta-level can propose a coordinated
        # structural edit, not only one operator at a time.  This is still a
        # generated program, and it often has a larger, more stable effect.
        if len(selected) >= 3:
            bundle = selected[:3]
            ops = list(program.ops) + [
                UpdateOp(t.kind, dict(t.params)) for t in bundle
            ]
            signature = tuple(o.kind for o in ops)
            if signature not in seen:
                record = MutationRecord(
                    "bundle",
                    len(ops) - len(bundle),
                    "bundle",
                    {"kinds": [t.kind for t in bundle]},
                    "bundle insert " + " + ".join(t.kind for t in bundle),
                    structural=True,
                )
                out.append(
                    (
                        program.copy_with(
                            ops,
                            version=program.version + 1,
                            rationale=record.description,
                            lineage=[record.description],
                        ),
                        record,
                    )
                )
        return out

    def parameter_portfolio(
        self,
        program: UpdateProgram,
        limit: int = 6,
    ) -> list[tuple[UpdateProgram, MutationRecord]]:
        out: list[tuple[UpdateProgram, MutationRecord]] = []
        for position, op in enumerate(program.ops):
            for key, value in op.params.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                for factor in (0.5, 1.5):
                    params = dict(op.params)
                    new_value = value * factor
                    if isinstance(value, int):
                        new_value = max(1, int(round(new_value)))
                    params[key] = new_value
                    ops = list(program.ops)
                    ops[position] = UpdateOp(op.kind, params)
                    record = MutationRecord(
                        "param",
                        position,
                        op.kind,
                        params,
                        f"portfolio tune {op.kind}.{key}={new_value}",
                        structural=False,
                    )
                    out.append(
                        (
                            program.copy_with(
                                ops,
                                version=program.version + 1,
                                rationale=record.description,
                                lineage=[record.description],
                            ),
                            record,
                        )
                    )
                    if len(out) >= limit:
                        return out
        return out


# ----------------------------------------------------------------------
# Metrics and diagnosis
# ----------------------------------------------------------------------


@dataclass
class EpisodeMetrics:
    buyer_mse: float
    seller_mse: float
    buyer_mse_rel: float
    seller_mse_rel: float
    buyer_quote_regret: float
    seller_quote_regret: float
    regret: float
    trade_rate: float
    buyer_utility: float
    seller_utility: float
    surplus: float
    instability: float
    sample_efficiency: float
    objective: float
    llm_calls: int
    llm_failures: int

    def to_dict(self) -> dict[str, Any]:
        return {k: round(float(v), 6) if isinstance(v, float) else int(v) for k, v in asdict(self).items()}


def episode_objective(
    *,
    buyer_mse_rel: float,
    seller_mse_rel: float,
    buyer_quote_regret: float,
    seller_quote_regret: float,
    surplus: float,
    instability: float,
) -> float:
    return float(
        -(buyer_mse_rel + seller_mse_rel)
        - 0.5 * (buyer_quote_regret + seller_quote_regret)
        + 0.5 * surplus
        - 0.1 * instability
    )


def run_episode(
    program: UpdateProgram,
    config: LLMMarketConfig,
    *,
    llm: LLMBilateralClient,
    steps: int | None = None,
) -> tuple[EpisodeMetrics, list[Transaction]]:
    start_calls = llm.calls
    start_failures = llm.failures
    market = LLMBilateralMarket(config, llm)
    transactions = market.run(program, steps=steps)
    buyer_sq, seller_sq, buyer_reg, seller_reg = [], [], [], []
    buyer_util, seller_util, surplus = [], [], []
    learning_curve = []
    for tx in transactions:
        buyer_sq.append((tx.realized_value - tx.predicted_value) ** 2)
        seller_sq.append((tx.realized_cost - tx.predicted_cost) ** 2)
        buyer_reg.append(tx.buyer_quote_regret)
        seller_reg.append(tx.seller_quote_regret)
        if tx.traded:
            buyer_util.append(tx.buyer_utility)
            seller_util.append(tx.seller_utility)
            surplus.append(tx.surplus)
        learning_curve.append(0.5 * (buyer_sq[-1] + seller_sq[-1]))
    buyer_mse = float(mean(buyer_sq)) if buyer_sq else 0.0
    seller_mse = float(mean(seller_sq)) if seller_sq else 0.0
    true_values = [tx.realized_value for tx in transactions]
    true_costs = [tx.realized_cost for tx in transactions]
    buyer_scale = float(np.var(true_values)) + 1e-6
    seller_scale = float(np.var(true_costs)) + 1e-6
    buyer_mse_rel = buyer_mse / buyer_scale
    seller_mse_rel = seller_mse / seller_scale
    buyer_quote_regret = float(mean(buyer_reg)) if buyer_reg else 0.0
    seller_quote_regret = float(mean(seller_reg)) if seller_reg else 0.0
    regret = buyer_quote_regret + seller_quote_regret
    trade_rate = sum(1 for tx in transactions if tx.traded) / len(transactions) if transactions else 0.0
    mean_buyer_utility = float(mean(buyer_util)) if buyer_util else 0.0
    mean_seller_utility = float(mean(seller_util)) if seller_util else 0.0
    mean_surplus = float(mean(surplus)) if surplus else 0.0
    if len(learning_curve) > 1:
        instability = float(np.std(np.diff(learning_curve)))
    else:
        instability = 0.0
    initial_error = learning_curve[0] if learning_curve else 0.0
    final_error = learning_curve[-1] if learning_curve else 0.0
    sample_efficiency = (
        float(1.0 - final_error / initial_error) if initial_error > 1e-9 else 0.0
    )
    objective = episode_objective(
        buyer_mse_rel=buyer_mse_rel,
        seller_mse_rel=seller_mse_rel,
        buyer_quote_regret=buyer_quote_regret,
        seller_quote_regret=seller_quote_regret,
        surplus=mean_surplus,
        instability=instability,
    )
    metrics = EpisodeMetrics(
        buyer_mse=buyer_mse,
        seller_mse=seller_mse,
        buyer_mse_rel=buyer_mse_rel,
        seller_mse_rel=seller_mse_rel,
        buyer_quote_regret=buyer_quote_regret,
        seller_quote_regret=seller_quote_regret,
        regret=regret,
        trade_rate=trade_rate,
        buyer_utility=mean_buyer_utility,
        seller_utility=mean_seller_utility,
        surplus=mean_surplus,
        instability=instability,
        sample_efficiency=sample_efficiency,
        objective=objective,
        llm_calls=llm.calls - start_calls,
        llm_failures=llm.failures - start_failures,
    )
    return metrics, transactions


@dataclass
class Diagnosis:
    round_idx: int
    tags: dict[str, float]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_idx,
            "tags": {k: round(float(v), 4) for k, v in self.tags.items()},
            "rationale": self.rationale,
        }


def diagnose(metrics: EpisodeMetrics, round_idx: int) -> Diagnosis:
    tags: dict[str, float] = {}
    reasons: list[str] = []
    if metrics.buyer_mse_rel > 0.1:
        tags["buyer_valuation"] = min(5.0, metrics.buyer_mse_rel)
        tags["demand"] = min(2.0, metrics.buyer_mse_rel)
        tags["calibration"] = max(tags.get("calibration", 0.0), min(2.0, metrics.buyer_mse_rel))
        if metrics.buyer_mse_rel > 0.8:
            tags["replay"] = max(tags.get("replay", 0.0), 0.8)
        reasons.append(f"buyer value rel-MSE {metrics.buyer_mse_rel:.3f}")
    if metrics.seller_mse_rel > 0.1:
        tags["seller_valuation"] = min(5.0, metrics.seller_mse_rel)
        tags["calibration"] = max(tags.get("calibration", 0.0), min(2.0, metrics.seller_mse_rel))
        if metrics.seller_mse_rel > 0.8:
            tags["replay"] = max(tags.get("replay", 0.0), 0.8)
        reasons.append(f"seller cost rel-MSE {metrics.seller_mse_rel:.3f}")
    if metrics.regret > 0.1:
        tags["quote"] = min(2.0, metrics.regret)
        reasons.append(f"quote regret {metrics.regret:.3f}")
    if metrics.trade_rate < 0.5:
        tags["exploration"] = (0.5 - metrics.trade_rate) * 2.0
        reasons.append(f"low trade rate {metrics.trade_rate:.3f}")
    if metrics.instability > 0.05:
        tags["stability"] = min(2.0, metrics.instability * 10.0)
        reasons.append(f"instability {metrics.instability:.3f}")
    if metrics.sample_efficiency < 0.2:
        tags["replay"] = 0.5
        reasons.append(f"low sample efficiency {metrics.sample_efficiency:.3f}")
    if not tags:
        tags["buyer_valuation"] = 0.2
        tags["quote"] = 0.2
        reasons.append("no strong failure; explore buyer valuation / quote")
    return Diagnosis(round_idx=round_idx, tags=tags, rationale="; ".join(reasons))


# ----------------------------------------------------------------------
# Meta-level
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
    mean_buyer_quote_regret: float
    mean_seller_quote_regret: float
    mean_regret: float
    mean_trade_rate: float
    mean_surplus: float
    mean_buyer_utility: float
    mean_seller_utility: float
    mean_sample_efficiency: float
    llm_calls: int
    llm_failures: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_version": self.program.version,
            "program_name": self.program.name,
            "ops": [op.kind for op in self.program.ops],
            "seeds": list(self.seeds),
            "mean_objective": round(self.mean_objective, 6),
            "std_objective": round(self.std_objective, 6),
            "lcb": round(self.lcb, 6),
            "mean_buyer_mse": round(self.mean_buyer_mse, 6),
            "mean_seller_mse": round(self.mean_seller_mse, 6),
            "mean_buyer_mse_rel": round(self.mean_buyer_mse_rel, 6),
            "mean_seller_mse_rel": round(self.mean_seller_mse_rel, 6),
            "mean_buyer_quote_regret": round(self.mean_buyer_quote_regret, 6),
            "mean_seller_quote_regret": round(self.mean_seller_quote_regret, 6),
            "mean_regret": round(self.mean_regret, 6),
            "mean_trade_rate": round(self.mean_trade_rate, 6),
            "mean_surplus": round(self.mean_surplus, 6),
            "mean_buyer_utility": round(self.mean_buyer_utility, 6),
            "mean_seller_utility": round(self.mean_seller_utility, 6),
            "mean_sample_efficiency": round(self.mean_sample_efficiency, 6),
            "llm_calls": int(self.llm_calls),
            "llm_failures": int(self.llm_failures),
        }


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def evaluate_program(
    program: UpdateProgram,
    *,
    seeds: Sequence[int],
    config: LLMMarketConfig,
    steps: int,
    llm: LLMBilateralClient,
) -> EvaluationResult:
    metrics = []
    for seed in seeds:
        episode_config = replace(config, seed=int(seed))
        m, _ = run_episode(program, episode_config, llm=llm, steps=steps)
        metrics.append(m)
    objectives = [m.objective for m in metrics]
    mean_obj = _mean(objectives)
    std_obj = _std(objectives)
    lcb = mean_obj - 1.96 * std_obj / math.sqrt(len(objectives)) if len(objectives) > 1 else mean_obj
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
        mean_buyer_quote_regret=_mean([m.buyer_quote_regret for m in metrics]),
        mean_seller_quote_regret=_mean([m.seller_quote_regret for m in metrics]),
        mean_regret=_mean([m.regret for m in metrics]),
        mean_trade_rate=_mean([m.trade_rate for m in metrics]),
        mean_surplus=_mean([m.surplus for m in metrics]),
        mean_buyer_utility=_mean([m.buyer_utility for m in metrics]),
        mean_seller_utility=_mean([m.seller_utility for m in metrics]),
        mean_sample_efficiency=_mean([m.sample_efficiency for m in metrics]),
        llm_calls=sum(m.llm_calls for m in metrics),
        llm_failures=sum(m.llm_failures for m in metrics),
    )


@dataclass
class MutationResult:
    mutation: MutationRecord
    candidate: UpdateProgram
    evaluation: EvaluationResult
    delta: float
    accepted: bool
    gain_lcb: float
    constraint_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation": self.mutation.to_dict(),
            "candidate": self.candidate.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "delta": round(self.delta, 6),
            "gain_lcb": round(self.gain_lcb, 6),
            "accepted": bool(self.accepted),
            "constraint_violations": list(self.constraint_violations),
        }


class LLMMetaController:
    def __init__(
        self,
        config: LLMMarketConfig,
        *,
        llm: LLMBilateralClient,
        meta_seeds: Sequence[int],
        meta_steps: int,
        candidates: int = 3,
        delta: float = 0.001,
        mutator_seed: int = 0,
        allowed_kinds: Sequence[str] | None = None,
        structural_only: bool = True,
        use_prompt_candidate: bool = True,
    ):
        self.config = config
        self.llm = llm
        self.meta_seeds = [int(s) for s in meta_seeds]
        self.meta_steps = int(meta_steps)
        self.candidates = int(candidates)
        self.delta = float(delta)
        self.mutator = ProgramMutator(mutator_seed, allowed_kinds=allowed_kinds)
        self.structural_only = bool(structural_only)
        self.use_prompt_candidate = bool(use_prompt_candidate)

    def step(
        self,
        program: UpdateProgram,
        round_idx: int,
    ) -> tuple[UpdateProgram, Diagnosis, list[MutationResult], MutationResult | None]:
        base_eval = evaluate_program(
            program,
            seeds=self.meta_seeds,
            config=self.config,
            steps=self.meta_steps,
            llm=self.llm,
        )
        diagnosis = diagnose(base_eval.metrics[0], round_idx)
        results: list[MutationResult] = []
        accepted: list[MutationResult] = []

        if self.structural_only:
            proposals = self.mutator.portfolio(program, diagnosis, self.candidates)
        else:
            proposals = self.mutator.parameter_portfolio(program, self.candidates)
        for candidate, mutation in proposals:
            evaluation = evaluate_program(
                candidate,
                seeds=self.meta_seeds,
                config=self.config,
                steps=self.meta_steps,
                llm=self.llm,
            )
            gains = [c.objective - b.objective for c, b in zip(evaluation.metrics, base_eval.metrics)]
            delta = _mean(gains)
            std = _std(gains)
            lcb = delta - 1.96 * std / math.sqrt(len(gains)) if len(gains) > 1 else delta
            accepted_flag = bool(lcb >= self.delta)
            result = MutationResult(mutation, candidate, evaluation, delta, accepted_flag, lcb)
            results.append(result)
            if accepted_flag:
                accepted.append(result)

        # A prompt-level structural candidate: the LLM rewrites the valuation
        # operator itself.  This is the "improve the operator that improves
        # valuation" layer.
        if self.use_prompt_candidate:
            addendum = self.llm.propose_prompt_addendum(
                side="buyer",
                diagnosis=diagnosis.rationale,
                current_addendum=program.buyer_prompt_addendum,
            )
            if addendum and addendum != program.buyer_prompt_addendum:
                candidate = program.copy_with(
                    list(program.ops),
                    version=program.version + 1,
                    name=f"{program.name}_prompt",
                    source="llm_prompt_revision",
                    rationale=f"LLM prompt revision: {addendum[:120]}",
                    buyer_prompt_addendum=addendum,
                    lineage=[f"prompt_addendum: {addendum[:160]}"],
                )
                evaluation = evaluate_program(
                    candidate,
                    seeds=self.meta_seeds,
                    config=self.config,
                    steps=self.meta_steps,
                    llm=self.llm,
                )
                gains = [c.objective - b.objective for c, b in zip(evaluation.metrics, base_eval.metrics)]
                delta = _mean(gains)
                std = _std(gains)
                lcb = delta - 1.96 * std / math.sqrt(len(gains)) if len(gains) > 1 else delta
                mutation = MutationRecord(
                    "prompt_revision",
                    -1,
                    "buyer_prompt_addendum",
                    {"addendum": addendum},
                    f"LLM prompt revision: {addendum[:120]}",
                    structural=True,
                )
                result = MutationResult(mutation, candidate, evaluation, delta, bool(lcb >= self.delta), lcb)
                results.append(result)
                if result.accepted:
                    accepted.append(result)

        if not accepted:
            return program, diagnosis, results, None
        accepted.sort(key=lambda r: (r.gain_lcb, r.delta), reverse=True)
        best = accepted[0]
        # Structural tie-break: if a multi-operator bundle is statistically
        # indistinguishable from the best single edit, prefer the bundle.  This
        # favours a genuinely new multi-op update program over a one-op tweak.
        bundles = [r for r in accepted if r.mutation.kind == "bundle"]
        if bundles:
            bundle = max(bundles, key=lambda r: (r.gain_lcb, r.delta))
            if (
                bundle.gain_lcb >= best.gain_lcb - 0.5
                and bundle.delta >= best.delta - 0.5
            ):
                best = bundle
        return best.candidate, diagnosis, results, best


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


@dataclass
class LLMRSISettings:
    seeds: tuple[int, ...] = (0,)
    meta_rounds: int = 3
    meta_steps: int = 6
    meta_seeds: int = 2
    candidates: int = 3
    delta: float = 0.001
    test_steps: int = 6
    test_seeds: int = 2
    test_shift: float = 0.3
    n_buyers: int = 2
    n_sellers: int = 2
    n_memories: int = 6
    max_llm_tokens: int = 600
    temperature: float = 0.0
    max_llm_calls: int = 1200
    model: str = "deepseek-chat"
    output_dir: str = str(Path(DEFAULT_RESULTS_DIR) / "llm_bilateral_rsi_v1")

    def market_config(self, seed: int) -> LLMMarketConfig:
        return LLMMarketConfig(
            seed=int(seed),
            n_buyers=self.n_buyers,
            n_sellers=self.n_sellers,
            n_memories=self.n_memories,
            n_steps=self.meta_steps,
            max_llm_tokens=self.max_llm_tokens,
            temperature=self.temperature,
        )


def seed_program() -> UpdateProgram:
    """Deliberately incomplete: buyer calibration only.

    The seller side, demand weights, source penalty, and quote learning are
    missing.  The meta-level must generate them.
    """

    return UpdateProgram(
        version=0,
        name="seed_llm_buyer_calibration_only",
        ops=[UpdateOp("update_buyer_demand_weights", {"lr": 0.03})],
        buyer_prompt_addendum="Value the memory by how well it covers the demand's required facets.",
        seller_prompt_addendum="Ask for the production cost plus a modest scarcity premium.",
        source="human",
        rationale="seed: only buyer demand-weight learning; calibration/seller/quote/source operators absent",
    )


class LLMBilateralRSIRunner:
    MODES = (
        "fixed",
        "meta_structural",
        "meta_param",
        "random_mutation",
        "self_validated",
        "closed_library",
    )

    def __init__(self, settings: LLMRSISettings | None = None, *, llm: LLMBilateralClient | None = None):
        self.settings = settings or LLMRSISettings()
        self.llm = llm or LLMBilateralClient(
            DeepSeekClient(
                LLMConfig(
                    model=self.settings.model,
                    max_tokens=self.settings.max_llm_tokens,
                    temperature=self.settings.temperature,
                    max_calls=self.settings.max_llm_calls,
                )
            ),
            max_tokens=self.settings.max_llm_tokens,
            temperature=self.settings.temperature,
        )

    def _controller(
        self,
        *,
        seed: int,
        structural_only: bool,
        allowed_kinds: Sequence[str] | None,
        use_prompt_candidate: bool = True,
    ) -> LLMMetaController:
        return LLMMetaController(
            self.settings.market_config(seed),
            llm=self.llm,
            meta_seeds=[seed + 100 + 37 * i for i in range(self.settings.meta_seeds)],
            meta_steps=self.settings.meta_steps,
            candidates=self.settings.candidates,
            delta=self.settings.delta,
            mutator_seed=seed * 10007 + 17,
            allowed_kinds=allowed_kinds,
            structural_only=structural_only,
            use_prompt_candidate=use_prompt_candidate,
        )

    def run_seed(self, seed: int, mode: str) -> dict[str, Any]:
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        settings = self.settings
        program = seed_program()
        rounds: list[dict[str, Any]] = []
        controller: LLMMetaController | None = None
        open_language = mode != "closed_library"
        if mode != "fixed":
            structural_only = mode in {"meta_structural", "closed_library", "self_validated", "random_mutation"}
            allowed_kinds = None
            if mode == "closed_library":
                allowed_kinds = (
                    "calibrate_buyer",
                    "update_buyer_demand_weights",
                    "update_buyer_quote",
                )
            controller = self._controller(
                seed=seed,
                structural_only=structural_only,
                allowed_kinds=allowed_kinds,
                use_prompt_candidate=(mode != "meta_param"),
            )

        for round_idx in range(settings.meta_rounds):
            base_metrics, _ = run_episode(
                program,
                settings.market_config(seed),
                llm=self.llm,
                steps=settings.meta_steps,
            )
            round_eval_seeds = [seed + 500 + 11 * i for i in range(max(2, settings.meta_seeds))]
            round_eval = evaluate_program(
                program,
                seeds=round_eval_seeds,
                config=settings.market_config(seed),
                steps=settings.meta_steps,
                llm=self.llm,
            )
            record: dict[str, Any] = {
                "round": round_idx,
                "program_version": program.version,
                "program_name": program.name,
                "ops": [op.kind for op in program.ops],
                "buyer_prompt_addendum": program.buyer_prompt_addendum,
                "base_objective": base_metrics.objective,
                "base_buyer_mse_rel": base_metrics.buyer_mse_rel,
                "base_seller_mse_rel": base_metrics.seller_mse_rel,
                "base_regret": base_metrics.regret,
                "round_heldout_objective": round_eval.mean_objective,
                "adopted": False,
                "diagnosis": None,
                "mutation": None,
                "candidate_results": [],
            }
            if controller is not None and mode != "fixed":
                if mode == "self_validated":
                    new_program, diagnosis, results, best = self._self_validated_step(
                        controller, program, round_idx, training_seed=seed
                    )
                elif mode == "random_mutation":
                    diagnosis = diagnose(base_metrics, round_idx)
                    candidate, mutation = controller.mutator.random_mutate(program)
                    evaluation = evaluate_program(
                        candidate,
                        seeds=controller.meta_seeds,
                        config=controller.config,
                        steps=controller.meta_steps,
                        llm=self.llm,
                    )
                    best = MutationResult(
                        mutation,
                        candidate,
                        evaluation,
                        evaluation.mean_objective - base_metrics.objective,
                        True,
                        evaluation.mean_objective - base_metrics.objective,
                    )
                    results = [best]
                    new_program = candidate
                else:
                    new_program, diagnosis, results, best = controller.step(program, round_idx)
                record["diagnosis"] = diagnosis.to_dict()
                record["candidate_results"] = [r.to_dict() for r in results]
                if best is not None:
                    program = new_program
                    record["adopted"] = True
                    record["adopted_program_version"] = program.version
                    record["adopted_ops"] = [op.kind for op in program.ops]
                    record["adopted_prompt"] = program.buyer_prompt_addendum
                    record["mutation"] = best.mutation.to_dict()
            rounds.append(record)

        # Held-out test on shifted markets.
        test_seeds = [seed + 900 + 13 * i for i in range(settings.test_seeds)]
        test_config = replace(settings.market_config(seed), shift=settings.test_shift)
        initial = seed_program()
        initial_eval = evaluate_program(initial, seeds=test_seeds, config=test_config, steps=settings.test_steps, llm=self.llm)
        final_eval = evaluate_program(program, seeds=test_seeds, config=test_config, steps=settings.test_steps, llm=self.llm)
        gains = [f.objective - i.objective for i, f in zip(initial_eval.metrics, final_eval.metrics)]
        gain = _mean(gains)
        gain_std = _std(gains)
        gain_lcb = gain - 1.96 * gain_std / math.sqrt(len(gains)) if len(gains) > 1 else gain
        structural_edits = sum(
            1 for r in rounds
            if r.get("adopted") and isinstance(r.get("mutation"), dict) and r["mutation"].get("structural")
        )
        reuse = {
            str(v): sum(1 for r in rounds if int(r["program_version"]) == int(v))
            for v in sorted({int(r["adopted_program_version"]) for r in rounds if r.get("adopted")})
        }
        sample_efficiency_gain = final_eval.mean_sample_efficiency - initial_eval.mean_sample_efficiency
        seed_ops = len(seed_program().ops)
        structural_ops_added = max(0, len(program.ops) - seed_ops)
        prompt_edits = sum(
            1
            for r in rounds
            if r.get("adopted")
            and isinstance(r.get("mutation"), dict)
            and r["mutation"].get("kind") == "prompt_revision"
        )
        meta_edits = structural_ops_added + prompt_edits
        certificate = {
            "n1_persistence": bool(len({r["program_version"] for r in rounds}) >= 2 and gain > 0),
            "n2_meta_level_generated": bool(
                open_language
                and meta_edits >= 2
                and (len(program.ops) >= 3 or prompt_edits >= 1)
            ),
            "n3_independent_validation": bool(gain_lcb >= 0.05),
            "n4_compounding": bool(sample_efficiency_gain > 0.05 and gain_lcb > 0),
            "n5_autonomy_over_criteria": False,
            "bounded": True,
            "open_language": bool(open_language),
            "structural_edits": int(structural_edits),
            "structural_ops_added": int(structural_ops_added),
            "prompt_edits": int(prompt_edits),
            "meta_edits": int(meta_edits),
            "sample_efficiency_gain": float(sample_efficiency_gain),
            "gain_lcb": float(gain_lcb),
            "min_gain_lcb": 0.05,
        }
        certificate["passed"] = bool(
            certificate["n1_persistence"]
            and certificate["n2_meta_level_generated"]
            and certificate["n3_independent_validation"]
            and certificate["n4_compounding"]
        )
        certificate["rsi_level"] = "bounded_rsi" if certificate["passed"] else "bounded_adaptation"
        return {
            "seed": int(seed),
            "mode": mode,
            "initial_program": initial.to_dict(),
            "final_program": program.to_dict(),
            "rounds": rounds,
            "heldout_test": {
                "seeds": test_seeds,
                "test_shift": settings.test_shift,
                "initial_objective": initial_eval.mean_objective,
                "final_objective": final_eval.mean_objective,
                "objective_gain": gain,
                "objective_gain_std": gain_std,
                "objective_gain_lcb": gain_lcb,
                "initial_buyer_mse_rel": initial_eval.mean_buyer_mse_rel,
                "final_buyer_mse_rel": final_eval.mean_buyer_mse_rel,
                "initial_seller_mse_rel": initial_eval.mean_seller_mse_rel,
                "final_seller_mse_rel": final_eval.mean_seller_mse_rel,
                "initial_regret": initial_eval.mean_regret,
                "final_regret": final_eval.mean_regret,
                "initial_trade_rate": initial_eval.mean_trade_rate,
                "final_trade_rate": final_eval.mean_trade_rate,
                "final_surplus": final_eval.mean_surplus,
                "initial_sample_efficiency": initial_eval.mean_sample_efficiency,
                "final_sample_efficiency": final_eval.mean_sample_efficiency,
                "sample_efficiency_gain": sample_efficiency_gain,
                "llm_calls": final_eval.llm_calls + initial_eval.llm_calls,
                "llm_failures": final_eval.llm_failures + initial_eval.llm_failures,
            },
            "reuse": reuse,
            "rsi_certificate": certificate,
        }

    def _self_validated_step(
        self,
        controller: LLMMetaController,
        program: UpdateProgram,
        round_idx: int,
        *,
        training_seed: int,
    ) -> tuple[UpdateProgram, Diagnosis, list[MutationResult], MutationResult | None]:
        base_eval = evaluate_program(
            program,
            seeds=[training_seed],
            config=controller.config,
            steps=controller.meta_steps,
            llm=self.llm,
        )
        diagnosis = diagnose(base_eval.metrics[0], round_idx)
        results: list[MutationResult] = []
        best: MutationResult | None = None
        best_utility = None
        for _ in range(controller.candidates):
            candidate, mutation = controller.mutator.mutate(program, diagnosis, structural=True)
            evaluation = evaluate_program(
                candidate,
                seeds=[training_seed],
                config=controller.config,
                steps=controller.meta_steps,
                llm=self.llm,
            )
            utility = (
                evaluation.mean_buyer_utility
                + evaluation.mean_seller_utility
                - base_eval.mean_buyer_utility
                - base_eval.mean_seller_utility
            )
            result = MutationResult(
                mutation,
                candidate,
                evaluation,
                evaluation.mean_objective - base_eval.mean_objective,
                utility > 0,
                utility,
            )
            results.append(result)
            if utility > 0 and (best_utility is None or utility > best_utility):
                best = result
                best_utility = utility
        return (best.candidate if best else program), diagnosis, results, best

    def run(self, modes: Sequence[str] | None = None) -> dict[str, Any]:
        modes = list(modes or self.MODES)
        results = [self.run_seed(int(seed), mode) for mode in modes for seed in self.settings.seeds]
        summary = self.aggregate(results)
        return {
            "settings": asdict(self.settings),
            "results": results,
            "summary": summary,
            "rsi_evidence": self.rsi_evidence(summary),
            "llm_usage": {
                "calls": self.llm.calls,
                "failures": self.llm.failures,
            },
        }

    @staticmethod
    def rsi_evidence(summary: dict[str, Any]) -> dict[str, Any]:
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
            "controls_present": all(name in summary for name in controls),
            "meta_pass_rate": float(meta.get("rsi_pass_rate", 0.0)) >= 0.8,
            "meta_gain_positive": meta_gain > 0.0,
            "meta_lcb_positive": float(meta.get("mean_gain_lcb", 0.0)) > 0.0,
            "beats_fixed": margins.get("fixed", 0.0) > 0.0,
            "beats_meta_param": margins.get("meta_param", 0.0) > 0.0,
            "beats_random_mutation": margins.get("random_mutation", 0.0) > 0.0,
            "beats_self_validated": margins.get("self_validated", 0.0) > 0.0,
            "beats_closed_library": margins.get("closed_library", 0.0) > 0.0,
        }
        return {
            "claim": "bounded_rsi" if all(checks.values()) else "bounded_adaptation",
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
                "bounded RSI requires N1-N4 per seed in >=80% of seeds and a "
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
            gains = [r["heldout_test"]["objective_gain"] for r in rows]
            lcbs = [r["heldout_test"]["objective_gain_lcb"] for r in rows]
            pass_rate = [1.0 if r["rsi_certificate"]["passed"] else 0.0 for r in rows]
            summary[mode] = {
                "n_seeds": len(rows),
                "mean_objective_gain": _mean(gains),
                "std_objective_gain": _std(gains),
                "mean_gain_lcb": _mean(lcbs),
                "rsi_pass_rate": _mean(pass_rate),
                "gains": [round(float(g), 6) for g in gains],
            }
        return summary

    def save(self, payload: dict[str, Any], output_dir: str | Path | None = None) -> Path:
        out = Path(output_dir or self.settings.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["mode", "seed", "objective_gain", "objective_gain_lcb", "rsi_level", "llm_calls", "llm_failures"])
            for r in payload["results"]:
                writer.writerow([
                    r["mode"], r["seed"], r["heldout_test"]["objective_gain"],
                    r["heldout_test"]["objective_gain_lcb"],
                    r["rsi_certificate"]["rsi_level"],
                    r["heldout_test"]["llm_calls"],
                    r["heldout_test"]["llm_failures"],
                ])
        (out / "report.md").write_text(self.render_report(payload), encoding="utf-8")
        return out

    @staticmethod
    def render_report(payload: dict[str, Any]) -> str:
        lines = [
            "# LLM-backed bilateral buyer--seller bounded RSI",
            "",
            "Buyer: demand -> LLM valuation -> quote. Seller: memory cost -> LLM valuation -> ask.",
            "The meta-level generates new update programs (including LLM prompt revisions),",
            "validates them on held-out meta-episodes, and versions/reuses the adopted program.",
            "",
            "| mode | seeds | mean gain | gain LCB | RSI pass |",
            "|---|---:|---:|---:|---:|",
        ]
        for mode, row in sorted(payload["summary"].items()):
            lines.append(
                f"| {mode} | {row['n_seeds']} | {row['mean_objective_gain']:.4f} | "
                f"{row['mean_gain_lcb']:.4f} | {row['rsi_pass_rate']:.2f} |"
            )
        evidence = payload.get("rsi_evidence", {})
        lines.extend([
            "",
            "## Aggregate RSI certificate",
            "",
            f"Claim: **{evidence.get('claim', 'unknown')}**",
            "",
            "| check | passed |",
            "|---|---:|",
        ])
        for name, value in evidence.get("checks", {}).items():
            lines.append(f"| {name} | {value} |")
        lines.extend([
            "",
            f"Meta pass rate: {evidence.get('meta_pass_rate', 0.0):.2f}; "
            f"mean gain: {evidence.get('meta_mean_gain', 0.0):.4f}; "
            f"LCB: {evidence.get('meta_mean_gain_lcb', 0.0):.4f}.",
            "",
            "## Per-seed final programs",
            "",
        ])
        for result in payload["results"]:
            if result["mode"] != "meta_structural":
                continue
            cert = result["rsi_certificate"]
            lines.append(
                f"- seed {result['seed']}: `{'>'.join(op['kind'] for op in result['final_program']['ops'])}` "
                f"gain={result['heldout_test']['objective_gain']:.3f} "
                f"LCB={result['heldout_test']['objective_gain_lcb']:.3f} "
                f"sample-eff={cert['sample_efficiency_gain']:.3f} "
                f"ops-added={cert.get('structural_ops_added', 0)} "
                f"certificate={cert['rsi_level']}"
            )
            lines.append(
                f"  - buyer prompt: {result['final_program'].get('buyer_prompt_addendum', '')}"
            )
        lines.extend([
            "",
            "## Round-by-round trace (meta_structural)",
            "",
            "| seed | round | base objective | held-out objective | adopted | mutation |",
            "|---:|---:|---:|---:|---|---|",
        ])
        for result in payload["results"]:
            if result["mode"] != "meta_structural":
                continue
            for rd in result["rounds"]:
                mutation = ""
                if isinstance(rd.get("mutation"), dict):
                    mutation = rd["mutation"].get("description", "")
                lines.append(
                    f"| {result['seed']} | {rd['round']} | "
                    f"{rd['base_objective']:.4f} | "
                    f"{rd['round_heldout_objective']:.4f} | "
                    f"{rd['adopted']} | {mutation} |"
                )
        lines.extend([
            "",
            "## LLM usage",
            "",
            f"- calls: {payload.get('llm_usage', {}).get('calls', 0)}",
            f"- failures: {payload.get('llm_usage', {}).get('failures', 0)}",
            "",
        ])
        return "\n".join(lines)
