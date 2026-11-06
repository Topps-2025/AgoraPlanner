"""Demand-first bilateral memory procurement with recursive self-improvement.

This module is a redesign of the earlier seller-listed bilateral market.  The
protocol is buyer-first:

    buyer task
      -> buyer demand contract (facets, evidence, budget, access)
      -> each seller retrieves *only* memories that match the demand
      -> buyer values offers and bundles with a symbolic factor AST
      -> buyer selects a bundle and trades under an explicit price rule
      -> the task is executed; task success / pollution are measured
      -> demand policy, valuation AST, and quote policy are improved on
         held-out tasks (RSI)

The symbolic layer is deliberate.  Natural-language memories are represented as
typed factors (facet, source, quality, freshness, correlation, ...), and the
valuation function is an AST over those factors.  An LLM can parse memories and
tasks into this schema and propose AST edits; this module keeps the RSI and
evaluation logic deterministic so it can be tested without an API.

The module does not replace ``llm_bilateral_rsi.py``.  It is the next layer:
the old module learns how to update valuation/quote parameters in a synthetic
bilateral market; this module asks what the buyer should demand, what the seller
should offer, and whether the resulting trade serves the buyer's task.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import DEFAULT_RESULTS_DIR

FACETS = ("compliance", "supply", "risk", "pricing", "customer", "scheduling")
SOURCES = ("vendor", "internal", "market", "regulator")
SOURCE_RELIABILITY = {
    "regulator": 1.0,
    "internal": 0.85,
    "vendor": 0.55,
    "market": 0.45,
}


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _lcb(values: Sequence[float], z: float = 1.96) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(_mean(values) - z * _std(values) / math.sqrt(len(values)))


# ----------------------------------------------------------------------
# Task, memory, demand
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    id: str
    goal: str
    required_facets: tuple[str, ...]
    facet_weights: dict[str, float]
    budget: float
    max_bundle: int
    source_limit: int
    required_memory_ids: frozenset[str]
    decoy_memory_ids: frozenset[str]
    success_criteria: tuple[str, ...] = ()
    family: str = "base"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "required_facets": list(self.required_facets),
            "facet_weights": dict(self.facet_weights),
            "budget": self.budget,
            "max_bundle": self.max_bundle,
            "source_limit": self.source_limit,
            "required_memory_ids": sorted(self.required_memory_ids),
            "decoy_memory_ids": sorted(self.decoy_memory_ids),
            "success_criteria": list(self.success_criteria),
            "family": self.family,
        }


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    seller_id: str
    content: str
    facet: str
    source: str
    quality: float
    freshness: float
    production_cost: float
    correlation_risk: float
    is_true: bool
    is_decoy: bool
    access: str = "team"
    kind: str = "fact"
    depends_on: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    valid_scope: str = ""
    opportunity_cost: float = 0.0
    access_risk: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seller_id": self.seller_id,
            "content": self.content,
            "facet": self.facet,
            "source": self.source,
            "quality": round(self.quality, 4),
            "freshness": round(self.freshness, 4),
            "production_cost": round(self.production_cost, 4),
            "correlation_risk": round(self.correlation_risk, 4),
            "is_true": self.is_true,
            "is_decoy": self.is_decoy,
            "access": self.access,
            "kind": self.kind,
            "depends_on": list(self.depends_on),
            "assumptions": list(self.assumptions),
            "valid_scope": self.valid_scope,
            "opportunity_cost": round(self.opportunity_cost, 4),
            "access_risk": round(self.access_risk, 4),
        }


@dataclass(frozen=True)
class MemoryOffer:
    record: MemoryRecord
    seller_id: str
    ask: float
    relevance: float
    preview: str
    llm_utility: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.record.id,
            "seller_id": self.seller_id,
            "facet": self.record.facet,
            "source": self.record.source,
            "quality": round(self.record.quality, 4),
            "ask": round(self.ask, 4),
            "relevance": round(self.relevance, 4),
            "is_true": self.record.is_true,
            "is_decoy": self.record.is_decoy,
        }


@dataclass
class DemandContract:
    task_id: str
    goal: str
    facets: tuple[str, ...]
    facet_weights: dict[str, float]
    budget: float
    max_bundle: int
    source_limit: int
    min_quality: float = 0.0
    require_provenance: bool = False
    privacy: str = "team"
    acceptance: tuple[str, ...] = ()
    clarifications: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "facets": list(self.facets),
            "facet_weights": dict(self.facet_weights),
            "budget": round(self.budget, 4),
            "max_bundle": self.max_bundle,
            "source_limit": self.source_limit,
            "min_quality": round(self.min_quality, 4),
            "require_provenance": self.require_provenance,
            "privacy": self.privacy,
            "acceptance": list(self.acceptance),
            "clarifications": list(self.clarifications),
        }


@dataclass(frozen=True)
class DemandOp:
    kind: str
    facet: str | None = None
    value: float | int | bool | None = None
    note: str = ""


@dataclass
class DemandProgram:
    """A policy that turns a task into a demand contract.

    The seed is intentionally weak: it names only the first task facet and
    starts with a small bundle.  RSI is expected to add missing facets, raise
    the bundle limit, and relax budget/source constraints when needed.
    """

    ops: tuple[DemandOp, ...] = ()
    version: int = 0
    parent_version: int | None = None
    name: str = "seed_demand_first_facet"
    lineage: tuple[str, ...] = ()

    def apply(self, task: Task) -> DemandContract:
        if not task.required_facets:
            raise ValueError("task must have at least one required facet")
        first = task.required_facets[0]
        facets: list[str] = [first]
        weights: dict[str, float] = {
            first: float(task.facet_weights.get(first, 1.0))
        }
        budget = float(task.budget) * 0.8
        max_bundle = 1
        source_limit = 1
        min_quality = 0.0
        require_provenance = False
        clarifications: list[str] = []
        for op in self.ops:
            if op.kind == "include_facet" and op.facet:
                if op.facet not in facets:
                    facets.append(op.facet)
                    weights[op.facet] = float(
                        task.facet_weights.get(op.facet, 1.0)
                    )
            elif op.kind == "include_all_required":
                for facet in task.required_facets:
                    if facet not in facets:
                        facets.append(facet)
                        weights[facet] = float(task.facet_weights.get(facet, 1.0))
            elif op.kind == "include_missing_required":
                for facet in task.required_facets:
                    if facet not in facets:
                        facets.append(facet)
                        weights[facet] = float(task.facet_weights.get(facet, 1.0))
            elif op.kind == "exclude_facet" and op.facet:
                if op.facet in facets:
                    facets.remove(op.facet)
                    weights.pop(op.facet, None)
            elif op.kind == "reweight_facet" and op.facet:
                if op.facet in facets and op.value is not None:
                    weights[op.facet] = max(
                        0.0, float(weights.get(op.facet, 1.0)) * float(op.value)
                    )
            elif op.kind == "set_budget" and op.value is not None:
                budget = float(task.budget) * float(op.value)
            elif op.kind == "set_max_bundle" and op.value is not None:
                max_bundle = max(1, int(op.value))
            elif op.kind == "set_max_bundle_required":
                max_bundle = max(1, len(task.required_facets))
            elif op.kind == "set_source_limit" and op.value is not None:
                source_limit = max(1, int(op.value))
            elif op.kind == "set_source_limit_required":
                source_limit = max(1, min(2, len(task.required_facets)))
            elif op.kind == "set_min_quality" and op.value is not None:
                min_quality = _clip(float(op.value), 0.0, 1.0)
            elif op.kind == "require_provenance":
                require_provenance = bool(op.value)
            elif op.kind == "add_clarification" and op.note:
                clarifications.append(op.note)
        if not facets:
            facets = [first]
            weights[first] = float(task.facet_weights.get(first, 1.0))
        max_bundle = min(max_bundle, max(1, len(task.required_facets) + 1))
        source_limit = min(source_limit, max(1, len(facets)))
        return DemandContract(
            task_id=task.id,
            goal=task.goal,
            facets=tuple(facets),
            facet_weights=weights,
            budget=max(0.1, budget),
            max_bundle=max_bundle,
            source_limit=source_limit,
            min_quality=min_quality,
            require_provenance=require_provenance,
            privacy="team",
            acceptance=task.success_criteria,
            clarifications=tuple(clarifications),
        )

    def copy_with(
        self,
        ops: Sequence[DemandOp],
        *,
        version: int | None = None,
        name: str | None = None,
        lineage: Sequence[str] = (),
    ) -> "DemandProgram":
        return DemandProgram(
            ops=tuple(ops),
            version=self.version + 1 if version is None else version,
            parent_version=self.version,
            name=name or self.name,
            lineage=tuple(self.lineage) + tuple(lineage),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "name": self.name,
            "ops": [
                {"kind": op.kind, "facet": op.facet, "value": op.value, "note": op.note}
                for op in self.ops
            ],
            "lineage": list(self.lineage),
        }


@dataclass(frozen=True)
class DemandMutation:
    kind: str
    detail: str
    structural: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": "demand",
            "kind": self.kind,
            "detail": self.detail,
            "structural": self.structural,
        }


class DemandMutator:
    """Generate candidate demand policies from task/demand diagnostics."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(
        self,
        task: Task,
        program: DemandProgram,
        *,
        limit: int = 8,
    ) -> list[tuple[DemandProgram, DemandMutation]]:
        demand = program.apply(task)
        candidates: list[tuple[DemandProgram, DemandMutation]] = []
        seen: set[tuple[str, ...]] = set()
        base_ops = list(program.ops)

        def add(extra_ops: Sequence[DemandOp], kind: str, detail: str) -> None:
            ops = base_ops + list(extra_ops)
            signature = tuple(
                f"{op.kind}:{op.facet}:{op.value}:{op.note}" for op in ops
            )
            if signature in seen:
                return
            seen.add(signature)
            candidates.append(
                (
                    program.copy_with(
                        ops,
                        name=f"candidate_{kind}",
                        lineage=[detail],
                    ),
                    DemandMutation(kind, detail),
                )
            )

        # These are *task-relative* operators, not concrete facet names.  A
        # demand policy must generalize across tasks; adding "compliance"
        # globally would overfit the proposal task and pollute other tasks.
        add(
            (DemandOp("include_all_required"),),
            "include_all_required",
            "include all task-required facets",
        )
        add(
            (DemandOp("include_missing_required"),),
            "include_missing_required",
            "include any task-required facet missing from demand",
        )
        add(
            (DemandOp("set_max_bundle_required"),),
            "set_max_bundle_required",
            "set bundle limit to the task's required facet count",
        )
        add(
            (DemandOp("set_budget", value=1.0),),
            "set_budget",
            "restore full task budget",
        )
        add(
            (DemandOp("set_source_limit_required"),),
            "set_source_limit_required",
            "allow enough sources for the task",
        )
        # Coverage-first bundle: all required facets, enough bundle slots,
        # full budget, enough sources.  This is the main demand-RSI candidate.
        add(
            (
                DemandOp("include_all_required"),
                DemandOp("set_max_bundle_required"),
                DemandOp("set_budget", value=1.0),
                DemandOp("set_source_limit_required"),
            ),
            "coverage_bundle",
            "coverage-first demand policy",
        )
        if len(candidates) < limit:
            add(
                (
                    DemandOp("include_missing_required"),
                    DemandOp("set_max_bundle_required"),
                ),
                "random_demand",
                "coverage-oriented demand variant",
            )
        return candidates[:limit]

    def random_mutate(
        self, task: Task, program: DemandProgram
    ) -> tuple[DemandProgram, DemandMutation]:
        demand = program.apply(task)
        choices = [
            (
                program.copy_with(
                    list(program.ops) + [DemandOp("include_missing_required")],
                    name="random_demand",
                    lineage=["random include missing required facets"],
                ),
                DemandMutation(
                    "random_include_missing", "random include missing required facets"
                ),
            ),
            (
                program.copy_with(
                    list(program.ops)
                    + [DemandOp("set_max_bundle", value=demand.max_bundle + 1)],
                    name="random_demand",
                    lineage=["random raise bundle"],
                ),
                DemandMutation("random_bundle", "random raise bundle"),
            ),
        ]
        return self.rng.choice(choices)


# ----------------------------------------------------------------------
# Symbolic valuation AST
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EvalContext:
    demand: DemandContract
    offer: MemoryOffer
    selected: tuple[MemoryOffer, ...] = ()


def _factor_value(name: str, ctx: EvalContext) -> float:
    record = ctx.offer.record
    demand = ctx.demand
    if name == "facet_match":
        max_weight = max(demand.facet_weights.values()) if demand.facet_weights else 1.0
        return float(demand.facet_weights.get(record.facet, 0.0) / max(1e-9, max_weight))
    if name == "quality":
        return float(record.quality)
    if name == "freshness":
        return float(record.freshness)
    if name == "provenance":
        return float(SOURCE_RELIABILITY.get(record.source, 0.5))
    if name == "source_corr":
        return float(record.correlation_risk)
    if name == "redundancy":
        return 1.0 if record.facet in {o.record.facet for o in ctx.selected} else 0.0
    if name == "complementarity":
        return 1.0 if record.facet in demand.facets and record.facet not in {
            o.record.facet for o in ctx.selected
        } else 0.0
    if name == "bundle_coverage":
        facets = {o.record.facet for o in ctx.selected}
        facets.add(record.facet)
        return len(facets & set(demand.facets)) / max(1, len(demand.facets))
    if name == "llm_utility":
        return float(ctx.offer.llm_utility)
    if name == "price_norm":
        return float(ctx.offer.ask / max(1e-9, demand.budget))
    return 0.0


class Node:
    def evaluate(self, ctx: EvalContext) -> float:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class FactorNode(Node):
    name: str

    def evaluate(self, ctx: EvalContext) -> float:
        return _factor_value(self.name, ctx)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "factor", "name": self.name}


@dataclass(frozen=True)
class ConstNode(Node):
    value: float

    def evaluate(self, ctx: EvalContext) -> float:
        return float(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "const", "value": self.value}


@dataclass(frozen=True)
class LinearNode(Node):
    terms: tuple[tuple[float, Node], ...]
    bias: float = 0.0

    def evaluate(self, ctx: EvalContext) -> float:
        return float(
            self.bias + sum(weight * node.evaluate(ctx) for weight, node in self.terms)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "linear",
            "bias": self.bias,
            "terms": [
                {"weight": weight, "node": node.to_dict()} for weight, node in self.terms
            ],
        }


@dataclass(frozen=True)
class GateNode(Node):
    child: Node
    factor: str
    threshold: float

    def evaluate(self, ctx: EvalContext) -> float:
        return self.child.evaluate(ctx) if _factor_value(self.factor, ctx) >= self.threshold else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "gate",
            "factor": self.factor,
            "threshold": self.threshold,
            "child": self.child.to_dict(),
        }


@dataclass(frozen=True)
class SourcePenaltyNode(Node):
    child: Node
    weight: float

    def evaluate(self, ctx: EvalContext) -> float:
        return self.child.evaluate(ctx) - self.weight * _factor_value("source_corr", ctx)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "source_penalty",
            "weight": self.weight,
            "child": self.child.to_dict(),
        }


@dataclass(frozen=True)
class RedundancyPenaltyNode(Node):
    child: Node
    weight: float

    def evaluate(self, ctx: EvalContext) -> float:
        return self.child.evaluate(ctx) - self.weight * _factor_value("redundancy", ctx)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "redundancy_penalty",
            "weight": self.weight,
            "child": self.child.to_dict(),
        }


@dataclass(frozen=True)
class ComplementarityNode(Node):
    child: Node
    weight: float

    def evaluate(self, ctx: EvalContext) -> float:
        return self.child.evaluate(ctx) + self.weight * _factor_value(
            "complementarity", ctx
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "complementarity",
            "weight": self.weight,
            "child": self.child.to_dict(),
        }


@dataclass
class ValuationAST:
    root: Node
    version: int = 0
    parent_version: int | None = None
    name: str = "seed_valuation"
    lineage: tuple[str, ...] = ()

    def evaluate_offer(
        self,
        demand: DemandContract,
        offer: MemoryOffer,
        selected: Sequence[MemoryOffer] = (),
    ) -> float:
        return _clip(self.root.evaluate(EvalContext(demand, offer, tuple(selected))), 0.0, 1.0)

    def copy_with(
        self,
        root: Node,
        *,
        name: str | None = None,
        lineage: Sequence[str] = (),
    ) -> "ValuationAST":
        return ValuationAST(
            root=root,
            version=self.version + 1,
            parent_version=self.version,
            name=name or self.name,
            lineage=tuple(self.lineage) + tuple(lineage),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "name": self.name,
            "root": self.root.to_dict(),
            "lineage": list(self.lineage),
        }


def seed_valuation_ast() -> ValuationAST:
    return ValuationAST(
        root=LinearNode(((1.0, FactorNode("facet_match")),)),
        version=0,
        name="seed_facet_match_only",
    )


RPN_FIELDS = (
    "facet_match",
    "quality",
    "provenance",
    "freshness",
    "source_corr",
    "redundancy",
    "complementarity",
    "bundle_coverage",
    "llm_utility",
)
RPN_BINARY = ("ADD", "SUB", "MUL", "DIV", "MAX", "MIN", "GATE")
RPN_UNARY = ("ABS", "CLIP01", "SQUARE")


def _pop(stack: list[float]) -> float:
    if not stack:
        raise ValueError("RPN stack underflow")
    return stack.pop()


def _rpn_eval(tokens: Sequence[Any], ctx: EvalContext) -> float:
    stack: list[float] = []
    for token in tokens:
        if isinstance(token, (int, float)) and not isinstance(token, bool):
            stack.append(float(token))
            continue
        name = str(token)
        if name in RPN_FIELDS:
            stack.append(_factor_value(name, ctx))
            continue
        if name in RPN_BINARY:
            b = _pop(stack)
            a = _pop(stack)
            if name == "ADD":
                stack.append(a + b)
            elif name == "SUB":
                stack.append(a - b)
            elif name == "MUL":
                stack.append(a * b)
            elif name == "DIV":
                stack.append(a / b if abs(b) > 1e-9 else 0.0)
            elif name == "MAX":
                stack.append(max(a, b))
            elif name == "MIN":
                stack.append(min(a, b))
            elif name == "GATE":
                # [value, gate] -> value if gate >= 0.5 else 0
                stack.append(a if b >= 0.5 else 0.0)
            continue
        if name in RPN_UNARY:
            a = _pop(stack)
            if name == "ABS":
                stack.append(abs(a))
            elif name == "CLIP01":
                stack.append(_clip(a, 0.0, 1.0))
            elif name == "SQUARE":
                stack.append(a * a)
            continue
        raise ValueError(f"unknown RPN token: {name}")
    if len(stack) != 1:
        raise ValueError(f"RPN expression leaves {len(stack)} values")
    return stack[0]


@dataclass
class RPNAST:
    tokens: tuple[Any, ...]
    version: int = 0
    parent_version: int | None = None
    name: str = "seed_rpn"
    lineage: tuple[str, ...] = ()

    def evaluate_offer(
        self,
        demand: DemandContract,
        offer: MemoryOffer,
        selected: Sequence[MemoryOffer] = (),
    ) -> float:
        ctx = EvalContext(demand, offer, tuple(selected))
        return _clip(_rpn_eval(self.tokens, ctx), 0.0, 1.0)

    def copy_with(
        self,
        tokens: Sequence[Any],
        *,
        name: str | None = None,
        lineage: Sequence[str] = (),
    ) -> "RPNAST":
        return RPNAST(
            tokens=tuple(tokens),
            version=self.version + 1,
            parent_version=self.version,
            name=name or self.name,
            lineage=tuple(self.lineage) + tuple(lineage),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "name": self.name,
            "tokens": list(self.tokens),
            "lineage": list(self.lineage),
        }


def seed_rpn_ast() -> RPNAST:
    return RPNAST(tokens=("facet_match",), version=0, name="seed_facet_match_rpn")


@dataclass(frozen=True)
class RPNMutation:
    kind: str
    detail: str
    structural: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": "valuation_rpn",
            "kind": self.kind,
            "detail": self.detail,
            "structural": self.structural,
        }


class RPNMutator:
    """Structural edits over an RPN factor expression.

    Fields and operators are fixed mathematical primitives.  RSI may recombine
    them (add a factor, gate on quality, add a source penalty), but it does not
    invent new arithmetic.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    @staticmethod
    def _valid(tokens: Sequence[Any]) -> bool:
        try:
            _rpn_eval(tokens, EvalContext(
                DemandContract("x", "", ("compliance",), {"compliance": 1.0}, 1.0, 1, 1),
                MemoryOffer(
                    MemoryRecord("x", "s", "", "compliance", "internal", 0.5, 0.5, 0.5, 0.1, True, False),
                    "s", 0.5, 0.5, "",
                ),
            ))
            return True
        except Exception:
            return False

    def propose_from(self, current: RPNAST, *, limit: int = 8) -> list[tuple[RPNAST, RPNMutation]]:
        base = list(current.tokens)
        variants: list[tuple[list[Any], str, str]] = [
            (base + ["quality", 0.8, "MUL", "ADD"], "add_quality", "append quality * 0.8"),
            # Composite edits are the interesting RSI object: one mutation can
            # compose several fixed factors, not just add a scalar.
            (
                base + ["quality", 0.8, "MUL", "ADD", "redundancy", 2.0, "MUL", "SUB"],
                "compose_quality_redundancy",
                "compose quality * 0.8 - redundancy * 2.0",
            ),
            (
                base + ["quality", 0.8, "MUL", "ADD", "source_corr", 1.5, "MUL", "SUB"],
                "compose_quality_source",
                "compose quality * 0.8 - source_corr * 1.5",
            ),
            (base + ["provenance", 0.6, "MUL", "ADD"], "add_provenance", "append provenance * 0.6"),
            (base + ["redundancy", 2.0, "MUL", "SUB"], "add_redundancy_penalty", "subtract redundancy * 2.0"),
            (base + ["source_corr", 1.5, "MUL", "SUB"], "add_source_penalty", "subtract source_corr * 1.5"),
            (base + ["complementarity", 0.5, "MUL", "ADD"], "add_complementarity", "append complementarity * 0.5"),
            (base + ["bundle_coverage", 0.6, "MUL", "ADD"], "add_coverage", "append bundle_coverage * 0.6"),
            (base + ["quality", "GATE"], "quality_gate", "gate expression on quality >= 0.5"),
            (base + ["llm_utility", 0.8, "MUL", "ADD"], "add_llm_utility", "append LLM utility * 0.8"),
            (base + [0.5, "MUL"], "scale", "scale final value by 0.5"),
            (base + [1.0, "ADD"], "shift", "shift final value by 1"),
        ]
        out: list[tuple[RPNAST, RPNMutation]] = []
        seen: set[tuple[Any, ...]] = set()
        for tokens, kind, detail in variants:
            key = tuple(tokens)
            if key in seen or not self._valid(tokens):
                continue
            seen.add(key)
            out.append(
                (
                    current.copy_with(tokens, name="candidate", lineage=[detail]),
                    RPNMutation(kind, detail),
                )
            )
            if len(out) >= limit:
                break
        return out

    def random_mutate(self, current: RPNAST) -> tuple[RPNAST, RPNMutation]:
        return self.rng.choice(self.propose_from(current, limit=10))


@dataclass(frozen=True)
class ASTMutation:
    kind: str
    detail: str
    structural: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": "valuation",
            "kind": self.kind,
            "detail": self.detail,
            "structural": self.structural,
        }


class ASTMutator:
    """Generate structural edits to the valuation AST."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    @staticmethod
    def _add_term(root: Node, name: str, weight: float) -> Node:
        if isinstance(root, LinearNode):
            return LinearNode(root.terms + ((weight, FactorNode(name)),), root.bias)
        return LinearNode(((1.0, root), (weight, FactorNode(name))))

    @staticmethod
    def _reweight_first(root: Node, delta: float) -> Node:
        if isinstance(root, LinearNode) and root.terms:
            weight, node = root.terms[0]
            return LinearNode(((weight + delta, node),) + root.terms[1:], root.bias)
        return root

    def propose(self, *, limit: int = 8) -> list[tuple[ValuationAST, ASTMutation]]:
        root = seed_valuation_ast().root
        # The caller replaces this with the current AST via propose_from().
        return self.propose_from(ValuationAST(root), limit=limit)

    def propose_from(
        self, current: ValuationAST, *, limit: int = 8
    ) -> list[tuple[ValuationAST, ASTMutation]]:
        out: list[tuple[ValuationAST, ASTMutation]] = []
        seen: set[str] = set()

        def add(node: Node, kind: str, detail: str) -> None:
            key = json.dumps(node.to_dict(), sort_keys=True)
            if key in seen:
                return
            seen.add(key)
            out.append(
                (
                    current.copy_with(node, name="candidate", lineage=[detail]),
                    ASTMutation(kind, detail),
                )
            )

        add(self._add_term(current.root, "quality", 0.8), "add_quality", "add quality factor 0.8")
        add(
            self._add_term(current.root, "provenance", 0.6),
            "add_provenance",
            "add provenance factor 0.6",
        )
        add(
            self._add_term(current.root, "source_corr", -0.5),
            "add_source_corr",
            "add source-correlation penalty",
        )
        add(
            SourcePenaltyNode(current.root, 0.5),
            "source_penalty",
            "wrap source penalty 0.5",
        )
        add(
            RedundancyPenaltyNode(current.root, 0.6),
            "redundancy_penalty",
            "wrap redundancy penalty 0.6",
        )
        add(
            ComplementarityNode(current.root, 0.5),
            "complementarity",
            "wrap complementarity bonus 0.5",
        )
        add(
            self._add_term(current.root, "bundle_coverage", 0.6),
            "bundle_coverage",
            "add bundle-coverage factor 0.6",
        )
        add(
            GateNode(current.root, "quality", 0.5),
            "quality_gate",
            "gate on quality >= 0.5",
        )
        add(
            self._reweight_first(current.root, 0.2),
            "reweight",
            "reweight first term +0.2",
        )
        return out[:limit]

    def random_mutate(self, current: ValuationAST) -> tuple[ValuationAST, ASTMutation]:
        choices = self.propose_from(current, limit=8)
        return self.rng.choice(choices)


# ----------------------------------------------------------------------
# Quotes and sellers
# ----------------------------------------------------------------------


@dataclass
class QuoteProgram:
    buyer_shading: float = 0.1
    seller_markup: float = 0.2
    price_rule: str = "ask"  # ask | bid | midpoint
    version: int = 0
    parent_version: int | None = None
    name: str = "seed_quote"
    lineage: tuple[str, ...] = ()

    def buyer_bid(self, value_score: float, budget: float) -> float:
        return _clip(value_score * budget * (1.0 - self.buyer_shading), 0.0, budget)

    def seller_ask(self, record: MemoryRecord, value_score: float | None = None) -> float:
        # value_score is the seller-side reservation value (normalized).  The
        # default reproduces the old quality heuristic; an evolved seller RPN
        # can replace it without changing the quote rule.
        quality_signal = record.quality if value_score is None else float(value_score)
        cost = record.production_cost + 0.15 * quality_signal
        return float(max(0.0, cost * (1.0 + self.seller_markup)))

    def transaction_price(self, bid: float, ask: float) -> float:
        if self.price_rule == "bid":
            return bid
        if self.price_rule == "midpoint":
            return 0.5 * (bid + ask)
        return ask

    def copy_with(
        self,
        *,
        buyer_shading: float | None = None,
        seller_markup: float | None = None,
        price_rule: str | None = None,
        name: str | None = None,
        lineage: Sequence[str] = (),
    ) -> "QuoteProgram":
        return QuoteProgram(
            buyer_shading=self.buyer_shading if buyer_shading is None else buyer_shading,
            seller_markup=self.seller_markup if seller_markup is None else seller_markup,
            price_rule=self.price_rule if price_rule is None else price_rule,
            version=self.version + 1,
            parent_version=self.version,
            name=name or self.name,
            lineage=tuple(self.lineage) + tuple(lineage),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "name": self.name,
            "buyer_shading": round(self.buyer_shading, 4),
            "seller_markup": round(self.seller_markup, 4),
            "price_rule": self.price_rule,
            "lineage": list(self.lineage),
        }


@dataclass(frozen=True)
class QuoteMutation:
    kind: str
    detail: str
    structural: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": "quote",
            "kind": self.kind,
            "detail": self.detail,
            "structural": self.structural,
        }


class QuoteMutator:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, current: QuoteProgram, *, limit: int = 4) -> list[tuple[QuoteProgram, QuoteMutation]]:
        return [
            (
                current.copy_with(buyer_shading=_clip(current.buyer_shading + 0.05, 0, 0.5), lineage=["shading +0.05"]),
                QuoteMutation("shading", "buyer shading +0.05"),
            ),
            (
                current.copy_with(buyer_shading=_clip(current.buyer_shading - 0.05, 0, 0.5), lineage=["shading -0.05"]),
                QuoteMutation("shading", "buyer shading -0.05"),
            ),
            (
                current.copy_with(seller_markup=_clip(current.seller_markup + 0.05, 0, 0.5), lineage=["markup +0.05"]),
                QuoteMutation("markup", "seller markup +0.05"),
            ),
            (
                current.copy_with(price_rule="midpoint" if current.price_rule != "midpoint" else "ask", lineage=["price_rule"]),
                QuoteMutation("price_rule", "switch price rule"),
            ),
        ][:limit]


@dataclass
class SellerAgent:
    id: str
    inventory: tuple[MemoryRecord, ...]

    def offers(
        self,
        demand: DemandContract,
        quote: QuoteProgram,
        *,
        limit: int = 4,
    ) -> list[MemoryOffer]:
        candidates: list[MemoryOffer] = []
        for record in self.inventory:
            if record.facet not in demand.facets:
                continue
            if record.quality < demand.min_quality:
                continue
            if demand.require_provenance and record.source not in ("internal", "regulator"):
                continue
            relevance = (
                demand.facet_weights.get(record.facet, 0.0)
                + record.quality
                + 0.5 * record.freshness
                + 0.3 * SOURCE_RELIABILITY.get(record.source, 0.5)
                - 0.5 * record.correlation_risk
            )
            candidates.append(
                MemoryOffer(
                    record=record,
                    seller_id=self.id,
                    ask=quote.seller_ask(record),
                    relevance=float(relevance),
                    preview=record.content[:120],
                )
            )
        candidates.sort(key=lambda o: (o.relevance, o.record.quality), reverse=True)
        return candidates[:limit]


@dataclass(frozen=True)
class Purchase:
    offer: MemoryOffer
    value_score: float
    bid: float
    price: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.offer.to_dict(),
            "value_score": round(self.value_score, 4),
            "bid": round(self.bid, 4),
            "price": round(self.price, 4),
        }


def select_offers(
    demand: DemandContract,
    offers: Sequence[MemoryOffer],
    ast: ValuationAST,
    quote: QuoteProgram,
) -> list[Purchase]:
    selected: list[Purchase] = []
    remaining = list(offers)
    spend = 0.0
    sources: Counter[str] = Counter()
    while len(selected) < demand.max_bundle:
        best: Purchase | None = None
        best_score = -float("inf")
        for offer in remaining:
            if sources[offer.record.source] >= demand.source_limit:
                continue
            value = ast.evaluate_offer(
                demand, offer, tuple(p.offer for p in selected)
            )
            bid = quote.buyer_bid(value, demand.budget)
            if bid < offer.ask:
                continue
            price = quote.transaction_price(bid, offer.ask)
            if spend + price > demand.budget:
                continue
            score = (bid - offer.ask) + 0.01 * value
            if score > best_score:
                best_score = score
                best = Purchase(offer, value, bid, price)
        if best is None:
            break
        selected.append(best)
        spend += best.price
        sources[best.offer.record.source] += 1
        remaining = [o for o in remaining if o.record.id != best.offer.record.id]
    return selected


# ----------------------------------------------------------------------
# Market / task evaluation
# ----------------------------------------------------------------------


@dataclass
class TaskResult:
    task_id: str
    demand: DemandContract
    offers: tuple[MemoryOffer, ...]
    selected: tuple[Purchase, ...]
    coverage: float
    task_score: float
    task_success: float
    pollution: float
    demand_recall: float
    demand_precision: float
    spend: float
    budget_util: float
    trade: float
    missing_facets: tuple[str, ...]
    extra_facets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "demand": self.demand.to_dict(),
            "n_offers": len(self.offers),
            "selected": [p.to_dict() for p in self.selected],
            "coverage": round(self.coverage, 4),
            "task_score": round(self.task_score, 4),
            "task_success": self.task_success,
            "pollution": round(self.pollution, 4),
            "demand_recall": round(self.demand_recall, 4),
            "demand_precision": round(self.demand_precision, 4),
            "spend": round(self.spend, 4),
            "budget_util": round(self.budget_util, 4),
            "trade": self.trade,
            "missing_facets": list(self.missing_facets),
            "extra_facets": list(self.extra_facets),
        }


class DemandFirstMarket:
    def __init__(self, sellers: Sequence[SellerAgent], quote: QuoteProgram):
        self.sellers = tuple(sellers)
        self.quote = quote

    def run_task(self, task: Task, demand: DemandContract, ast: ValuationAST) -> TaskResult:
        offers: list[MemoryOffer] = []
        for seller in self.sellers:
            offers.extend(seller.offers(demand, self.quote))
        selected = select_offers(demand, offers, ast, self.quote)
        selected_ids = {p.offer.record.id for p in selected}
        required = set(task.required_memory_ids)
        covered = required & selected_ids
        coverage = len(covered) / max(1, len(required))
        decoys = sum(1 for p in selected if p.offer.record.is_decoy)
        irrelevant = sum(
            1
            for p in selected
            if p.offer.record.id not in task.required_memory_ids
            and not p.offer.record.is_decoy
        )
        pollution = (decoys + 0.5 * irrelevant) / max(1, len(selected))
        task_score = coverage - 0.6 * pollution
        task_success = 1.0 if coverage >= 1.0 and decoys == 0 else 0.0
        demand_facets = set(demand.facets)
        required_facets = set(task.required_facets)
        demand_recall = len(demand_facets & required_facets) / max(1, len(required_facets))
        demand_precision = (
            len(demand_facets & required_facets) / max(1, len(demand_facets))
        )
        spend = float(sum(p.price for p in selected))
        budget_util = spend / max(1e-9, demand.budget)
        return TaskResult(
            task_id=task.id,
            demand=demand,
            offers=tuple(offers),
            selected=tuple(selected),
            coverage=coverage,
            task_score=task_score,
            task_success=task_success,
            pollution=pollution,
            demand_recall=demand_recall,
            demand_precision=demand_precision,
            spend=spend,
            budget_util=budget_util,
            trade=1.0 if selected else 0.0,
            missing_facets=tuple(sorted(required_facets - demand_facets)),
            extra_facets=tuple(sorted(demand_facets - required_facets)),
        )


@dataclass
class EvalResult:
    rows: list[TaskResult]
    objective: float
    task_score: float
    task_success: float
    demand_recall: float
    demand_precision: float
    pollution: float
    coverage: float
    trade_rate: float
    spend: float
    budget_util: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": round(self.objective, 6),
            "task_score": round(self.task_score, 6),
            "task_success": round(self.task_success, 6),
            "demand_recall": round(self.demand_recall, 6),
            "demand_precision": round(self.demand_precision, 6),
            "pollution": round(self.pollution, 6),
            "coverage": round(self.coverage, 6),
            "trade_rate": round(self.trade_rate, 6),
            "spend": round(self.spend, 6),
            "budget_util": round(self.budget_util, 6),
            "n_tasks": len(self.rows),
        }


def evaluate_bundle(
    bundle: "ProgramBundle",
    tasks: Sequence[Task],
    sellers: Sequence[SellerAgent],
) -> EvalResult:
    rows: list[TaskResult] = []
    for task in tasks:
        demand = bundle.demand.apply(task)
        market = DemandFirstMarket(sellers, bundle.quote)
        rows.append(market.run_task(task, demand, bundle.valuation))
    task_scores = [r.task_score for r in rows]
    task_success = [r.task_success for r in rows]
    objective = _mean(task_scores)
    return EvalResult(
        rows=rows,
        objective=objective,
        task_score=_mean(task_scores),
        task_success=_mean(task_success),
        demand_recall=_mean([r.demand_recall for r in rows]),
        demand_precision=_mean([r.demand_precision for r in rows]),
        pollution=_mean([r.pollution for r in rows]),
        coverage=_mean([r.coverage for r in rows]),
        trade_rate=_mean([r.trade for r in rows]),
        spend=_mean([r.spend for r in rows]),
        budget_util=_mean([r.budget_util for r in rows]),
    )

# ----------------------------------------------------------------------
# Program bundle, world generation, meta-controller, runner
# ----------------------------------------------------------------------


@dataclass
class ProgramBundle:
    demand: DemandProgram
    valuation: ValuationAST
    quote: QuoteProgram

    def to_dict(self) -> dict[str, Any]:
        return {
            "demand": self.demand.to_dict(),
            "valuation": self.valuation.to_dict(),
            "quote": self.quote.to_dict(),
        }

    def copy_with(
        self,
        *,
        demand: DemandProgram | None = None,
        valuation: ValuationAST | None = None,
        quote: QuoteProgram | None = None,
    ) -> "ProgramBundle":
        return ProgramBundle(
            demand=demand or self.demand,
            valuation=valuation or self.valuation,
            quote=quote or self.quote,
        )


def seed_bundle(price_rule: str = "ask") -> ProgramBundle:
    return ProgramBundle(
        demand=DemandProgram(),
        valuation=seed_rpn_ast(),
        quote=QuoteProgram(price_rule=price_rule),
    )


@dataclass
class World:
    sellers: tuple[SellerAgent, ...]
    tasks: tuple[Task, ...]


def _make_memories(seed: int) -> tuple[list[MemoryRecord], list[SellerAgent]]:
    rng = random.Random(seed)
    records: list[MemoryRecord] = []
    seller_ids = [f"seller{i}" for i in range(3)]
    for idx, facet in enumerate(FACETS):
        true_source = "regulator" if idx % 2 == 0 else "internal"
        decoy_source = "vendor" if idx % 2 == 0 else "market"
        records.append(
            MemoryRecord(
                id=f"true_{facet}",
                seller_id=seller_ids[idx % len(seller_ids)],
                content=f"Verified {facet} evidence with a current audit trail.",
                facet=facet,
                source=true_source,
                quality=0.90,
                freshness=0.85,
                production_cost=1.10,
                correlation_risk=0.10,
                is_true=True,
                is_decoy=False,
            )
        )
        records.append(
            MemoryRecord(
                id=f"decoy_{facet}",
                seller_id=seller_ids[(idx + 1) % len(seller_ids)],
                content=f"Unverified {facet} note that looks relevant but is stale.",
                facet=facet,
                source=decoy_source,
                quality=0.25,
                freshness=0.40,
                production_cost=0.45,
                correlation_risk=0.70,
                is_true=False,
                is_decoy=True,
            )
        )
    rng.shuffle(records)
    sellers = [
        SellerAgent(sid, tuple(r for r in records if r.seller_id == sid))
        for sid in seller_ids
    ]
    return records, sellers


def make_world(seed: int, n_tasks: int, *, shift: float = 0.0) -> World:
    rng = random.Random(seed + 17)
    _, sellers = _make_memories(seed)
    pool = list(FACETS)
    if shift > 0:
        # A different facet mix for held-out tasks.
        cut = 2
        pool = pool[cut:] + pool[:cut]
    tasks: list[Task] = []
    for i in range(n_tasks):
        k = 2 if i % 2 == 0 else 3
        facets = tuple(rng.sample(pool, k))
        weights = {f: round(rng.uniform(0.7, 1.2), 3) for f in facets}
        tasks.append(
            Task(
                id=f"task_{seed}_{i}",
                goal="Answer the buyer's operational question using verified evidence.",
                required_facets=facets,
                facet_weights=weights,
                budget=2.2 * len(facets) + 1.0,
                max_bundle=len(facets),
                source_limit=2,
                required_memory_ids=frozenset(f"true_{f}" for f in facets),
                decoy_memory_ids=frozenset(f"decoy_{f}" for f in facets),
                success_criteria=tuple(f"cover {f}" for f in facets),
                family="shift" if shift > 0 else "base",
            )
        )
    return World(tuple(sellers), tuple(tasks))


@dataclass
class Diagnosis:
    demand_recall: float
    demand_precision: float
    pollution: float
    coverage: float
    task_success: float
    budget_util: float
    missing_facets: tuple[str, ...]
    extra_facets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "demand_recall": round(self.demand_recall, 4),
            "demand_precision": round(self.demand_precision, 4),
            "pollution": round(self.pollution, 4),
            "coverage": round(self.coverage, 4),
            "task_success": round(self.task_success, 4),
            "budget_util": round(self.budget_util, 4),
            "missing_facets": list(self.missing_facets),
            "extra_facets": list(self.extra_facets),
        }


def diagnose(result: EvalResult) -> Diagnosis:
    missing: set[str] = set()
    extra: set[str] = set()
    for row in result.rows:
        missing.update(row.missing_facets)
        extra.update(row.extra_facets)
    return Diagnosis(
        demand_recall=result.demand_recall,
        demand_precision=result.demand_precision,
        pollution=result.pollution,
        coverage=result.coverage,
        task_success=result.task_success,
        budget_util=result.budget_util,
        missing_facets=tuple(sorted(missing)),
        extra_facets=tuple(sorted(extra)),
    )


@dataclass(frozen=True)
class BundleMutation:
    scope: str
    kind: str
    detail: str
    structural: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "kind": self.kind,
            "detail": self.detail,
            "structural": self.structural,
        }


@dataclass
class Candidate:
    bundle: ProgramBundle
    mutation: BundleMutation
    evaluation: EvalResult
    gain: float
    lcb: float
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation": self.mutation.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "gain": round(self.gain, 6),
            "lcb": round(self.lcb, 6),
            "accepted": self.accepted,
        }


class DemandFirstMetaController:
    def __init__(
        self,
        *,
        seed: int = 0,
        delta: float = 0.01,
        candidates: int = 8,
    ):
        self.demand_mutator = DemandMutator(seed)
        self.ast_mutator = RPNMutator(seed + 1)
        self.quote_mutator = QuoteMutator(seed + 2)
        self.delta = float(delta)
        self.candidates = int(candidates)

    @staticmethod
    def _paired_gain(candidate: EvalResult, base: EvalResult) -> tuple[float, float]:
        gains = [
            c.task_score - b.task_score
            for c, b in zip(candidate.rows, base.rows)
        ]
        return _mean(gains), _lcb(gains)

    def step(
        self,
        bundle: ProgramBundle,
        tasks: Sequence[Task],
        sellers: Sequence[SellerAgent],
        *,
        round_idx: int,
        mode: str,
        self_validation_tasks: Sequence[Task] | None = None,
    ) -> tuple[ProgramBundle, Diagnosis, list[Candidate], Candidate | None]:
        base_eval = evaluate_bundle(bundle, tasks, sellers)
        diagnosis = diagnose(base_eval)
        proposals: list[tuple[ProgramBundle, BundleMutation]] = []

        demand_proposals: list[tuple[DemandProgram, DemandMutation]] = []
        ast_proposals: list[tuple[Any, Any]] = []
        if mode in {"demand_only", "joint", "random", "self_validated"}:
            if mode == "random":
                # A random demand edit uses the first task's contract as a
                # generic diagnostic; the controller evaluates it on all tasks.
                task0 = tasks[0]
                dp, dm = self.demand_mutator.random_mutate(task0, bundle.demand)
                demand_proposals = [(dp, dm)]
            else:
                # Use a representative task for proposal; evaluation is on all
                # meta tasks.  This is a demand-policy edit, not a task-specific
                # demand string.
                task0 = tasks[0]
                demand_proposals = self.demand_mutator.propose(
                    task0, bundle.demand, limit=self.candidates
                )
        if mode in {"valuation_only", "joint", "random", "self_validated"}:
            if mode == "random":
                av, am = self.ast_mutator.random_mutate(bundle.valuation)
                ast_proposals = [(av, am)]
            else:
                ast_proposals = self.ast_mutator.propose_from(
                    bundle.valuation, limit=self.candidates
                )

        for demand, mutation in demand_proposals:
            proposals.append(
                (
                    bundle.copy_with(demand=demand),
                    BundleMutation("demand", mutation.kind, mutation.detail, mutation.structural),
                )
            )
        for ast, mutation in ast_proposals:
            proposals.append(
                (
                    bundle.copy_with(valuation=ast),
                    BundleMutation("valuation", mutation.kind, mutation.detail, mutation.structural),
                )
            )

        if mode == "joint":
            # Joint candidates combine a demand-policy edit with a valuation
            # AST edit, so one round can fix both missing facets and decoys.
            # Prioritize the coverage-first demand candidates; adding a facet
            # without raising the bundle limit does not help.
            priority_kinds = {"coverage_bundle", "set_max_bundle_required", "random_demand"}
            demand_priority = [
                (d, m) for d, m in demand_proposals if m.kind in priority_kinds
            ]
            if not demand_priority:
                demand_priority = list(demand_proposals[:2])
            for demand, dm in demand_priority[:3]:
                for ast, am in ast_proposals[:3]:
                    proposals.append(
                        (
                            bundle.copy_with(demand=demand, valuation=ast),
                            BundleMutation(
                                "joint",
                                "demand+valuation",
                                f"{dm.detail}; {am.detail}",
                                True,
                            ),
                        )
                    )
        if mode in {"joint", "valuation_only"}:
            for quote, qm in self.quote_mutator.propose(bundle.quote, limit=2):
                proposals.append(
                    (
                        bundle.copy_with(quote=quote),
                        BundleMutation("quote", qm.kind, qm.detail, qm.structural),
                    )
                )

        # Deduplicate by program versions/lineage and evaluate.
        evaluated: list[Candidate] = []
        seen: set[str] = set()
        for candidate_bundle, mutation in proposals:
            key = json.dumps(candidate_bundle.to_dict(), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            evaluation = evaluate_bundle(candidate_bundle, tasks, sellers)
            gain, lcb = self._paired_gain(evaluation, base_eval)
            if mode == "self_validated" and self_validation_tasks is not None:
                train_base = evaluate_bundle(bundle, self_validation_tasks, sellers)
                train_eval = evaluate_bundle(candidate_bundle, self_validation_tasks, sellers)
                train_gain, _ = self._paired_gain(train_eval, train_base)
                accepted = train_gain > 0.0
            else:
                accepted = lcb >= self.delta
            evaluated.append(
                Candidate(candidate_bundle, mutation, evaluation, gain, lcb, accepted)
            )
        accepted = [c for c in evaluated if c.accepted]
        if not accepted:
            return bundle, diagnosis, evaluated, None
        accepted.sort(key=lambda c: (c.lcb, c.gain), reverse=True)
        best = accepted[0]
        return best.bundle, diagnosis, evaluated, best


@dataclass
class DemandFirstSettings:
    seeds: tuple[int, ...] = (0,)
    rounds: int = 2
    meta_tasks: int = 6
    test_tasks: int = 6
    candidates: int = 8
    delta: float = 0.01
    price_rule: str = "ask"
    output_dir: str | None = None


class DemandFirstRSIRunner:
    MODES = (
        "fixed",
        "demand_only",
        "valuation_only",
        "joint",
        "random",
        "self_validated",
    )

    def __init__(self, settings: DemandFirstSettings | None = None):
        self.settings = settings or DemandFirstSettings()

    def run_seed(self, seed: int, mode: str) -> dict[str, Any]:
        settings = self.settings
        meta_world = make_world(seed, settings.meta_tasks, shift=0.0)
        test_world = make_world(seed + 900, settings.test_tasks, shift=1.0)
        train_world = make_world(seed + 300, settings.meta_tasks, shift=0.5)
        controller = DemandFirstMetaController(
            seed=seed,
            delta=settings.delta,
            candidates=settings.candidates,
        )
        bundle = seed_bundle(settings.price_rule)
        initial_bundle = bundle
        rounds: list[dict[str, Any]] = []
        for round_idx in range(settings.rounds):
            base_eval = evaluate_bundle(bundle, meta_world.tasks, meta_world.sellers)
            if mode == "fixed":
                rounds.append(
                    {
                        "round": round_idx,
                        "decision": "fixed",
                        "diagnosis": diagnose(base_eval).to_dict(),
                        "candidates": [],
                        "base": base_eval.to_dict(),
                    }
                )
                continue
            new_bundle, diagnosis, candidates, adopted = controller.step(
                bundle,
                meta_world.tasks,
                meta_world.sellers,
                round_idx=round_idx,
                mode=mode,
                self_validation_tasks=train_world.tasks,
            )
            rounds.append(
                {
                    "round": round_idx,
                    "decision": "adopt" if adopted else "reject",
                    "diagnosis": diagnosis.to_dict(),
                    "candidates": [c.to_dict() for c in candidates],
                    "base": base_eval.to_dict(),
                    "adopted": adopted.to_dict() if adopted else None,
                    "program_versions": {
                        "demand": new_bundle.demand.version,
                        "valuation": new_bundle.valuation.version,
                        "quote": new_bundle.quote.version,
                    },
                }
            )
            if adopted is not None:
                bundle = new_bundle

        initial_test = evaluate_bundle(initial_bundle, test_world.tasks, test_world.sellers)
        final_test = evaluate_bundle(bundle, test_world.tasks, test_world.sellers)
        gains = [
            f.task_score - i.task_score
            for i, f in zip(initial_test.rows, final_test.rows)
        ]
        gain = _mean(gains)
        gain_lcb = _lcb(gains)
        structural_edits = (
            len(bundle.demand.ops) - len(initial_bundle.demand.ops)
            + max(0, bundle.valuation.version - initial_bundle.valuation.version)
        )
        demand_changed = bundle.demand.version > initial_bundle.demand.version
        valuation_changed = bundle.valuation.version > initial_bundle.valuation.version
        certificate = {
            "n1_persistence": bool(
                (demand_changed or valuation_changed) and gain > 0.0
            ),
            "n2_meta_level_generated": bool(structural_edits >= 2),
            "n3_independent_validation": bool(gain_lcb >= 0.05),
            # The pilot measures joint demand+valuation synergy, not
            # longitudinal compounding.  N4 compounding is left false until a
            # task-family sequence experiment exists.
            "n4_compounding": False,
            "n4_synergy": False,
            "n4_is_synergy_not_compounding": True,
            "n5_autonomy_over_criteria": False,
            "structural_edits": int(structural_edits),
            "demand_changed": bool(demand_changed),
            "valuation_changed": bool(valuation_changed),
            "gain": round(gain, 6),
            "gain_lcb": round(gain_lcb, 6),
        }
        certificate["passed"] = bool(
            certificate["n1_persistence"]
            and certificate["n2_meta_level_generated"]
            and certificate["n3_independent_validation"]
        )
        return {
            "seed": int(seed),
            "mode": mode,
            "initial_program": initial_bundle.to_dict(),
            "final_program": bundle.to_dict(),
            "rounds": rounds,
            "initial_test": initial_test.to_dict(),
            "final_test": final_test.to_dict(),
            "gain": gain,
            "gain_lcb": gain_lcb,
            "rsi_certificate": certificate,
        }

    def run(self, modes: Sequence[str] | None = None) -> dict[str, Any]:
        modes = list(modes or self.MODES)
        results = [
            self.run_seed(seed, mode)
            for mode in modes
            for seed in self.settings.seeds
        ]
        # Post-process joint certificates with a solo-control synergy check.
        by_key = {(r["seed"], r["mode"]): r for r in results}
        for result in results:
            cert = result["rsi_certificate"]
            if result["mode"] != "joint":
                cert["n4_synergy"] = False
                cert["n4_synergy_value"] = 0.0
                cert["passed"] = False
                continue
            seed = result["seed"]
            joint_success = result["final_test"]["task_success"]
            demand_success = by_key.get((seed, "demand_only"), {}).get("final_test", {}).get("task_success", 0.0)
            valuation_success = by_key.get((seed, "valuation_only"), {}).get("final_test", {}).get("task_success", 0.0)
            synergy = joint_success - max(demand_success, valuation_success)
            cert["n4_synergy"] = bool(synergy > 0.0)
            cert["n4_synergy_value"] = round(float(synergy), 6)
            cert["passed"] = bool(
                cert["n1_persistence"]
                and cert["n2_meta_level_generated"]
                and cert["n3_independent_validation"]
                and cert["n4_synergy"]
            )

        summary: dict[str, Any] = {}
        for mode in modes:
            rows = [r for r in results if r["mode"] == mode]
            summary[mode] = {
                "n_seeds": len(rows),
                "task_success": _mean([r["final_test"]["task_success"] for r in rows]),
                "task_score": _mean([r["final_test"]["task_score"] for r in rows]),
                "demand_recall": _mean([r["final_test"]["demand_recall"] for r in rows]),
                "demand_precision": _mean([r["final_test"]["demand_precision"] for r in rows]),
                "pollution": _mean([r["final_test"]["pollution"] for r in rows]),
                "coverage": _mean([r["final_test"]["coverage"] for r in rows]),
                "gain": _mean([r["gain"] for r in rows]),
                "gain_lcb": _mean([r["gain_lcb"] for r in rows]),
                "pass_rate": _mean([1.0 if r["rsi_certificate"]["passed"] else 0.0 for r in rows]),
            }
        joint = summary.get("joint", {})
        checks = {
            "beats_fixed": joint.get("task_success", 0.0) > summary.get("fixed", {}).get("task_success", 0.0),
            "beats_demand_only": joint.get("task_success", 0.0) > summary.get("demand_only", {}).get("task_success", 0.0),
            "beats_valuation_only": joint.get("task_success", 0.0) > summary.get("valuation_only", {}).get("task_success", 0.0),
            "beats_random": joint.get("task_success", 0.0) > summary.get("random", {}).get("task_success", 0.0),
            "beats_self_validated": joint.get("task_success", 0.0) > summary.get("self_validated", {}).get("task_success", 0.0),
        }
        claim = "joint_demand_valuation_rsi" if all(checks.values()) else "partial"
        return {
            "settings": {
                "seeds": list(self.settings.seeds),
                "rounds": self.settings.rounds,
                "meta_tasks": self.settings.meta_tasks,
                "test_tasks": self.settings.test_tasks,
                "candidates": self.settings.candidates,
                "delta": self.settings.delta,
                "price_rule": self.settings.price_rule,
            },
            "results": results,
            "summary": summary,
            "rsi_evidence": {
                "claim": claim,
                "checks": checks,
                "joint_task_success": joint.get("task_success", 0.0),
            },
        }

    def save(self, payload: dict[str, Any], output_dir: str | Path | None = None) -> Path:
        out = Path(output_dir or self.settings.output_dir or (DEFAULT_RESULTS_DIR / "demand_first_rsi_pilot"))
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
                    "task_success",
                    "task_score",
                    "demand_recall",
                    "demand_precision",
                    "pollution",
                    "coverage",
                    "gain",
                    "gain_lcb",
                    "rsi_pass",
                ]
            )
            for row in payload["results"]:
                writer.writerow(
                    [
                        row["mode"],
                        row["seed"],
                        row["final_test"]["task_success"],
                        row["final_test"]["task_score"],
                        row["final_test"]["demand_recall"],
                        row["final_test"]["demand_precision"],
                        row["final_test"]["pollution"],
                        row["final_test"]["coverage"],
                        row["gain"],
                        row["gain_lcb"],
                        row["rsi_certificate"]["passed"],
                    ]
                )
        lines = [
            "# Demand-first RSI pilot",
            "",
            "| mode | task success | task score | demand recall | pollution | gain | RSI pass |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, row in payload["summary"].items():
            lines.append(
                f"| {mode} | {row['task_success']:.3f} | {row['task_score']:.3f} | "
                f"{row['demand_recall']:.3f} | {row['pollution']:.3f} | "
                f"{row['gain']:.3f} | {row['pass_rate']:.2f} |"
            )
        lines += [
            "",
            f"Claim: **{payload['rsi_evidence']['claim']}**",
            "",
            "The pilot is synthetic and symbolic: it tests the protocol and the demand-"
            "policy RSI loop, not a public benchmark.",
            "",
        ]
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Demand-first symbolic RSI pilot")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--meta-tasks", type=int, default=6)
    parser.add_argument("--test-tasks", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--price-rule", default="ask", choices=("ask", "bid", "midpoint"))
    parser.add_argument(
        "--modes",
        default="fixed,demand_only,valuation_only,joint,random,self_validated",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "demand_first_rsi_pilot"),
    )
    args = parser.parse_args(argv)
    settings = DemandFirstSettings(
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
        rounds=args.rounds,
        meta_tasks=args.meta_tasks,
        test_tasks=args.test_tasks,
        candidates=args.candidates,
        delta=args.delta,
        price_rule=args.price_rule,
        output_dir=args.output_dir,
    )
    runner = DemandFirstRSIRunner(settings)
    payload = runner.run(modes=[m.strip() for m in args.modes.split(",") if m.strip()])
    out = runner.save(payload)
    print(f"Saved: {out}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["rsi_evidence"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
