"""Long-horizon / multi-hop natural-language demand-first RSI pilot.

This module extends the single-hop NL demand-first pipeline with:

    TaskGraph -> SubTask -> EpisodeState -> step-level verifier

The toy diagnostic is a six-step symbolic derivation (chain rule).  It is not a
public benchmark; it exists to test whether the framework can carry memory and
results across dependent steps, and whether per-agent RSI has enough meta
subtasks to validate buyer/seller edits.

The decomposition is fixed and exogenous.  The allocator is a fixed round-robin
assignment embedded in the graph; learning task allocation is out of scope.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.config import DEFAULT_RESULTS_DIR, LLMConfig  # noqa: E402
from agora_planner.demand_first_rsi import (  # noqa: E402
    DemandContract,
    DemandOp,
    MemoryOffer,
    MemoryRecord,
    QuoteProgram,
    RPNAST,
    RPNMutator,
    SellerAgent,
    _lcb,
    _mean,
    select_offers,
)
from agora_planner.llm import DeepSeekClient, _load_dotenv  # noqa: E402
from agora_planner.public_pool import MemoryPool  # noqa: E402
from agora_planner.nl_demand_first_rsi import (  # noqa: E402
    DEFAULT_DEMAND_INSTRUCTION,
    DEFAULT_RETRIEVAL_INSTRUCTION,
    RETRIEVAL_VARIANTS,
    NLBuyerPolicy,
    NLSellerPolicy,
    NLWorkflow,
    SEExplorer,
    SellerRPNMutator,
    _buyer_value,
    _candidate_kind,
    _seller_offers,
    apply_demand_ops,
)


# ----------------------------------------------------------------------
# Task graph and episode state
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SubTask:
    id: str
    question: str
    gold_answer: str
    buyer_id: str
    facet: str
    required_memory_ids: frozenset[str]
    decoy_memory_ids: frozenset[str]
    depends_on: tuple[str, ...] = ()
    verifier: str = "exact"
    answer_format: str = "a mathematical expression or a number"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "buyer_id": self.buyer_id,
            "facet": self.facet,
            "required_memory_ids": sorted(self.required_memory_ids),
            "decoy_memory_ids": sorted(self.decoy_memory_ids),
            "depends_on": list(self.depends_on),
            "verifier": self.verifier,
        }


@dataclass(frozen=True)
class TaskGraph:
    id: str
    subtasks: tuple[SubTask, ...]
    final_subtask_id: str

    def ordered(self) -> list[SubTask]:
        remaining = list(self.subtasks)
        done: set[str] = set()
        out: list[SubTask] = []
        while remaining:
            ready = [t for t in remaining if set(t.depends_on) <= done]
            if not ready:
                raise ValueError(f"cycle in task graph {self.id}")
            for task in ready:
                out.append(task)
                done.add(task.id)
                remaining.remove(task)
        return out


class TaskAllocator(Protocol):
    """External task-assignment interface; not part of the memory RSI object."""

    name: str

    def assign(self, tasks: Sequence[SubTask], buyers: Sequence[str]) -> dict[str, str]:
        ...


class RoundRobinAllocator:
    """Fixed round-robin allocator used by the diagnostic."""

    name = "round_robin"

    def assign(self, tasks: Sequence[SubTask], buyers: Sequence[str]) -> dict[str, str]:
        if not buyers:
            raise ValueError("at least one buyer is required")
        return {task.id: buyers[i % len(buyers)] for i, task in enumerate(tasks)}


def apply_allocator(
    graph: TaskGraph, allocator: TaskAllocator, buyers: Sequence[str]
) -> TaskGraph:
    mapping = allocator.assign(graph.subtasks, buyers)
    return TaskGraph(
        id=graph.id,
        subtasks=tuple(replace(task, buyer_id=mapping[task.id]) for task in graph.subtasks),
        final_subtask_id=graph.final_subtask_id,
    )


@dataclass
class EpisodeState:
    results: dict[str, "StepResult"] = field(default_factory=dict)
    derived: list[MemoryRecord] = field(default_factory=list)

    def context(self, limit: int = 12) -> str:
        lines: list[str] = []
        for sid, result in self.results.items():
            lines.append(f"{sid} result: {result.answer}")
        for mem in self.derived[-limit:]:
            lines.append(f"derived memory {mem.id}: {mem.content}")
        return "\n".join(lines[-limit:])

    def dependency_coverage(self, task: SubTask) -> float:
        if not task.depends_on:
            return 1.0
        ok = 0
        for dep in task.depends_on:
            result = self.results.get(dep)
            if result is not None and result.success >= 1.0:
                ok += 1
        return ok / len(task.depends_on)


@dataclass
class StepResult:
    graph_id: str
    subtask_id: str
    buyer_id: str
    demand: str
    offers: tuple[MemoryOffer, ...]
    selected: tuple[MemoryOffer, ...]
    answer: str
    success: float
    coverage: float
    pollution: float
    dependency_coverage: float
    spend: float
    verifier: str

    @property
    def objective(self) -> float:
        return float(self.success - 0.5 * self.pollution)

    @property
    def offer_precision(self) -> float:
        return sum(1.0 for o in self.offers if not o.record.is_decoy) / max(1, len(self.offers))

    @property
    def retrieval_objective(self) -> float:
        return float(self.objective + 0.25 * self.offer_precision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "subtask_id": self.subtask_id,
            "buyer_id": self.buyer_id,
            "demand": self.demand,
            "n_offers": len(self.offers),
            "selected": [o.record.id for o in self.selected],
            "answer": self.answer,
            "success": round(self.success, 4),
            "coverage": round(self.coverage, 4),
            "pollution": round(self.pollution, 4),
            "dependency_coverage": round(self.dependency_coverage, 4),
            "spend": round(self.spend, 4),
            "verifier": self.verifier,
        }


@dataclass
class EpisodeResult:
    graph_id: str
    rows: list[StepResult]
    final_success: float
    success: float
    pollution: float
    objective: float
    dependency_coverage: float
    error_propagation_depth: float
    spend: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "final_success": round(self.final_success, 4),
            "success": round(self.success, 4),
            "pollution": round(self.pollution, 4),
            "objective": round(self.objective, 4),
            "dependency_coverage": round(self.dependency_coverage, 4),
            "error_propagation_depth": round(self.error_propagation_depth, 4),
            "spend": round(self.spend, 4),
            "rows": [r.to_dict() for r in self.rows],
        }


@dataclass
class LHEval:
    rows: list[StepResult]
    episodes: list[EpisodeResult]
    success: float
    pollution: float
    objective: float
    final_success: float
    dependency_coverage: float
    spend: float
    offer_precision: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": round(self.success, 4),
            "pollution": round(self.pollution, 4),
            "objective": round(self.objective, 4),
            "final_success": round(self.final_success, 4),
            "dependency_coverage": round(self.dependency_coverage, 4),
            "spend": round(self.spend, 4),
            "offer_precision": round(self.offer_precision, 4),
            "n_steps": len(self.rows),
            "episodes": [e.to_dict() for e in self.episodes],
        }


@dataclass
class LongHorizonWorld:
    meta_graph: TaskGraph
    test_graph: TaskGraph
    meta_sellers: tuple[SellerAgent, ...]
    test_sellers: tuple[SellerAgent, ...]
    buyer_ids: tuple[str, ...]
    pool: MemoryPool | None = None


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------


def _norm(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("−", "-").replace("×", "*").replace("·", "*")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".;,")
    return text


def _safe_sympy(expr: str) -> Any:
    expr = str(expr).strip()
    if len(expr) > 300 or "__" in expr or "lambda" in expr or "import" in expr:
        raise ValueError("unsafe expression")
    if "=" in expr:
        expr = expr.split("=")[-1]
    # Normalize common LaTeX fragments the LLM may emit despite instructions.
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = expr.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    expr = expr.replace("\\cdot", "*").replace("\\times", "*")
    for name in ("sin", "cos", "tan", "log", "ln", "exp", "sqrt"):
        expr = expr.replace("\\" + name, name)
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.replace("^", "**").replace("π", "pi")
    import sympy as sp  # local import: optional dependency at runtime
    from sympy.parsing.sympy_parser import (  # noqa: WPS433
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    # Allow LLM-style implicit multiplication such as "3x" or "2sin(x)".
    # Keep multi-letter statistics symbols (Sxx, Sxy) as single symbols.
    local_dict = {
        name: sp.Symbol(name)
        for name in ("n", "Sx", "Sy", "Sxx", "Sxy", "Syy", "a", "b", "x", "y")
    }
    return parse_expr(
        expr,
        local_dict=local_dict,
        transformations=standard_transformations + (implicit_multiplication_application,),
        evaluate=True,
    )


def verify_answer(answer: str, gold: str, mode: str = "exact") -> bool:
    if _norm(answer) == _norm(gold):
        return True
    if mode == "numeric":
        nums_a = re.findall(r"-?\d+(?:\.\d+)?", str(answer))
        nums_g = re.findall(r"-?\d+(?:\.\d+)?", str(gold))
        if nums_a and nums_g:
            return abs(float(nums_a[-1]) - float(nums_g[-1])) <= 1e-6
        return False
    if mode == "sympy":
        try:
            import sympy as sp

            return bool(sp.simplify(_safe_sympy(answer) - _safe_sympy(gold)) == 0)
        except Exception:
            return False
    return False


# ----------------------------------------------------------------------
# Step execution
# ----------------------------------------------------------------------


def _contract(task: SubTask, demand_ops: Sequence[DemandOp] = ()) -> DemandContract:
    contract = DemandContract(
        task_id=task.id,
        goal=task.question,
        facets=(task.facet,),
        facet_weights={task.facet: 1.0},
        budget=6.0,
        max_bundle=2,
        source_limit=2,
        acceptance=(task.gold_answer,),
    )
    return apply_demand_ops(contract, demand_ops)


def run_subtask(
    graph_id: str,
    task: SubTask,
    state: EpisodeState,
    buyer_policy: NLBuyerPolicy,
    seller_policies: dict[str, NLSellerPolicy],
    sellers: dict[str, SellerAgent],
    workflow: NLWorkflow,
    *,
    pool: MemoryPool | None = None,
    step_index: int = 0,
) -> StepResult:
    context = state.context()
    demand_text = workflow.demand(task, buyer_policy.demand_instruction, context=context)
    contract = _contract(task, buyer_policy.demand_ops)
    offers: list[MemoryOffer] = []
    for seller_id, seller in sellers.items():
        offers.extend(
            _seller_offers(
                seller,
                seller_policies[seller_id],
                demand_text,
                contract,
                buyer_policy.quote,
                workflow,
            )
        )
    # Public pool: reviewed reusable memory is free at the point of access and
    # is governed by review/embargo/diversity, not by bid/ask.
    if pool is not None:
        for entry in pool.available(step_index, facet=task.facet, limit=6):
            record = entry.memory
            if record.quality < contract.min_quality:
                continue
            if contract.require_provenance and record.source not in ("internal", "regulator"):
                continue
            offers.append(
                MemoryOffer(
                    record=record,
                    seller_id="pool",
                    ask=0.0,
                    relevance=float(entry.reputation),
                    preview=record.content[:120],
                )
            )
    # A scoped offer must match the demand facet.  The semantic reranker can
    # still choose among same-facet true/decoy memories; unrelated facets are
    # not allowed to leak in just because they are cheap.
    offers = [o for o in offers if o.record.facet == task.facet]
    # If a memory is both in the pool and on sale, the free pool copy wins.
    by_id: dict[str, MemoryOffer] = {}
    for offer in offers:
        old = by_id.get(offer.record.id)
        if old is None or offer.ask < old.ask:
            by_id[offer.record.id] = offer
    offers = list(by_id.values())
    offers = _buyer_value(offers, demand_text, workflow)
    selected = select_offers(contract, offers, buyer_policy.valuation, buyer_policy.quote)
    chosen = tuple(p.offer for p in selected)
    answer = workflow.answer_general(
        task.question,
        chosen,
        context=context,
        answer_format=task.answer_format,
    )
    selected_ids = {o.record.id for o in chosen}
    covered = set(task.required_memory_ids) & selected_ids
    coverage = len(covered) / max(1, len(task.required_memory_ids))
    pollution = sum(1 for o in chosen if o.record.is_decoy) / max(1, len(chosen))
    success = 1.0 if verify_answer(answer, task.gold_answer, task.verifier) and coverage >= 1.0 else 0.0
    return StepResult(
        graph_id=graph_id,
        subtask_id=task.id,
        buyer_id=task.buyer_id,
        demand=demand_text,
        offers=tuple(offers),
        selected=chosen,
        answer=answer,
        success=success,
        coverage=coverage,
        pollution=pollution,
        dependency_coverage=state.dependency_coverage(task),
        spend=float(sum(p.price for p in selected)),
        verifier=task.verifier,
    )


def run_graph(
    graph: TaskGraph,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    sellers: dict[str, SellerAgent],
    workflow: NLWorkflow,
    *,
    pool: MemoryPool | None = None,
) -> EpisodeResult:
    state = EpisodeState()
    rows: list[StepResult] = []
    for step_index, task in enumerate(graph.ordered()):
        row = run_subtask(
            graph.id,
            task,
            state,
            buyer_policies[task.buyer_id],
            seller_policies,
            sellers,
            workflow,
            pool=pool,
            step_index=step_index,
        )
        state.results[task.id] = row
        rows.append(row)
        if row.success >= 1.0:
            state.derived.append(
                MemoryRecord(
                    id=f"derived_{graph.id}_{task.id}",
                    seller_id="workspace",
                    content=f"{task.id} result: {row.answer}",
                    facet=task.facet,
                    source="derived",
                    quality=1.0,
                    freshness=1.0,
                    production_cost=0.0,
                    correlation_risk=0.05,
                    is_true=True,
                    is_decoy=False,
                    access="team",
                    kind="derived",
                    depends_on=task.depends_on,
                )
            )
    final = state.results.get(graph.final_subtask_id)
    final_success = final.success if final else 0.0
    return EpisodeResult(
        graph_id=graph.id,
        rows=rows,
        final_success=final_success,
        success=_mean([r.success for r in rows]),
        pollution=_mean([r.pollution for r in rows]),
        objective=_mean([r.objective for r in rows]),
        dependency_coverage=_mean([r.dependency_coverage for r in rows]),
        error_propagation_depth=float(sum(1 for r in rows if r.dependency_coverage < 1.0)),
        spend=_mean([r.spend for r in rows]),
    )


def evaluate_graphs(
    graphs: Sequence[TaskGraph],
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    sellers: dict[str, SellerAgent],
    workflow: NLWorkflow,
    *,
    pool: MemoryPool | None = None,
) -> LHEval:
    episodes = [
        run_graph(g, buyer_policies, seller_policies, sellers, workflow, pool=pool)
        for g in graphs
    ]
    rows = [r for ep in episodes for r in ep.rows]
    return LHEval(
        rows=rows,
        episodes=episodes,
        success=_mean([r.success for r in rows]),
        pollution=_mean([r.pollution for r in rows]),
        objective=_mean([r.objective for r in rows]),
        final_success=_mean([ep.final_success for ep in episodes]),
        dependency_coverage=_mean([r.dependency_coverage for r in rows]),
        spend=_mean([r.spend for r in rows]),
        offer_precision=_mean([r.offer_precision for r in rows]),
    )


# ----------------------------------------------------------------------
# Toy math diagnostic world
# ----------------------------------------------------------------------


def _math_specs(graph_id: str):
    if graph_id == "meta":
        return [
            ("s1", "Retrieve the current definition of f(x). Give the formula.", "x**3 + 2*x", "function_f",
             "Function f is f(x) = x**3 + 2*x; source: current calculus sheet.",
             "Function f is f(x) = x**3 - 2*x; stale revision.", (), "sympy"),
            ("s2", "Retrieve the current definition of g(x). Give the formula.", "sin(x)", "function_g",
             "Function g is g(x) = sin(x); source: current calculus sheet.",
             "Function g is g(x) = cos(x); stale revision.", (), "sympy"),
            ("s3", "Differentiate f(x) from s1. Give f'(x).", "3*x**2 + 2", "derivative_f",
             "Derivative rule: d/dx x^n = n*x^(n-1); d/dx (2*x) = 2.",
             "Derivative rule: d/dx x^n = n*x^(n+1); wrong rule.", ("s1",), "sympy"),
            ("s4", "Differentiate g(x) from s2. Give g'(x).", "cos(x)", "derivative_g",
             "Derivative rule: d/dx sin(x) = cos(x).",
             "Derivative rule: d/dx sin(x) = -cos(x); wrong sign.", ("s2",), "sympy"),
            ("s5", "Use the chain rule with f'(x) from s3 and g'(x) from s4 to give h'(x) for h(x)=f(g(x)).",
             "(3*sin(x)**2 + 2)*cos(x)", "chain_rule",
             "Chain rule: h'(x) = f'(g(x)) * g'(x).",
             "Chain rule: h'(x) = f'(x) * g'(x); missing inner derivative.", ("s3", "s4"), "sympy"),
            ("s6", "Evaluate h'(0) using the formula from s5. Give the number.", "2", "evaluate",
             "At x=0: sin(0)=0 and cos(0)=1.",
             "At x=0: sin(0)=1 and cos(0)=0; wrong values.", ("s5",), "numeric"),
        ]
    return [
        ("s1", "Retrieve the current definition of f(x). Give the formula.", "x**2 + 3*x", "function_f",
         "Function f is f(x) = x**2 + 3*x; source: current calculus sheet.",
         "Function f is f(x) = x**2 - 3*x; stale revision.", (), "sympy"),
        ("s2", "Retrieve the current definition of g(x). Give the formula.", "cos(x)", "function_g",
         "Function g is g(x) = cos(x); source: current calculus sheet.",
         "Function g is g(x) = sin(x); stale revision.", (), "sympy"),
        ("s3", "Differentiate f(x) from s1. Give f'(x).", "2*x + 3", "derivative_f",
         "Derivative rule: d/dx x^n = n*x^(n-1); d/dx (3*x) = 3.",
         "Derivative rule: d/dx x^n = n*x^(n+1); wrong rule.", ("s1",), "sympy"),
        ("s4", "Differentiate g(x) from s2. Give g'(x).", "-sin(x)", "derivative_g",
         "Derivative rule: d/dx cos(x) = -sin(x).",
         "Derivative rule: d/dx cos(x) = sin(x); wrong sign.", ("s2",), "sympy"),
        ("s5", "Use the chain rule with f'(x) from s3 and g'(x) from s4 to give h'(x) for h(x)=f(g(x)).",
         "(2*cos(x) + 3)*(-sin(x))", "chain_rule",
         "Chain rule: h'(x) = f'(g(x)) * g'(x).",
         "Chain rule: h'(x) = f'(x) * g'(x); missing inner derivative.", ("s3", "s4"), "sympy"),
        ("s6", "Evaluate h'(0) using the formula from s5. Give the number.", "0", "evaluate",
         "At x=0: cos(0)=1; f'(1)=5; sin(0)=0 so the product is 0.",
         "At x=0: cos(0)=0 and sin(0)=1; wrong values.", ("s5",), "numeric"),
    ]


def _build_graph_and_sellers(graph_id: str) -> tuple[TaskGraph, tuple[SellerAgent, ...]]:
    specs = _math_specs(graph_id)
    tasks: list[SubTask] = []
    memories: list[MemoryRecord] = []
    for idx, (sid, question, gold, facet, true_text, decoy_text, deps, verifier) in enumerate(specs):
        true_id = f"{graph_id}_{sid}_true"
        decoy_id = f"{graph_id}_{sid}_decoy"
        seller_true = "sellerA" if idx % 2 == 0 else "sellerB"
        seller_decoy = "sellerB" if seller_true == "sellerA" else "sellerA"
        memories.append(
            MemoryRecord(
                id=true_id,
                seller_id=seller_true,
                content=true_text,
                facet=facet,
                source="internal" if idx % 2 == 0 else "regulator",
                quality=0.92,
                freshness=0.9,
                production_cost=0.10,
                correlation_risk=0.1,
                is_true=True,
                is_decoy=False,
                kind="lemma",
                depends_on=tuple(f"{graph_id}_{d}" for d in deps),
                assumptions=(),
                valid_scope="current",
            )
        )
        memories.append(
            MemoryRecord(
                id=decoy_id,
                seller_id=seller_decoy,
                content=decoy_text,
                facet=facet,
                source="vendor" if idx % 2 == 0 else "market",
                quality=0.22,
                freshness=0.3,
                production_cost=0.02,
                correlation_risk=0.8,
                is_true=False,
                is_decoy=True,
                kind="lemma",
                depends_on=(),
                assumptions=("stale",),
                valid_scope="superseded",
            )
        )
        tasks.append(
            SubTask(
                id=f"{graph_id}_{sid}",
                question=question,
                gold_answer=gold,
                buyer_id="buyer0" if idx % 2 == 0 else "buyer1",
                facet=facet,
                required_memory_ids=frozenset({true_id}),
                decoy_memory_ids=frozenset({decoy_id}),
                depends_on=tuple(f"{graph_id}_{d}" for d in deps),
                verifier=verifier,
                answer_format="a mathematical expression or a number",
            )
        )
    graph = TaskGraph(id=graph_id, subtasks=tuple(tasks), final_subtask_id=f"{graph_id}_s6")
    sellers = tuple(
        SellerAgent(sid, tuple(m for m in memories if m.seller_id == sid))
        for sid in ("sellerA", "sellerB")
    )
    return graph, sellers


def build_math_diagnostic_world() -> LongHorizonWorld:
    meta_graph, meta_sellers = _build_graph_and_sellers("meta")
    test_graph, test_sellers = _build_graph_and_sellers("test")
    return LongHorizonWorld(
        meta_graph=meta_graph,
        test_graph=test_graph,
        meta_sellers=meta_sellers,
        test_sellers=test_sellers,
        buyer_ids=("buyer0", "buyer1"),
    )


# ----------------------------------------------------------------------
# Per-agent RSI over long-horizon graphs
# ----------------------------------------------------------------------


@dataclass
class LHCandidate:
    scope: str
    detail: str
    buyer_policies: dict[str, NLBuyerPolicy]
    seller_policies: dict[str, NLSellerPolicy]
    evaluation: LHEval
    gain: float
    lcb: float
    accepted: bool


class LongHorizonRSIRunner:
    def __init__(self, settings: dict[str, Any], workflow: NLWorkflow):
        self.settings = settings
        self.workflow = workflow
        self.mutator = RPNMutator(int(settings.get("seed", 0)) + 11)
        self.seller_mutator = SellerRPNMutator(int(settings.get("seed", 0)) + 12)
        self.se_buyer: dict[str, SEExplorer] = {}
        self.se_seller: dict[str, SEExplorer] = {}

    @staticmethod
    def _buyer_gains(candidate: LHEval, base: LHEval, buyer_id: str) -> tuple[float, float]:
        base_by = {r.subtask_id: r for r in base.rows if r.buyer_id == buyer_id}
        gains = [
            r.objective - base_by[r.subtask_id].objective
            for r in candidate.rows
            if r.buyer_id == buyer_id and r.subtask_id in base_by
        ]
        return _mean(gains), _lcb(gains)

    @staticmethod
    def _seller_gains(candidate: LHEval, base: LHEval) -> tuple[float, float]:
        base_by = {r.subtask_id: r for r in base.rows}
        gains = [
            r.retrieval_objective - base_by[r.subtask_id].retrieval_objective
            for r in candidate.rows
            if r.subtask_id in base_by
        ]
        return _mean(gains), _lcb(gains)

    def _buyer_candidates(self, policy: NLBuyerPolicy, limit: int = 5):
        out = []
        # Demand is a lossy compression; reserve candidates for editing the
        # demand contract itself, not only the RPN valuation.
        rpn_limit = max(1, limit - 2)
        for ast, mutation in self.mutator.propose_from(policy.valuation, limit=rpn_limit):
            out.append(
                (
                    policy.copy_with(valuation=ast, lineage=[mutation.detail]),
                    "valuation_rpn",
                    mutation.detail,
                )
            )
        demand_ops = tuple(policy.demand_ops) + (
            DemandOp("require_provenance", value=True),
            DemandOp("set_min_quality", value=0.5),
        )
        if not any(
            op.kind == "require_provenance" and op.value is True
            for op in policy.demand_ops
        ):
            out.append(
                (
                    policy.copy_with(
                        demand_ops=demand_ops,
                        lineage=["demand: require provenance + min_quality 0.5"],
                    ),
                    "demand_ops",
                    "require provenance + quality floor 0.5",
                )
            )
            rpn_choices = self.mutator.propose_from(policy.valuation, limit=3)
            combo_ast = next(
                (ast for ast, mutation in rpn_choices if "compose" in mutation.detail),
                rpn_choices[0][0] if rpn_choices else None,
            )
            if combo_ast is not None:
                out.append(
                    (
                        policy.copy_with(
                            valuation=combo_ast,
                            demand_ops=demand_ops,
                            lineage=[
                                "demand: require provenance + min_quality 0.5",
                                "RPN: compose quality/source/redundancy",
                            ],
                        ),
                        "demand_rpn",
                        "joint demand policy + RPN composition",
                    )
                )
        return out[:limit]

    def _seller_candidates(self, policy: NLSellerPolicy, limit: int = 6):
        out: list[tuple[NLSellerPolicy, str, str]] = []
        # Seller RSI mirrors buyer valuation: seller RPN over fixed cost/risk
        # fields, plus retrieval policy.  Ask is a fixed mapping from V_j^s.
        for ast, detail in self.seller_mutator.propose_from(policy.valuation, limit=2):
            out.append(
                (
                    policy.copy_with(valuation=ast, lineage=[detail]),
                    "seller_valuation_rpn",
                    detail,
                )
            )
        for instr in RETRIEVAL_VARIANTS[:2]:
            if instr == policy.retrieval_instruction:
                continue
            out.append(
                (
                    policy.copy_with(
                        retrieval_instruction=instr,
                        lineage=[f"retrieval: {instr[:80]}"],
                    ),
                    "retrieval_instruction",
                    instr[:80],
                )
            )
        if policy.min_quality < 0.5:
            out.append(
                (
                    policy.copy_with(min_quality=0.5, lineage=["min_quality -> 0.50"]),
                    "min_quality",
                    "min_quality -> 0.50",
                )
            )
        tk = max(1, policy.top_k - 1)
        if tk != policy.top_k:
            out.append(
                (
                    policy.copy_with(top_k=tk, lineage=[f"top_k -> {tk}"]),
                    "top_k",
                    f"top_k -> {tk}",
                )
            )
        return out[:limit]

    def run_seed(self, seed: int, mode: str) -> dict[str, Any]:
        world = build_math_diagnostic_world()
        allocator_name = str(self.settings.get("allocator", "round_robin"))
        if allocator_name != "round_robin":
            raise ValueError(f"unsupported allocator: {allocator_name}")
        allocator = RoundRobinAllocator()
        world.meta_graph = apply_allocator(world.meta_graph, allocator, world.buyer_ids)
        world.test_graph = apply_allocator(world.test_graph, allocator, world.buyer_ids)
        buyer_policies = {
            bid: NLBuyerPolicy(
                DEFAULT_DEMAND_INSTRUCTION,
                RPNAST(("llm_utility", "facet_match", "MUL"), version=0, name="seed_llm_facet"),
                QuoteProgram(price_rule="ask"),
            )
            for bid in world.buyer_ids
        }
        seller_policies = {
            s.id: NLSellerPolicy(DEFAULT_RETRIEVAL_INSTRUCTION, top_k=6)
            for s in world.meta_sellers
        }
        meta_sellers = {s.id: s for s in world.meta_sellers}
        test_sellers = {s.id: s for s in world.test_sellers}
        initial_buyer = {k: v.copy_with() for k, v in buyer_policies.items()}
        initial_seller = {k: v.copy_with() for k, v in seller_policies.items()}
        self.se_buyer = {bid: SEExplorer() for bid in buyer_policies}
        self.se_seller = {sid: SEExplorer() for sid in seller_policies}

        rounds: list[dict[str, Any]] = []
        if mode == "fixed":
            rounds.append({"round": 0, "decision": "fixed"})
        elif mode == "se":
            # SE-style baseline: tune scalar parameters inside the fixed
            # policy structure.  It can change buyer shading and seller top-k,
            # but cannot add demand ops, RPN compositions, or retrieval
            # instruction structures.
            grid: list[dict[str, Any]] = []
            best = None
            for shading in (0.0, 0.1, 0.3, 0.5):
                for top_k in (3, 6):
                    trial_buyer = {
                        bid: policy.copy_with(
                            quote=QuoteProgram(
                                buyer_shading=shading,
                                seller_markup=0.2,
                                price_rule="ask",
                            ),
                            lineage=[f"se: shading={shading}"],
                        )
                        for bid, policy in buyer_policies.items()
                    }
                    trial_seller = {
                        sid: policy.copy_with(
                            top_k=top_k,
                            lineage=[f"se: top_k={top_k}"],
                        )
                        for sid, policy in seller_policies.items()
                    }
                    evaluation = evaluate_graphs(
                        [world.meta_graph], trial_buyer, trial_seller, meta_sellers, self.workflow
                    )
                    grid.append(
                        {
                            "shading": shading,
                            "top_k": top_k,
                            "objective": evaluation.objective,
                        }
                    )
                    if best is None or evaluation.objective > best[0]:
                        best = (evaluation.objective, shading, top_k)
            if best is not None:
                _, shading, top_k = best
                buyer_policies = {
                    bid: policy.copy_with(
                        quote=QuoteProgram(
                            buyer_shading=shading,
                            seller_markup=0.2,
                            price_rule="ask",
                        ),
                        lineage=[f"se: shading={shading}"],
                    )
                    for bid, policy in buyer_policies.items()
                }
                seller_policies = {
                    sid: policy.copy_with(
                        top_k=top_k,
                        lineage=[f"se: top_k={top_k}"],
                    )
                    for sid, policy in seller_policies.items()
                }
            rounds.append(
                {
                    "round": 0,
                    "mode": "se_parameter_tuning",
                    "grid": grid,
                    "selected": best,
                }
            )
        else:
            round_record: dict[str, Any] = {"round": 0, "buyer_updates": [], "seller_updates": []}
            meta_base = evaluate_graphs(
                [world.meta_graph], buyer_policies, seller_policies, meta_sellers, self.workflow
            )
            for buyer_id, policy in list(buyer_policies.items()):
                base = meta_base
                candidates = self._buyer_candidates(policy)
                ordered = self.se_buyer[buyer_id].order(
                    [(c, _candidate_kind(c[1], c[2])) for c in candidates]
                )
                evaluated: list[LHCandidate] = []
                for (candidate, scope, detail), kind in ordered:
                    trial = dict(buyer_policies)
                    trial[buyer_id] = candidate
                    evaluation = evaluate_graphs(
                        [world.meta_graph], trial, seller_policies, meta_sellers, self.workflow
                    )
                    gain, lcb = self._buyer_gains(evaluation, base, buyer_id)
                    accepted = lcb > float(self.settings.get("delta", 0.0))
                    self.se_buyer[buyer_id].update(kind, gain, accepted)
                    evaluated.append(
                        LHCandidate(scope, detail, trial, dict(seller_policies), evaluation, gain, lcb, accepted)
                    )
                accepted_cands = [c for c in evaluated if c.accepted]
                if accepted_cands:
                    def _priority(scope: str) -> int:
                        if scope == "demand_rpn":
                            return 2
                        if scope == "demand_ops":
                            return 1
                        return 0

                    accepted_cands.sort(
                        key=lambda c: (c.lcb, c.gain, _priority(c.scope)),
                        reverse=True,
                    )
                    best = accepted_cands[0]
                    buyer_policies = best.buyer_policies
                    round_record["buyer_updates"].append(
                        {"buyer": buyer_id, "detail": best.detail, "gain": best.gain, "lcb": best.lcb}
                    )

            # Seller RSI is validated against the initial buyer reference so
            # seller edits cannot free-ride on the buyer's accepted edit.
            seller_base = evaluate_graphs(
                [world.meta_graph], initial_buyer, seller_policies, meta_sellers, self.workflow
            )
            for seller_id, policy in list(seller_policies.items()):
                candidates = self._seller_candidates(policy)
                ordered = self.se_seller[seller_id].order(
                    [(c, _candidate_kind(c[1], c[2])) for c in candidates]
                )
                best_seller = None
                for (candidate, scope, detail), kind in ordered:
                    trial = dict(seller_policies)
                    trial[seller_id] = candidate
                    evaluation = evaluate_graphs(
                        [world.meta_graph], initial_buyer, trial, meta_sellers, self.workflow
                    )
                    gain, lcb = self._seller_gains(evaluation, seller_base)
                    accepted = lcb > float(self.settings.get("delta", 0.0))
                    self.se_seller[seller_id].update(kind, gain, accepted)
                    cand = LHCandidate(scope, detail, dict(buyer_policies), trial, evaluation, gain, lcb, accepted)
                    def _seller_priority(scope: str) -> int:
                        if scope == "seller_valuation_rpn":
                            return 3
                        if scope == "retrieval_instruction":
                            return 2
                        if scope == "top_k":
                            return 1
                        return 0

                    if cand.accepted and (
                        best_seller is None
                        or (cand.lcb, cand.gain, _seller_priority(scope))
                        > (best_seller.lcb, best_seller.gain, _seller_priority(best_seller.scope))
                    ):
                        best_seller = cand
                if best_seller is not None:
                    seller_policies = best_seller.seller_policies
                    round_record["seller_updates"].append(
                        {"seller": seller_id, "detail": best_seller.detail, "gain": best_seller.gain, "lcb": best_seller.lcb}
                    )
            rounds.append(round_record)

        initial_test = evaluate_graphs(
            [world.test_graph], initial_buyer, initial_seller, test_sellers, self.workflow
        )
        final_test = evaluate_graphs(
            [world.test_graph], buyer_policies, seller_policies, test_sellers, self.workflow
        )
        meta_initial = evaluate_graphs(
            [world.meta_graph], initial_buyer, initial_seller, meta_sellers, self.workflow
        )
        meta_final = evaluate_graphs(
            [world.meta_graph], buyer_policies, seller_policies, meta_sellers, self.workflow
        )
        gains = [f.objective - i.objective for i, f in zip(initial_test.rows, final_test.rows)]
        gain = _mean(gains)
        gain_lcb = _lcb(gains)
        structural = sum(len(v.lineage) for v in buyer_policies.values()) + sum(
            len(v.lineage) for v in seller_policies.values()
        )
        certificate = {
            "n1_persistence": bool(structural > 0 and gain > 0),
            "n2_meta_level_generated": bool(structural >= 2),
            "n3_independent_validation": bool(gain_lcb >= 0.05),
            "n4_compounding": False,
            "n4_synergy": False,
            "n5_autonomy_over_criteria": False,
            "structural_edits": int(structural),
            "gain": round(gain, 4),
            "gain_lcb": round(gain_lcb, 4),
        }
        certificate["passed"] = bool(
            certificate["n1_persistence"]
            and certificate["n2_meta_level_generated"]
            and certificate["n3_independent_validation"]
        )
        return {
            "seed": seed,
            "mode": mode,
            "rounds": rounds,
            "initial_test": initial_test.to_dict(),
            "final_test": final_test.to_dict(),
            "meta_initial": meta_initial.to_dict(),
            "meta_final": meta_final.to_dict(),
            "gain": gain,
            "gain_lcb": gain_lcb,
            "buyer_policies": {k: v.to_dict() for k, v in buyer_policies.items()},
            "seller_policies": {k: v.to_dict() for k, v in seller_policies.items()},
            "rsi_certificate": certificate,
            "se_state": {
                "buyers": {k: v.snapshot() for k, v in self.se_buyer.items()},
                "sellers": {k: v.snapshot() for k, v in self.se_seller.items()},
            },
            "llm_calls": self.workflow.calls,
            "llm_failures": self.workflow.failures,
        }

    def run(self, modes: Sequence[str]) -> dict[str, Any]:
        results = [
            self.run_seed(int(seed), mode)
            for mode in modes
            for seed in self.settings.get("seeds", [0])
        ]
        summary: dict[str, Any] = {}
        for mode in modes:
            rows = [r for r in results if r["mode"] == mode]
            summary[mode] = {
                "n_seeds": len(rows),
                "success": _mean([r["final_test"]["success"] for r in rows]),
                "final_success": _mean([r["final_test"]["final_success"] for r in rows]),
                "pollution": _mean([r["final_test"]["pollution"] for r in rows]),
                "gain": _mean([r["gain"] for r in rows]),
                "gain_lcb": _mean([r["gain_lcb"] for r in rows]),
                "pass_rate": _mean([1.0 if r["rsi_certificate"]["passed"] else 0.0 for r in rows]),
                "llm_calls": _mean([r["llm_calls"] for r in rows]),
                "llm_failures": _mean([r["llm_failures"] for r in rows]),
            }
        return {"settings": self.settings, "results": results, "summary": summary}

    def save(self, payload: dict[str, Any], output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["mode", "seed", "success", "final_success", "pollution", "gain", "gain_lcb", "rsi_pass", "llm_calls", "llm_failures"]
            )
            for r in payload["results"]:
                writer.writerow(
                    [
                        r["mode"], r["seed"], r["final_test"]["success"],
                        r["final_test"]["final_success"], r["final_test"]["pollution"],
                        r["gain"], r["gain_lcb"], r["rsi_certificate"]["passed"],
                        r["llm_calls"], r["llm_failures"],
                    ]
                )
        lines = [
            "# Long-horizon / multi-hop NL demand-first RSI pilot",
            "",
            "| mode | step success | final success | pollution | gain | RSI pass | LLM calls |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, row in payload["summary"].items():
            lines.append(
                f"| {mode} | {row['success']:.3f} | {row['final_success']:.3f} | "
                f"{row['pollution']:.3f} | {row['gain']:.3f} | {row['pass_rate']:.2f} | "
                f"{row['llm_calls']:.1f} |"
            )
        lines += [
            "",
            "Toy six-step chain-rule diagnostic; fixed decomposition; not a public benchmark.",
            "",
        ]
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Long-horizon NL demand-first RSI diagnostic")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--modes", default="fixed,joint")
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--max-calls", type=int, default=700)
    parser.add_argument("--allocator", default="round_robin", choices=("round_robin",))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR / "long_horizon_nl_rsi"))
    args = parser.parse_args(argv)
    _load_dotenv()
    if args.offline:
        workflow = NLWorkflow(None, use_llm=False)
    else:
        client = DeepSeekClient(
            LLMConfig(
                model="deepseek-flash",
                max_tokens=2500,
                temperature=0.0,
                max_calls=args.max_calls,
            )
        )
        workflow = NLWorkflow(client, use_llm=True)
    settings = {
        "seeds": [int(s) for s in args.seeds.split(",") if s.strip()],
        "modes": [m.strip() for m in args.modes.split(",") if m.strip()],
        "delta": args.delta,
        "model": "offline-mock" if args.offline else "deepseek-flash",
        "dataset": "math_chain_rule_toy",
        "decomposition": "fixed_gold_dag",
        "allocator": args.allocator,
    }
    runner = LongHorizonRSIRunner(settings, workflow)
    payload = runner.run(settings["modes"])
    out = runner.save(payload, args.output_dir)
    print(f"Saved: {out}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
