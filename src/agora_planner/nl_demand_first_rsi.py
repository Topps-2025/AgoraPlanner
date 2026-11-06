"""Natural-language demand-first RSI pilot.

This layer connects the symbolic demand-first framework to a real LLM workflow:

    natural-language task
      -> buyer LLM writes a demand
      -> seller LLM semantically retrieves from its own inventory
      -> buyer LLM values the offers
      -> symbolic RPN factor AST combines LLM utility with provenance/quality
      -> buyer selects a bundle
      -> reader LLM answers the task
      -> demand instruction / seller retrieval instruction / RPN AST are
         improved on held-out natural-language tasks

The task/memory source is the repo's ``TextMemoryBenchmark`` (natural-language
sentences with access codes and decoys), not a facet-vector toy.  The pipeline
is still a controlled benchmark, not LongMemEval; it is the bridge needed to
test the real LLM workflow before connecting a public benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.config import DEFAULT_RESULTS_DIR, LLMConfig  # noqa: E402
from agora_planner.demand_first_rsi import (  # noqa: E402
    DemandContract,
    DemandFirstSettings,
    DemandOp,
    MemoryOffer,
    MemoryRecord,
    ProgramBundle,
    QuoteProgram,
    RPNMutator,
    RPNAST,
    SellerAgent,
    Task,
    _lcb,
    _mean,
    select_offers,
    seed_rpn_ast,
)
from agora_planner.llm import DeepSeekClient, LLMError, _load_dotenv  # noqa: E402
from agora_planner.text_benchmark import (  # noqa: E402
    TextBenchmarkConfig,
    TextMemoryBenchmark,
    extract_code,
)


# ----------------------------------------------------------------------
# Natural-language world
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class NLTask:
    task: Task
    question: str
    gold_answer: str
    buyer_id: str


@dataclass
class NLWorld:
    tasks: tuple[NLTask, ...]
    sellers: tuple[SellerAgent, ...]
    buyer_ids: tuple[str, ...]


def _to_memory(asset: Any) -> MemoryRecord:
    return MemoryRecord(
        id=asset.id,
        seller_id=asset.owner,
        content=asset.content,
        facet=asset.facet,
        source=asset.source_class,
        quality=float(asset.quality),
        freshness=0.85,
        production_cost=float(asset.production_cost),
        correlation_risk=float(asset.correlation_risk),
        is_true=bool(asset.metadata.get("is_true", False)),
        is_decoy=bool(asset.metadata.get("is_decoy", False)),
        access="team",
    )


def build_nl_world(seed: int, n_tasks: int, *, offset: int = 0) -> NLWorld:
    bench = TextMemoryBenchmark(
        TextBenchmarkConfig(
            seed=seed,
            n_agents=2,
            n_facets=8,
            facts_per_agent=3,
            n_public_facts=3,
            decoy_fraction=0.40,
            n_train_tasks=6,
            n_eval_tasks=max(18, n_tasks + offset + 6),
            max_memories_per_agent=3,
            solver="beam",
        )
    )
    memories = [_to_memory(asset) for asset in bench.memories]
    sellers = []
    for seller_id in sorted({m.seller_id for m in memories}):
        inventory = tuple(m for m in memories if m.seller_id == seller_id)
        sellers.append(SellerAgent(seller_id, inventory))
    # Prefer tasks with exactly one required memory and at least one decoy:
    # this is the cleanest natural-language test of relevance vs pollution.
    eligible = [
        t for t in bench.eval_tasks
        if len(bench.queries[t.id].required_fact_ids) == 1
        and bench.queries[t.id].decoy_fact_ids
    ]
    selected_tasks = eligible[offset : offset + n_tasks]
    if len(selected_tasks) < n_tasks:
        selected_tasks = list(bench.eval_tasks[offset : offset + n_tasks])
    tasks: list[NLTask] = []
    for task in selected_tasks:
        query = bench.queries[task.id]
        tasks.append(
            NLTask(
                task=Task(
                    id=query.id,
                    goal=query.question,
                    required_facets=(query.facet,),
                    facet_weights={query.facet: 1.0},
                    budget=6.0,
                    max_bundle=2,
                    source_limit=2,
                    required_memory_ids=frozenset(query.required_fact_ids),
                    decoy_memory_ids=frozenset(query.decoy_fact_ids),
                    success_criteria=(query.gold_answer,),
                ),
                question=query.question,
                gold_answer=query.gold_answer,
                buyer_id=query.agent_id,
            )
        )
    return NLWorld(tuple(tasks), tuple(sellers), tuple(a.id for a in bench.agents))


def build_controlled_nl_world(
    seed: int, n_tasks: int, *, offset: int = 0
) -> NLWorld:
    """A small hand-built natural-language procurement pilot.

    Each task has one true memory and one stale/decoy memory with the same
    facet.  Unlike the access-code benchmark, the task text is a realistic
    operational request, and the decoy is guaranteed to sit in a seller's
    inventory.  This is a controlled LLM-workflow pilot, not a public benchmark.
    """
    specs = [
        (
            "compliance",
            "A vendor switch needs the compliance access code for the Acme audit. What is the current code?",
            "C7Q2",
            "Acme audit record: the compliance access code is C7Q2; issuer: internal compliance.",
            "Acme audit checklist: the legacy compliance code is X9K4; this note is stale.",
        ),
        (
            "supply",
            "What is the current supply-risk code for the Helios component?",
            "R4M8",
            "Helios component record: the current supply-risk code is R4M8; source: internal supply.",
            "Helios old risk note: the supply-risk code was T1B6; superseded.",
        ),
        (
            "pricing",
            "Which access code applies to the Meridian pricing exception?",
            "P9L3",
            "Meridian pricing exception: the approved access code is P9L3; source: regulator filing.",
            "Meridian pricing draft: the access code is H2V5; not approved.",
        ),
        (
            "risk",
            "What is the regulator reporting code for the Orion incident?",
            "K6D1",
            "Orion incident record: the regulator reporting code is K6D1; source: regulator.",
            "Orion incident draft: the reporting code is Q8W2; superseded.",
        ),
        (
            "vendor",
            "Which access code is required in the Vantage vendor onboarding packet?",
            "V3N7",
            "Vantage onboarding packet: the required vendor access code is V3N7; source: internal vendor desk.",
            "Vantage onboarding draft: the vendor code is V0D2; replaced before approval.",
        ),
        (
            "audit",
            "What is the current audit-evidence code for the Drake quarterly review?",
            "A5E9",
            "Drake quarterly review: the current audit-evidence code is A5E9; source: internal audit.",
            "Drake audit working note: the evidence code is A1X8; superseded by the final review.",
        ),
        (
            "budget",
            "Which approval code applies to the Lumen budget exception?",
            "B8K2",
            "Lumen budget exception: the approved code is B8K2; source: regulator filing.",
            "Lumen budget draft: the approval code is B4Q6; not approved.",
        ),
        (
            "security",
            "What is the current security exception code for the Nimbus deployment?",
            "S2F5",
            "Nimbus deployment record: the current security exception code is S2F5; source: internal security.",
            "Nimbus security note: the exception code was S9J1; expired.",
        ),
    ]
    memories: list[MemoryRecord] = []
    tasks: list[NLTask] = []
    for i, (facet, question, gold, true_text, decoy_text) in enumerate(specs):
        true_id = f"true_{facet}_nl"
        decoy_id = f"decoy_{facet}_nl"
        seller_true = "sellerA" if i % 2 == 0 else "sellerB"
        seller_decoy = "sellerB" if seller_true == "sellerA" else "sellerA"
        memories.append(
            MemoryRecord(
                id=true_id, seller_id=seller_true, content=true_text, facet=facet,
                source="internal" if i % 2 == 0 else "regulator", quality=0.92,
                freshness=0.9, production_cost=1.0, correlation_risk=0.1,
                is_true=True, is_decoy=False,
            )
        )
        memories.append(
            MemoryRecord(
                id=decoy_id, seller_id=seller_decoy, content=decoy_text, facet=facet,
                source="vendor" if i % 2 == 0 else "market", quality=0.22,
                freshness=0.35, production_cost=0.45, correlation_risk=0.75,
                is_true=False, is_decoy=True,
            )
        )
        tasks.append(
            NLTask(
                task=Task(
                    id=f"controlled_{i}", goal=question, required_facets=(facet,),
                    facet_weights={facet: 1.0}, budget=6.0, max_bundle=2,
                    source_limit=2, required_memory_ids=frozenset({true_id}),
                    decoy_memory_ids=frozenset({decoy_id}),
                    success_criteria=(gold,), family="controlled",
                ),
                question=question, gold_answer=gold,
                buyer_id="buyer0" if i % 2 == 0 else "buyer1",
            )
        )
    sellers = []
    for sid in ("sellerA", "sellerB"):
        sellers.append(SellerAgent(sid, tuple(m for m in memories if m.seller_id == sid)))
    selected = tasks[offset : offset + n_tasks]
    return NLWorld(tuple(selected), tuple(sellers), ("buyer0", "buyer1"))


# ----------------------------------------------------------------------
# LLM workflow
# ----------------------------------------------------------------------


class NLWorkflow:
    """Thin LLM prompt layer with deterministic fallbacks."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        use_llm: bool = True,
        json_retries: int = 0,
    ):
        self.client = client
        self.use_llm = bool(use_llm)
        self.calls = 0
        self.failures = 0
        self.json_retries = max(0, int(json_retries))
        self.failure_kinds: dict[str, int] = {}

    def _record_failure(self, kind: str) -> None:
        self.failures += 1
        self.failure_kinds[kind] = self.failure_kinds.get(kind, 0) + 1

    def _json(self, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if not self.use_llm or self.client is None:
            return fallback
        for attempt in range(self.json_retries + 1):
            try:
                self.calls += 1
                return self.client.json(
                    [
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2500,
                    use_cache=(attempt == 0),
                    refresh_cache=(attempt > 0),
                )
            except Exception:
                self._record_failure("json")
                if attempt >= self.json_retries:
                    return fallback
        return fallback

    def demand(self, task: Any, instruction: str, context: str = "") -> str:
        context_block = f"\nPrior results / workspace:\n{context}\n" if context else ""
        prompt = f"""You are buyer {task.buyer_id} in a memory procurement market.

Task question: {task.question}
Demand-writing instruction: {instruction}
{context_block}
Write a concise demand for memory sellers. It should state what evidence is
needed and what answer format the buyer expects. Return JSON:
{{"demand": "one short paragraph"}}"""
        data = self._json(prompt, {"demand": task.question})
        text = str(data.get("demand", "")).strip()
        if not text:
            return task.question
        if context:
            return f"{text}\nContext: {context[:800]}"
        return text

    def retrieve(
        self,
        seller_id: str,
        demand: str,
        inventory: Sequence[MemoryRecord],
        instruction: str,
        top_k: int,
    ) -> dict[str, Any]:
        memories = "\n".join(
            f"- {m.id} | source={m.source} | quality={m.quality:.2f} | {m.content}"
            for m in inventory
        )
        prompt = f"""You are seller {seller_id}. Retrieve only from your own inventory.

Demand: {demand}
Retrieval instruction: {instruction}
Top-k: {top_k}

Inventory:
{memories}

Rank the memory ids by usefulness for the demand. Return JSON:
{{"ranked_ids": ["id1", "id2"], "relevance": {{"id1": 0.0-1.0}}}}"""
        fallback_ids = [m.id for m in inventory[:top_k]]
        data = self._json(
            prompt,
            {"ranked_ids": fallback_ids, "relevance": {}},
        )
        ranked = [str(x) for x in data.get("ranked_ids", []) if str(x)]
        if not ranked:
            ranked = fallback_ids
        return {
            "ranked_ids": ranked[: max(1, top_k)],
            "relevance": data.get("relevance", {}),
        }

    def value(self, demand: str, offers: Sequence[MemoryOffer]) -> dict[str, Any]:
        items = "\n".join(
            f"- {o.record.id} | source={o.record.source} | quality={o.record.quality:.2f} "
            f"| ask={o.ask:.2f} | {o.record.content}"
            for o in offers
        )
        prompt = f"""You are the buyer. Demand: {demand}

Offers:
{items}

Score each offer for usefulness to the demand, redundancy, and trustworthiness.
Return JSON:
{{"scores": {{"memory_id": {{"utility": 0.0-1.0, "redundancy": 0.0-1.0}}}}}}"""
        fallback = {
            "scores": {
                o.record.id: {
                    "utility": float(o.record.quality),
                    "redundancy": float(o.record.correlation_risk),
                }
                for o in offers
            }
        }
        data = self._json(prompt, fallback)
        scores = data.get("scores", {})
        return scores if isinstance(scores, dict) else fallback["scores"]

    def answer(self, question: str, selected: Sequence[MemoryOffer]) -> str:
        context = "\n".join(f"- {o.record.content}" for o in selected) or "(no memory selected)"
        if not self.use_llm or self.client is None:
            # Deterministic fallback: return the first access code in context.
            code = extract_code(context)
            return code or ""
        prompt = f"""Answer the question using only the selected memory passages.

Question: {question}
Passages:
{context}

Return JSON: {{"answer": "the exact 4-character access code"}}."""
        for attempt in range(self.json_retries + 1):
            try:
                self.calls += 1
                data = self.client.json(
                    [
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                    use_cache=(attempt == 0),
                    refresh_cache=(attempt > 0),
                )
                answer = str(data.get("answer", "")).strip()
                return answer or (extract_code(context) or "")
            except Exception:
                self._record_failure("answer")
                if attempt >= self.json_retries:
                    return extract_code(context) or ""
        return extract_code(context) or ""

    def answer_general(
        self,
        question: str,
        selected: Sequence[MemoryOffer],
        context: str = "",
        answer_format: str = "a mathematical expression or a number",
    ) -> str:
        """Answer a long-horizon subgoal from selected memory plus prior results."""
        passages = "\n".join(f"- {o.record.content}" for o in selected) or "(no memory selected)"
        prior = f"\nPrior results / workspace:\n{context}\n" if context else ""
        if not self.use_llm or self.client is None:
            # Offline fallback: prefer an explicitly labelled prior result,
            # otherwise return the first selected passage.  It is intentionally
            # weak; the live workflow uses the LLM.
            if context:
                lines = [ln for ln in context.splitlines() if "result" in ln.lower()]
                if lines:
                    return lines[-1].split(":", 1)[-1].strip()
            return selected[0].record.content if selected else ""
        prompt = f"""Answer the subgoal using only the selected memory passages and prior results.

Question: {question}
Passages:
{passages}
{prior}
Return the answer as {answer_format}.
Use plain text, not LaTeX. For math, write sin(x), cos(x), and use * for multiplication.
Return JSON: {{"answer": "the exact answer"}}."""
        for attempt in range(self.json_retries + 1):
            try:
                self.calls += 1
                data = self.client.json(
                    [
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                    use_cache=(attempt == 0),
                    refresh_cache=(attempt > 0),
                )
                answer = str(data.get("answer", "")).strip()
                return answer or (selected[0].record.content if selected else "")
            except Exception:
                self._record_failure("answer_general")
                if attempt >= self.json_retries:
                    return selected[0].record.content if selected else ""
        return selected[0].record.content if selected else ""


# ----------------------------------------------------------------------
# Policies and agents
# ----------------------------------------------------------------------


@dataclass
class NLBuyerPolicy:
    demand_instruction: str
    valuation: RPNAST
    quote: QuoteProgram
    demand_ops: tuple[DemandOp, ...] = ()
    version: int = 0
    parent_version: int | None = None
    lineage: tuple[str, ...] = ()

    def copy_with(
        self,
        *,
        demand_instruction: str | None = None,
        valuation: RPNAST | None = None,
        quote: QuoteProgram | None = None,
        demand_ops: Sequence[DemandOp] | None = None,
        lineage: Sequence[str] = (),
    ) -> "NLBuyerPolicy":
        return NLBuyerPolicy(
            demand_instruction=self.demand_instruction if demand_instruction is None else demand_instruction,
            valuation=valuation or self.valuation,
            quote=quote or self.quote,
            demand_ops=self.demand_ops if demand_ops is None else tuple(demand_ops),
            version=self.version + 1,
            parent_version=self.version,
            lineage=tuple(self.lineage) + tuple(lineage),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "demand_instruction": self.demand_instruction,
            "demand_ops": [
                {"kind": op.kind, "facet": op.facet, "value": op.value, "note": op.note}
                for op in self.demand_ops
            ],
            "valuation": self.valuation.to_dict(),
            "quote": self.quote.to_dict(),
            "lineage": list(self.lineage),
        }


SELLER_RPN_FIELDS = (
    "quality",
    "freshness",
    "production_cost",
    "scarcity",
    "access_risk",
    "opportunity_cost",
    "correlation",
    "demand_match",
)
SELLER_RPN_BINARY = ("ADD", "SUB", "MUL", "DIV", "MAX", "MIN", "GATE")
SELLER_RPN_UNARY = ("ABS", "CLIP01", "SQUARE")


@dataclass(frozen=True)
class SellerEvalContext:
    record: MemoryRecord
    demand_match: float = 0.5
    scarcity: float = 0.5


def _seller_factor(name: str, ctx: SellerEvalContext) -> float:
    record = ctx.record
    if name == "quality":
        return float(record.quality)
    if name == "freshness":
        return float(record.freshness)
    if name == "production_cost":
        return max(0.0, min(1.0, float(record.production_cost)))
    if name == "scarcity":
        return max(0.0, min(1.0, float(ctx.scarcity)))
    if name == "access_risk":
        return max(0.0, min(1.0, float(record.access_risk)))
    if name == "opportunity_cost":
        return max(0.0, min(1.0, float(record.opportunity_cost)))
    if name == "correlation":
        return max(0.0, min(1.0, float(record.correlation_risk)))
    if name == "demand_match":
        return max(0.0, min(1.0, float(ctx.demand_match)))
    return 0.0


def _seller_rpn_eval(tokens: Sequence[Any], ctx: SellerEvalContext) -> float:
    stack: list[float] = []
    for token in tokens:
        if isinstance(token, (int, float)) and not isinstance(token, bool):
            stack.append(float(token))
            continue
        name = str(token)
        if name in SELLER_RPN_FIELDS:
            stack.append(_seller_factor(name, ctx))
            continue
        if name in SELLER_RPN_BINARY:
            b = stack.pop()
            a = stack.pop()
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
                stack.append(a if b >= 0.5 else 0.0)
            continue
        if name in SELLER_RPN_UNARY:
            a = stack.pop()
            if name == "ABS":
                stack.append(abs(a))
            elif name == "CLIP01":
                stack.append(max(0.0, min(1.0, a)))
            elif name == "SQUARE":
                stack.append(a * a)
            continue
        raise ValueError(f"unknown seller RPN token: {name}")
    if len(stack) != 1:
        raise ValueError(f"seller RPN leaves {len(stack)} values")
    return stack[0]


@dataclass
class SellerRPNAST:
    tokens: tuple[Any, ...]
    version: int = 0
    parent_version: int | None = None
    name: str = "seed_seller_quality"
    lineage: tuple[str, ...] = ()

    def evaluate(
        self,
        record: MemoryRecord,
        *,
        demand_match: float = 0.5,
        scarcity: float = 0.5,
    ) -> float:
        value = _seller_rpn_eval(
            self.tokens,
            SellerEvalContext(record, demand_match=demand_match, scarcity=scarcity),
        )
        return max(0.0, min(1.0, value))

    def copy_with(
        self,
        tokens: Sequence[Any],
        *,
        name: str | None = None,
        lineage: Sequence[str] = (),
    ) -> "SellerRPNAST":
        return SellerRPNAST(
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


def seed_seller_valuation() -> SellerRPNAST:
    # The seed reproduces the old ask heuristic: value = quality.
    return SellerRPNAST(("quality",), version=0, name="seed_seller_quality")


class SellerRPNMutator:
    """Structural edits over seller valuation RPN.

    Seller fields/operators are fixed.  RSI searches the expression structure
    (quality, freshness, demand match, scarcity, access/opportunity costs,
    correlation) and the resulting seller reservation value feeds ask.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    @staticmethod
    def _valid(tokens: Sequence[Any]) -> bool:
        try:
            rec = MemoryRecord(
                "x", "s", "", "f", "internal", 0.5, 0.5, 0.1, 0.1,
                True, False,
            )
            _seller_rpn_eval(tokens, SellerEvalContext(rec))
            return True
        except Exception:
            return False

    def propose_from(
        self, current: SellerRPNAST, *, limit: int = 8
    ) -> list[tuple[SellerRPNAST, str]]:
        base = list(current.tokens)
        variants: list[tuple[list[Any], str]] = [
            (base + ["quality", 0.5, "MUL", "ADD"], "seller: append quality * 0.5"),
            (base + ["freshness", 0.5, "MUL", "ADD"], "seller: append freshness * 0.5"),
            (base + ["demand_match", 0.8, "MUL", "ADD"], "seller: append demand_match * 0.8"),
            (base + ["scarcity", 0.4, "MUL", "ADD"], "seller: append scarcity * 0.4"),
            (base + ["access_risk", 1.0, "MUL", "SUB"], "seller: subtract access_risk * 1.0"),
            (base + ["opportunity_cost", 1.0, "MUL", "SUB"], "seller: subtract opportunity_cost * 1.0"),
            (base + ["correlation", 0.8, "MUL", "SUB"], "seller: subtract correlation * 0.8"),
            (
                base + ["quality", 0.6, "MUL", "ADD", "demand_match", 0.6, "MUL", "ADD", "access_risk", 0.8, "MUL", "SUB"],
                "seller: compose quality + demand_match - access_risk",
            ),
            (base + ["freshness", "GATE"], "seller: gate on freshness"),
            (base + [0.5, "MUL"], "seller: scale 0.5"),
        ]
        out: list[tuple[SellerRPNAST, str]] = []
        seen: set[tuple[Any, ...]] = set()
        for tokens, detail in variants:
            key = tuple(tokens)
            if key in seen or not self._valid(tokens):
                continue
            seen.add(key)
            out.append(
                (
                    current.copy_with(tokens, name="seller_candidate", lineage=[detail]),
                    detail,
                )
            )
            if len(out) >= limit:
                break
        return out


@dataclass
class NLSellerPolicy:
    retrieval_instruction: str
    top_k: int = 3
    min_quality: float = 0.0
    valuation: SellerRPNAST = field(default_factory=seed_seller_valuation)
    version: int = 0
    parent_version: int | None = None
    lineage: tuple[str, ...] = ()

    def copy_with(
        self,
        *,
        retrieval_instruction: str | None = None,
        top_k: int | None = None,
        min_quality: float | None = None,
        valuation: SellerRPNAST | None = None,
        lineage: Sequence[str] = (),
    ) -> "NLSellerPolicy":
        return NLSellerPolicy(
            retrieval_instruction=self.retrieval_instruction if retrieval_instruction is None else retrieval_instruction,
            top_k=self.top_k if top_k is None else top_k,
            min_quality=self.min_quality if min_quality is None else min_quality,
            valuation=valuation or self.valuation,
            version=self.version + 1,
            parent_version=self.version,
            lineage=tuple(self.lineage) + tuple(lineage),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "retrieval_instruction": self.retrieval_instruction,
            "top_k": self.top_k,
            "min_quality": round(self.min_quality, 4),
            "valuation": self.valuation.to_dict(),
            "lineage": list(self.lineage),
        }


DEFAULT_DEMAND_INSTRUCTION = "Ask for memory that is relevant to the question."
DEFAULT_RETRIEVAL_INSTRUCTION = "Rank memories by keyword overlap with the demand."

DEMAND_VARIANTS = (
    "State the exact evidence needed, the answer format, and ask for source and quality.",
    "Ask for the exact access code and require internal/regulator provenance; reject stale or duplicate notes.",
    "Write a concise procurement demand: question, required facet, answer format, and a provenance constraint.",
)
RETRIEVAL_VARIANTS = (
    "Rank by semantic relevance to the full demand; prefer exact topic match and high-quality provenance; penalize stale duplicates.",
    "Prefer memories that contain the requested answer format and whose source is internal or regulator; ignore unrelated notes.",
    "Retrieve the smallest sufficient evidence set: exact facet, high freshness, low source redundancy.",
)


# ----------------------------------------------------------------------
# NL market and evaluation
# ----------------------------------------------------------------------


@dataclass
class NLTaskResult:
    task_id: str
    buyer_id: str
    demand: str
    offers: tuple[MemoryOffer, ...]
    selected: tuple[MemoryOffer, ...]
    answer: str
    success: float
    coverage: float
    pollution: float
    spend: float

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
            "task_id": self.task_id,
            "buyer_id": self.buyer_id,
            "demand": self.demand,
            "n_offers": len(self.offers),
            "selected": [o.record.id for o in self.selected],
            "answer": self.answer,
            "success": self.success,
            "coverage": round(self.coverage, 4),
            "pollution": round(self.pollution, 4),
            "spend": round(self.spend, 4),
        }


def _seller_offers(
    seller: SellerAgent,
    policy: NLSellerPolicy,
    demand: str,
    contract: DemandContract,
    quote: QuoteProgram,
    workflow: NLWorkflow,
) -> list[MemoryOffer]:
    data = workflow.retrieve(
        seller.id, demand, seller.inventory, policy.retrieval_instruction, policy.top_k
    )
    by_id = {m.id: m for m in seller.inventory}
    relevance = data.get("relevance", {})
    ranked = list(data.get("ranked_ids", []))
    # Tiny seller inventories: return all candidate offers so the buyer sees
    # both the true memory and the stale decoy.  This is a controlled pilot
    # choice, not a general retrieval shortcut.
    if len(seller.inventory) <= 3:
        for m in seller.inventory:
            if m.id not in ranked:
                ranked.append(m.id)
    facet_counts: dict[str, int] = {}
    for memory in seller.inventory:
        facet_counts[memory.facet] = facet_counts.get(memory.facet, 0) + 1
    offers: list[MemoryOffer] = []
    for mid in ranked:
        record = by_id.get(mid)
        if record is None:
            continue
        if record.quality < policy.min_quality:
            continue
        if record.quality < contract.min_quality:
            continue
        if contract.require_provenance and record.source not in ("internal", "regulator"):
            continue
        demand_match = max(0.0, min(1.0, float(relevance.get(mid, 0.5))))
        count = facet_counts.get(record.facet, 1)
        scarcity = 1.0 - (count - 1) / max(1, len(seller.inventory) - 1)
        if record.access_risk > 0.0:
            access_risk = record.access_risk
        else:
            access_risk = 0.1 if record.source in ("internal", "regulator") else 0.7
        seller_value = policy.valuation.evaluate(
            record,
            demand_match=demand_match,
            scarcity=max(0.0, min(1.0, scarcity)),
        )
        # The ask is a fixed mapping from the seller's reservation value; the
        # quote rule itself is not an RSI object in this round.
        ask = quote.seller_ask(record, value_score=seller_value)
        offers.append(
            MemoryOffer(
                record=record,
                seller_id=seller.id,
                ask=ask,
                relevance=demand_match,
                preview=record.content[:120],
            )
        )
    return offers


def _buyer_value(
    offers: Sequence[MemoryOffer],
    demand: str,
    workflow: NLWorkflow,
) -> list[MemoryOffer]:
    scores = workflow.value(demand, offers)
    out: list[MemoryOffer] = []
    for offer in offers:
        entry = scores.get(offer.record.id, {})
        utility = float(entry.get("utility", offer.record.quality))
        out.append(replace(offer, llm_utility=max(0.0, min(1.0, utility))))
    return out


def apply_demand_ops(
    contract: DemandContract,
    ops: Sequence[DemandOp],
) -> DemandContract:
    """Apply task-relative demand operators to a concrete contract.

    The contract is the part of demand that changes retrieval/selection:
    provenance requirement, quality floor, bundle/source limits.  Instruction
    text alone is not a demand policy.
    """
    for op in ops:
        if op.kind == "require_provenance":
            contract.require_provenance = bool(op.value) if op.value is not None else True
        elif op.kind == "set_min_quality" and op.value is not None:
            contract.min_quality = max(0.0, min(1.0, float(op.value)))
        elif op.kind == "set_max_bundle" and op.value is not None:
            contract.max_bundle = max(1, int(op.value))
        elif op.kind == "set_max_bundle_required":
            contract.max_bundle = max(1, len(contract.facets))
        elif op.kind == "set_source_limit" and op.value is not None:
            contract.source_limit = max(1, int(op.value))
        elif op.kind == "set_source_limit_required":
            contract.source_limit = max(1, min(2, len(contract.facets)))
        elif op.kind == "set_budget" and op.value is not None:
            contract.budget = max(0.1, contract.budget * float(op.value))
        elif op.kind == "add_clarification" and op.note:
            contract.clarifications = tuple(contract.clarifications) + (op.note,)
        # include_all_required / include_missing_required are already
        # represented by contract.facets in the NL task builder.
    return contract


def run_nl_task(
    task: NLTask,
    buyer_policy: NLBuyerPolicy,
    seller_policies: dict[str, NLSellerPolicy],
    sellers: dict[str, SellerAgent],
    workflow: NLWorkflow,
) -> NLTaskResult:
    demand_text = workflow.demand(task, buyer_policy.demand_instruction)
    contract = DemandContract(
        task_id=task.task.id,
        goal=task.question,
        facets=task.task.required_facets,
        facet_weights=dict(task.task.facet_weights),
        budget=task.task.budget,
        max_bundle=task.task.max_bundle,
        source_limit=task.task.source_limit,
        acceptance=task.task.success_criteria,
    )
    contract = apply_demand_ops(contract, buyer_policy.demand_ops)
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
    offers = _buyer_value(offers, demand_text, workflow)
    selected = select_offers(
        contract,
        offers,
        buyer_policy.valuation,
        buyer_policy.quote,
    )
    chosen = tuple(p.offer for p in selected)
    answer = workflow.answer(task.question, chosen)
    selected_ids = {o.record.id for o in chosen}
    required = set(task.task.required_memory_ids)
    covered = required & selected_ids
    decoys = sum(1 for o in chosen if o.record.is_decoy)
    coverage = len(covered) / max(1, len(required))
    pollution = decoys / max(1, len(chosen))
    parsed = extract_code(answer)
    success = 1.0 if parsed == task.gold_answer and coverage >= 1.0 else 0.0
    return NLTaskResult(
        task_id=task.task.id,
        buyer_id=task.buyer_id,
        demand=demand_text,
        offers=tuple(offers),
        selected=chosen,
        answer=answer,
        success=success,
        coverage=coverage,
        pollution=pollution,
        spend=float(sum(p.price for p in selected)),
    )


@dataclass
class NLEval:
    rows: list[NLTaskResult]
    success: float
    coverage: float
    pollution: float
    spend: float
    objective: float = 0.0
    offer_precision: float = 0.0
    retrieval_objective: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": round(self.objective, 4),
            "offer_precision": round(self.offer_precision, 4),
            "retrieval_objective": round(self.retrieval_objective, 4),
            "success": round(self.success, 4),
            "coverage": round(self.coverage, 4),
            "pollution": round(self.pollution, 4),
            "spend": round(self.spend, 4),
            "n_tasks": len(self.rows),
            "rows": [r.to_dict() for r in self.rows],
        }


def evaluate_nl(
    tasks: Sequence[NLTask],
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    sellers: dict[str, SellerAgent],
    workflow: NLWorkflow,
) -> NLEval:
    rows = [
        run_nl_task(
            task,
            buyer_policies[task.buyer_id],
            seller_policies,
            sellers,
            workflow,
        )
        for task in tasks
    ]
    success = _mean([r.success for r in rows])
    pollution = _mean([r.pollution for r in rows])
    offer_precision = _mean([
        sum(1.0 for o in r.offers if not o.record.is_decoy) / max(1, len(r.offers))
        for r in rows
    ])
    objective = success - 0.5 * pollution
    return NLEval(
        rows=rows,
        success=success,
        coverage=_mean([r.coverage for r in rows]),
        pollution=pollution,
        spend=_mean([r.spend for r in rows]),
        objective=objective,
        offer_precision=offer_precision,
        # Seller retrieval is scored on the buyer-facing offer set as well as
        # the downstream task outcome: a seller that exposes fewer stale offers
        # is better even when the buyer's current RPN already ignores them.
        retrieval_objective=objective + 0.25 * offer_precision,
    )


# ----------------------------------------------------------------------
# RSI over demand, seller retrieval, and RPN valuation
# ----------------------------------------------------------------------


@dataclass
class NLCandidate:
    scope: str
    detail: str
    buyer_policies: dict[str, NLBuyerPolicy]
    seller_policies: dict[str, NLSellerPolicy]
    evaluation: NLEval
    gain: float
    lcb: float
    accepted: bool


@dataclass
class SEMutationStat:
    """Per-agent exploration statistics for one mutation family.

    SE does not edit the RPN expression or the retrieval policy schema.  It
    only uses downstream gains to decide which fixed mutation family to try
    next.  Structural acceptance is still the RSI controller's job.
    """

    trials: int = 0
    reward_sum: float = 0.0
    accepted: int = 0
    last_gain: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / max(1, self.trials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trials": self.trials,
            "mean_gain": round(self.mean_reward, 4),
            "accepted": self.accepted,
            "last_gain": round(self.last_gain, 4),
        }


class SEExplorer:
    """Downstream-signal-driven exploration scheduler (inner SE loop).

    The candidate generators are fixed.  This class reorders the *exploration
    direction* using an optimistic bandit score over mutation families.  It is
    deliberately per-agent: two buyers/sellers can explore different families
    even when they see the same task family.
    """

    def __init__(self, *, exploration: float = 0.5):
        self.exploration = float(exploration)
        self.stats: dict[str, SEMutationStat] = {}

    def _stat(self, kind: str) -> SEMutationStat:
        return self.stats.setdefault(kind, SEMutationStat())

    def order(self, items: Sequence[tuple[Any, str]]) -> list[tuple[Any, str]]:
        total = sum(st.trials for st in self.stats.values())
        untried = [(item, kind) for item, kind in items if self._stat(kind).trials == 0]
        tried = [(item, kind) for item, kind in items if self._stat(kind).trials > 0]

        def score(pair: tuple[Any, str]) -> float:
            stat = self._stat(pair[1])
            bonus = self.exploration * math.sqrt(math.log(total + 2.0) / stat.trials)
            return stat.mean_reward + bonus

        tried.sort(key=score, reverse=True)
        return untried + tried

    def update(self, kind: str, gain: float, accepted: bool) -> None:
        stat = self._stat(kind)
        stat.trials += 1
        stat.reward_sum += float(gain)
        stat.last_gain = float(gain)
        if accepted:
            stat.accepted += 1

    def snapshot(self) -> dict[str, Any]:
        return {kind: stat.to_dict() for kind, stat in sorted(self.stats.items())}


def _candidate_kind(scope: str, detail: str) -> str:
    if scope == "valuation_rpn":
        if "compose" in detail:
            return "rpn:compose"
        for name in (
            "quality", "provenance", "redundancy", "source_corr",
            "complementarity", "coverage", "gate", "llm_utility", "scale", "shift",
        ):
            if name in detail:
                return f"rpn:{name}"
        return "rpn:other"
    if scope == "demand_instruction":
        return "demand:instruction"
    if scope == "demand_ops":
        return "demand:ops"
    if scope == "demand_rpn":
        return "demand:joint_rpn"
    if scope == "min_quality":
        return "seller:min_quality"
    if scope == "top_k":
        return "seller:top_k"
    if scope == "retrieval_instruction":
        return "seller:retrieval_instruction"
    if scope == "seller_valuation_rpn":
        return "seller:valuation_rpn"
    return scope


class NLRSIRunner:
    def __init__(self, settings: dict[str, Any], workflow: NLWorkflow):
        self.settings = settings
        self.workflow = workflow
        self.mutator = RPNMutator(int(settings.get("seed", 0)) + 1)
        self.seller_mutator = SellerRPNMutator(int(settings.get("seed", 0)) + 2)
        self.se_buyer: dict[str, SEExplorer] = {}
        self.se_seller: dict[str, SEExplorer] = {}

    @staticmethod
    def _gains(candidate: NLEval, base: NLEval) -> tuple[float, float]:
        gains = [c.objective - b.objective for c, b in zip(candidate.rows, base.rows)]
        return _mean(gains), _lcb(gains)

    @staticmethod
    def _seller_gains(candidate: NLEval, base: NLEval) -> tuple[float, float]:
        gains = [
            c.retrieval_objective - b.retrieval_objective
            for c, b in zip(candidate.rows, base.rows)
        ]
        return _mean(gains), _lcb(gains)

    def _buyer_candidates(
        self,
        buyer_id: str,
        policy: NLBuyerPolicy,
        limit: int = 6,
    ) -> list[tuple[NLBuyerPolicy, str, str]]:
        out: list[tuple[NLBuyerPolicy, str, str]] = []
        # Reserve slots for demand edits.  If RPN candidates consume the whole
        # budget, demand RSI is never actually evaluated (a bug in the earlier
        # pilot): the lossy demand compression must itself be editable.
        rpn_limit = max(1, limit - 3)
        for ast, mutation in self.mutator.propose_from(policy.valuation, limit=rpn_limit):
            out.append(
                (
                    policy.copy_with(
                        valuation=ast,
                        lineage=[mutation.detail],
                    ),
                    "valuation_rpn",
                    mutation.detail,
                )
            )
        demand_ops = tuple(policy.demand_ops) + (
            DemandOp("require_provenance", value=True),
            DemandOp("set_min_quality", value=0.5),
        )
        has_provenance = any(
            op.kind == "require_provenance" and op.value is True
            for op in policy.demand_ops
        )
        if not has_provenance:
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
            # Joint demand+RPN edit: demand is lossy, so updating only one of
            # the two layers is not enough when both contract and valuation
            # are wrong.
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
        for instr in DEMAND_VARIANTS[:2]:
            if instr == policy.demand_instruction:
                continue
            out.append(
                (
                    policy.copy_with(
                        demand_instruction=instr,
                        lineage=[f"demand: {instr[:80]}"],
                    ),
                    "demand_instruction",
                    instr[:80],
                )
            )
        return out[:limit]

    def _seller_candidates(
        self,
        seller_id: str,
        policy: NLSellerPolicy,
        limit: int = 4,
    ) -> list[tuple[NLSellerPolicy, str, str]]:
        out: list[tuple[NLSellerPolicy, str, str]] = []
        # Seller-side valuation mirrors buyer-side valuation: a RPN over
        # fixed seller fields (quality/freshness/scarcity/access risk/
        # opportunity cost/correlation/demand match).  The resulting value
        # feeds the seller's ask; quote rule stays fixed.
        for ast, detail in self.seller_mutator.propose_from(policy.valuation, limit=2):
            out.append(
                (
                    policy.copy_with(valuation=ast, lineage=[detail]),
                    "seller_valuation_rpn",
                    detail,
                )
            )
        # Retrieval policy has three parts: semantic reranking instruction,
        # how many candidates to expose, and the seller's own quality floor.
        # The quality floor is learned from buyer feedback, not hard-coded.
        for q in (0.5, 0.7):
            if q > policy.min_quality:
                out.append(
                    (
                        policy.copy_with(
                            min_quality=q,
                            lineage=[f"min_quality -> {q:.2f}"],
                        ),
                        "min_quality",
                        f"min_quality -> {q:.2f}",
                    )
                )
        for tk in (max(1, policy.top_k - 1), policy.top_k + 1):
            if tk == policy.top_k:
                continue
            out.append(
                (
                    policy.copy_with(
                        top_k=tk,
                        lineage=[f"top_k -> {tk}"],
                    ),
                    "top_k",
                    f"top_k -> {tk}",
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
        return out[:limit]

    def run_seed(self, seed: int, mode: str) -> dict[str, Any]:
        test_offset = int(self.settings.get("test_offset", self.settings.get("meta_tasks", 2)))
        if self.settings.get("dataset", "controlled") == "controlled":
            meta_world = build_controlled_nl_world(
                seed, int(self.settings.get("meta_tasks", 2))
            )
            test_world = build_controlled_nl_world(
                seed,
                int(self.settings.get("test_tasks", 2)),
                offset=test_offset,
            )
        else:
            meta_world = build_nl_world(seed, int(self.settings.get("meta_tasks", 2)))
            test_world = build_nl_world(
                seed,
                int(self.settings.get("test_tasks", 2)),
                offset=int(self.settings.get("meta_tasks", 2)),
            )
        assign_meta_to = self.settings.get("assign_meta_to")
        if assign_meta_to:
            # Task assignment is an explicit, exogenous layer in this pilot.
            # The allocator can hand a subset of tasks to one buyer; the RSI
            # object remains that buyer's policy, not a shared system policy.
            meta_world = NLWorld(
                tuple(replace(t, buyer_id=str(assign_meta_to)) for t in meta_world.tasks),
                meta_world.sellers,
                meta_world.buyer_ids,
            )
        buyer_policies = {
            bid: NLBuyerPolicy(
                DEFAULT_DEMAND_INSTRUCTION,
                RPNAST(("llm_utility", "facet_match", "MUL"), version=0, name="seed_llm_facet"),
                QuoteProgram(price_rule="ask"),
            )
            for bid in meta_world.buyer_ids
        }
        seller_policies = {
            s.id: NLSellerPolicy(DEFAULT_RETRIEVAL_INSTRUCTION, top_k=5)
            for s in meta_world.sellers
        }
        sellers = {s.id: s for s in meta_world.sellers}
        self.se_buyer = {bid: SEExplorer() for bid in buyer_policies}
        self.se_seller = {sid: SEExplorer() for sid in seller_policies}
        initial_buyer = {k: v.copy_with() for k, v in buyer_policies.items()}
        initial_seller = {k: v.copy_with() for k, v in seller_policies.items()}
        rounds: list[dict[str, Any]] = []
        for round_idx in range(int(self.settings.get("rounds", 1))):
            if mode == "fixed":
                rounds.append({"round": round_idx, "decision": "fixed"})
                continue
            round_record: dict[str, Any] = {"round": round_idx, "buyer_updates": [], "seller_updates": []}
            # Buyer-local RSI: each buyer has its own tasks and policy.
            for buyer_id, policy in list(buyer_policies.items()):
                tasks = [t for t in meta_world.tasks if t.buyer_id == buyer_id]
                if not tasks:
                    continue
                base = evaluate_nl(tasks, buyer_policies, seller_policies, sellers, self.workflow)
                candidates = self._buyer_candidates(buyer_id, policy)
                ordered = self.se_buyer[buyer_id].order([
                    (c, _candidate_kind(c[1], c[2])) for c in candidates
                ])
                evaluated: list[NLCandidate] = []
                for (candidate, scope, detail), kind in ordered:
                    trial = dict(buyer_policies)
                    trial[buyer_id] = candidate
                    evaluation = evaluate_nl(tasks, trial, seller_policies, sellers, self.workflow)
                    gain, lcb = self._gains(evaluation, base)
                    accepted = lcb > float(self.settings.get("delta", 0.0))
                    self.se_buyer[buyer_id].update(kind, gain, accepted)
                    evaluated.append(
                        NLCandidate(scope, detail, trial, dict(seller_policies), evaluation, gain, lcb, accepted)
                    )
                accepted = [c for c in evaluated if c.accepted]
                if accepted:
                    # On equal held-out evidence, prefer the joint demand+RPN
                    # edit so the lossy demand compression is actually updated.
                    def _priority(scope: str) -> int:
                        if scope == "demand_rpn":
                            return 2
                        if scope == "demand_ops":
                            return 1
                        return 0

                    accepted.sort(
                        key=lambda c: (c.lcb, c.gain, _priority(c.scope)),
                        reverse=True,
                    )
                    best = accepted[0]
                    buyer_policies = best.buyer_policies
                    round_record["buyer_updates"].append(
                        {"buyer": buyer_id, "detail": best.detail, "gain": best.gain, "lcb": best.lcb}
                    )
            # Seller-local RSI: retrieval policy is per seller.  Validate the
            # seller edit against the *initial* buyer policy, so the seller
            # cannot free-ride on the buyer's just-accepted RPN edit.  The
            # final test then composes both independently validated edits.
            for seller_id, policy in list(seller_policies.items()):
                base = evaluate_nl(meta_world.tasks, initial_buyer, seller_policies, sellers, self.workflow)
                candidates = self._seller_candidates(seller_id, policy)
                ordered = self.se_seller[seller_id].order([
                    (c, _candidate_kind(c[1], c[2])) for c in candidates
                ])
                best = None
                for (candidate, scope, detail), kind in ordered:
                    trial = dict(seller_policies)
                    trial[seller_id] = candidate
                    evaluation = evaluate_nl(meta_world.tasks, initial_buyer, trial, sellers, self.workflow)
                    gain, lcb = self._seller_gains(evaluation, base)
                    accepted = lcb > float(self.settings.get("delta", 0.0))
                    self.se_seller[seller_id].update(kind, gain, accepted)
                    cand = NLCandidate(scope, detail, dict(buyer_policies), trial, evaluation, gain, lcb, accepted)
                    def _seller_priority(scope: str) -> int:
                        if scope == "seller_valuation_rpn":
                            return 3
                        if scope == "retrieval_instruction":
                            return 2
                        if scope == "top_k":
                            return 1
                        return 0

                    if cand.accepted and (
                        best is None
                        or (cand.lcb, cand.gain, _seller_priority(scope))
                        > (best.lcb, best.gain, _seller_priority(best.scope))
                    ):
                        best = cand
                if best is not None:
                    seller_policies = best.seller_policies
                    round_record["seller_updates"].append(
                        {"seller": seller_id, "detail": best.detail, "gain": best.gain, "lcb": best.lcb}
                    )
            rounds.append(round_record)

        test_sellers = {s.id: s for s in test_world.sellers}
        initial_test = evaluate_nl(test_world.tasks, initial_buyer, initial_seller, test_sellers, self.workflow)
        final_test = evaluate_nl(test_world.tasks, buyer_policies, seller_policies, test_sellers, self.workflow)
        gains = [f.objective - i.objective for i, f in zip(initial_test.rows, final_test.rows)]
        gain = _mean(gains)
        gain_lcb = _lcb(gains)
        structural = sum(
            len(v.lineage)
            for v in buyer_policies.values()
        ) + sum(len(v.lineage) for v in seller_policies.values())
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
        results = [self.run_seed(int(s), mode) for mode in modes for s in self.settings.get("seeds", [0])]
        summary: dict[str, Any] = {}
        for mode in modes:
            rows = [r for r in results if r["mode"] == mode]
            summary[mode] = {
                "n_seeds": len(rows),
                "success": _mean([r["final_test"]["success"] for r in rows]),
                "coverage": _mean([r["final_test"]["coverage"] for r in rows]),
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
            writer.writerow(["mode", "seed", "success", "coverage", "pollution", "gain", "gain_lcb", "rsi_pass", "llm_calls", "llm_failures"])
            for r in payload["results"]:
                writer.writerow([
                    r["mode"], r["seed"], r["final_test"]["success"], r["final_test"]["coverage"],
                    r["final_test"]["pollution"], r["gain"], r["gain_lcb"],
                    r["rsi_certificate"]["passed"], r["llm_calls"], r["llm_failures"],
                ])
        lines = [
            "# Natural-language demand-first RSI pilot",
            "",
            "| mode | task success | coverage | pollution | gain | RSI pass | LLM calls |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, row in payload["summary"].items():
            lines.append(
                f"| {mode} | {row['success']:.3f} | {row['coverage']:.3f} | "
                f"{row['pollution']:.3f} | {row['gain']:.3f} | {row['pass_rate']:.2f} | "
                f"{row['llm_calls']:.1f} |"
            )
        lines += ["", "This is a natural-language LLM workflow on the repo's text-memory benchmark, not a public benchmark SOTA claim.", ""]
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Natural-language demand-first RSI pilot")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--meta-tasks", type=int, default=2)
    parser.add_argument("--test-tasks", type=int, default=2)
    parser.add_argument("--modes", default="fixed,joint")
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dataset", default="controlled", choices=("controlled", "text"))
    parser.add_argument("--test-offset", type=int, default=None)
    parser.add_argument("--assign-meta-to", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR / "nl_demand_first_rsi_pilot"))
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
        "rounds": args.rounds,
        "meta_tasks": args.meta_tasks,
        "test_tasks": args.test_tasks,
        "delta": args.delta,
        "model": "offline-mock" if args.offline else "deepseek-flash",
        "dataset": args.dataset,
        "test_offset": args.test_offset if args.test_offset is not None else args.meta_tasks,
        "assign_meta_to": args.assign_meta_to,
    }
    runner = NLRSIRunner(settings, workflow)
    payload = runner.run([m.strip() for m in args.modes.split(",") if m.strip()])
    out = runner.save(payload, args.output_dir)
    print(f"Saved: {out}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
