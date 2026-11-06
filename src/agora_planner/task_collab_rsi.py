"""Task-collaboration-grounded recursive self-improvement (N4 pilot).

Why this module exists
----------------------
The earlier long-horizon pilot (``long_horizon_nl_rsi.py``) edits a fixed
candidate list with a fixed acceptance rule.  Its update procedure ``U`` is
hand-written: ``_buyer_candidates``, ``_seller_candidates`` and the LCB gate do
not change.  That is bounded policy-structure search (L1), not RSI.

This module makes the *update procedure itself* the RSI object:

    h_{t+1} = U_t(h_t, D_t)                       # object layer
    U_{t+1} = Edit(U_t, held-out collaboration)   # meta layer (this module)

and grounds the entire signal in **task collaboration**, not transaction
outcomes:

    gain(o) = task_success(with seller offer memory o) - task_success(without)

The transaction (bid/ask/price) is only the bridge that decides whether an
offer can be acquired.  It is never the acceptance objective.  A memory is
valuable because it changes verified task completion.

The counterfactual probe dataset
--------------------------------
For each meta/validation step we run the real LLM reader once per seller offer
under a *gold prior context*.  This isolates the marginal task-completion gain
of each memory.  The probe is a real LLM call plus the deterministic verifier;
the replay layer only re-uses those measured outcomes to evaluate candidate
policies without further API calls.

The final test is a live long-horizon rollout with the policy's own context
(no oracle), so cross-step error propagation is measured honestly.

What is editable (U)
--------------------
An ``UpdateProgram`` is a typed sequence of stages.  The meta-mutator inserts,
deletes, replaces, or swaps stages, and the meta-controller accepts a candidate
U only if the policies it produces achieve higher verified task completion on
held-out episodes/families.  Adopted U versions persist and are reused.

Stages:
    diagnose             measure coverage / pollution / affordability / Q error
    calibrate_q          fit Q(o)=w.phi(o) from per-memory task gains; emit RPN
    repair_demand        choose demand-contract filters by replay task success
    repair_quote         choose the transaction bridge policy by task success
    repair_seller        choose seller floor / seller valuation by task success
    joint_demand_q       atomic demand + Q update (coordination)
    joint_seller_quote   atomic seller-floor + quote-bridge update (coordination)

Boundaries (honest):
- bounded RSI: the meta-mutation language and the verifier are fixed (N5 not
  claimed); U is edited inside a typed stage library.
- demand *instruction text* and seller *retrieval instruction* are fixed in the
  main run because their effect is not replayable without new LLM traces.
- the counterfactual replay is anchored to the gold-context probe dataset;
  the final live test is the end-to-end check.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import statistics
import sys
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.config import DEFAULT_RESULTS_DIR, LLMConfig  # noqa: E402
from agora_planner.demand_first_rsi import (  # noqa: E402
    DemandContract,
    DemandOp,
    EvalContext,
    MemoryOffer,
    MemoryRecord,
    QuoteProgram,
    RPNAST,
    SellerAgent,
    _factor_value,
    _lcb,
    _mean,
    _std,
    select_offers,
)
from agora_planner.llm import DeepSeekClient, _load_dotenv  # noqa: E402
from agora_planner.nl_demand_first_rsi import (  # noqa: E402
    DEFAULT_DEMAND_INSTRUCTION,
    DEFAULT_RETRIEVAL_INSTRUCTION,
    NLBuyerPolicy,
    NLSellerPolicy,
    NLWorkflow,
    SellerRPNMutator,
    _buyer_value,
    _seller_offers,
    apply_demand_ops,
    seed_seller_valuation,
)


# ----------------------------------------------------------------------
# Long-horizon collaboration world
# ----------------------------------------------------------------------

FAMILY_ORDER = (
    "vendor_pricing",
    "incident_scoring",
    "budget_approval",
    "compliance_penalty",
)
META_FAMILIES = FAMILY_ORDER[:2]
VALIDATION_FAMILY = FAMILY_ORDER[2]
UNSEEN_FAMILY = FAMILY_ORDER[3]

FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "vendor_pricing": {
        "case": "the Helios vendor renewal",
        "unit": "rate",
        "labels": [
            "base rate",
            "compliance loading",
            "volume adjustment",
            "regional surcharge",
            "risk premium",
            "brokerage fee",
            "loyalty discount",
            "audit add-on",
            "currency adjustment",
            "final settlement rate",
        ],
    },
    "incident_scoring": {
        "case": "the Orion incident review",
        "unit": "severity index",
        "labels": [
            "baseline severity",
            "containment weight",
            "blast-radius adjustment",
            "detection penalty",
            "recovery credit",
            "escalation loading",
            "communication credit",
            "regulatory surcharge",
            "vendor dependency factor",
            "final severity index",
        ],
    },
    "budget_approval": {
        "case": "the Lumen budget exception",
        "unit": "approved amount",
        "labels": [
            "requested amount",
            "department deduction",
            "contingency addition",
            "compliance reserve",
            "executive credit",
            "vendor concentration loading",
            "multi-year discount",
            "audit withholding",
            "currency buffer",
            "final approved amount",
        ],
    },
    "compliance_penalty": {
        "case": "the Drake compliance finding",
        "unit": "penalty amount",
        "labels": [
            "base penalty",
            "repeat-offense loading",
            "mitigation credit",
            "reporting delay surcharge",
            "remediation discount",
            "scope expansion loading",
            "cooperation credit",
            "cross-border surcharge",
            "recidivism factor",
            "final penalty amount",
        ],
    },
}

# The operation schedule is identical across families so the held-out family
# tests transfer of the *update procedure*, not a new arithmetic skill.
OP_SCHEDULE = (
    "set",
    "mul_pct",
    "add_pct",
    "sub_pct",
    "mul_pct",
    "add_abs",
    "mul_pct",
    "sub_abs",
    "mul_pct",
    "sub_pct",
)

STEP_TEMPLATE_SET = (
    "[{step_id}] Retrieve the {label} for {case}. "
    "What is the starting {unit}? Give a single number."
)
STEP_TEMPLATE_APPLY = (
    "[{step_id}] The current {unit} is {prev:.2f}. "
    "Retrieve the {label} for {case} and apply it exactly as the memory states. "
    "Give the new {unit} rounded to two decimals as a single number."
)

ANSWER_FORMAT = "a number rounded to two decimals"


@dataclass(frozen=True)
class CollabStep:
    id: str
    episode_id: str
    index: int
    question: str
    gold_answer: float
    buyer_id: str
    facet: str
    required_memory_ids: frozenset[str]
    decoy_memory_ids: frozenset[str]
    depends_on: tuple[str, ...]
    family: str
    label: str
    unit: str
    op: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "index": self.index,
            "question": self.question,
            "gold_answer": round(self.gold_answer, 4),
            "buyer_id": self.buyer_id,
            "facet": self.facet,
            "required_memory_ids": sorted(self.required_memory_ids),
            "decoy_memory_ids": sorted(self.decoy_memory_ids),
            "depends_on": list(self.depends_on),
            "family": self.family,
            "label": self.label,
            "op": self.op,
        }


@dataclass
class CollabEpisode:
    id: str
    family: str
    variant: int
    steps: tuple[CollabStep, ...]
    final_step_id: str
    sellers: tuple[SellerAgent, ...]
    buyer_ids: tuple[str, ...]

    def ordered(self) -> list[CollabStep]:
        return list(self.steps)


@dataclass
class WorldSplit:
    fit: tuple[CollabEpisode, ...]
    gate: tuple[CollabEpisode, ...]
    validation: tuple[CollabEpisode, ...]
    test_seen: tuple[CollabEpisode, ...]
    test_unseen: tuple[CollabEpisode, ...]

    def all_trace_episodes(self) -> tuple[CollabEpisode, ...]:
        return self.fit + self.gate + self.validation

    def test_episodes(self) -> tuple[CollabEpisode, ...]:
        return self.test_seen + self.test_unseen


def _rng(family: str, variant: int) -> random.Random:
    salt = zlib.crc32(f"{family}:{variant}".encode("utf-8"))
    return random.Random(100_003 + salt)


def _op_text(op: str, value: float, unit: str) -> str:
    if op == "set":
        return f"the starting {unit} is {value:.2f}"
    if op == "mul_pct":
        return (
            f"multiply the current {unit} by {1.0 + value / 100.0:.4f} "
            f"({value:.1f}% increase)"
        )
    if op == "add_pct":
        return (
            f"multiply the current {unit} by {1.0 + value / 100.0:.4f} "
            f"({value:.1f}% addition)"
        )
    if op == "sub_pct":
        return (
            f"multiply the current {unit} by {1.0 - value / 100.0:.4f} "
            f"({value:.1f}% reduction)"
        )
    if op == "add_abs":
        return f"add {value:.2f} to the current {unit}"
    if op == "sub_abs":
        return f"subtract {value:.2f} from the current {unit}"
    raise ValueError(f"unknown op {op}")


def _apply_op(op: str, prev: float, value: float) -> float:
    if op == "set":
        return round(value, 2)
    if op == "mul_pct":
        return round(prev * (1.0 + value / 100.0), 2)
    if op == "add_pct":
        return round(prev * (1.0 + value / 100.0), 2)
    if op == "sub_pct":
        return round(prev * (1.0 - value / 100.0), 2)
    if op == "add_abs":
        return round(prev + value, 2)
    if op == "sub_abs":
        return round(prev - value, 2)
    raise ValueError(f"unknown op {op}")


def _draw_value(rng: random.Random, op: str) -> float:
    if op == "set":
        return round(rng.uniform(2.0, 8.0), 1)
    if op in ("mul_pct", "add_pct", "sub_pct"):
        return round(rng.uniform(2.0, 12.0), 1)
    return round(rng.uniform(0.5, 3.0), 1)


def _perturb(rng: random.Random, op: str, value: float) -> float:
    if op == "set":
        delta = rng.choice((-2.0, -1.5, 1.5, 2.0))
        return max(0.5, round(value + delta, 1))
    if op in ("mul_pct", "add_pct", "sub_pct"):
        delta = rng.choice((-4.0, -3.0, 3.0, 4.0))
        return max(1.0, round(value + delta, 1))
    delta = rng.choice((-1.2, -0.8, 0.8, 1.2))
    return max(0.1, round(value + delta, 1))


def build_episode(family: str, variant: int, *, n_steps: int = 10) -> CollabEpisode:
    spec = FAMILY_SPECS[family]
    rng = _rng(family, variant)
    episode_id = f"{family}_v{variant}"
    steps: list[CollabStep] = []
    memories: list[MemoryRecord] = []
    prev_true = 0.0
    prev_decoy = 0.0
    for i in range(n_steps):
        op = OP_SCHEDULE[i % len(OP_SCHEDULE)]
        label = spec["labels"][i % len(spec["labels"])]
        unit = spec["unit"]
        true_value = _draw_value(rng, op)
        decoy_value = _perturb(rng, op, true_value)
        if op == "set":
            true_step = _apply_op(op, 0.0, true_value)
            decoy_step = _apply_op(op, 0.0, decoy_value)
            question = STEP_TEMPLATE_SET.format(
                step_id=f"{episode_id}_s{i:02d}",
                label=label,
                case=spec["case"],
                unit=unit,
            )
        else:
            true_step = _apply_op(op, prev_true, true_value)
            decoy_step = _apply_op(op, prev_decoy, decoy_value)
            question = STEP_TEMPLATE_APPLY.format(
                step_id=f"{episode_id}_s{i:02d}",
                unit=unit,
                prev=prev_true,
                label=label,
                case=spec["case"],
            )
        # Keep the counterfactual branch numerically distinct; otherwise the
        # decoy cannot teach the policy anything.
        guard = 0
        while abs(true_step - decoy_step) < 0.2 and guard < 8:
            decoy_value = _perturb(rng, op, decoy_value)
            decoy_step = _apply_op(op, prev_decoy, decoy_value)
            guard += 1

        step_id = f"{episode_id}_s{i:02d}"
        facet = f"{family}_step_{i:02d}"
        true_id = f"{step_id}_true"
        decoy_id = f"{step_id}_decoy"
        seller_true = "sellerA" if i % 2 == 0 else "sellerB"
        seller_decoy = "sellerB" if seller_true == "sellerA" else "sellerA"
        low_quality_true = i % 3 == 1
        true_quality = 0.45 if low_quality_true else 0.92
        decoy_quality = 0.55 if low_quality_true else 0.22
        true_source = "internal" if i % 2 == 0 else "regulator"
        decoy_source = "vendor" if i % 2 == 0 else "market"

        memories.append(
            MemoryRecord(
                id=true_id,
                seller_id=seller_true,
                content=(
                    f"Current policy for {spec['case']}: {label} — "
                    f"{_op_text(op, true_value, unit)}. Round to two decimals."
                ),
                facet=facet,
                source=true_source,
                quality=true_quality,
                freshness=0.90,
                production_cost=0.80 if i % 4 == 2 else 0.35,
                correlation_risk=0.10,
                is_true=True,
                is_decoy=False,
                kind="policy",
                valid_scope="current",
                opportunity_cost=0.40,
                access_risk=0.05,
            )
        )
        memories.append(
            MemoryRecord(
                id=decoy_id,
                seller_id=seller_decoy,
                content=(
                    f"Legacy policy for {spec['case']}: {label} — "
                    f"{_op_text(op, decoy_value, unit)}. Superseded; keep for history only."
                ),
                facet=facet,
                source=decoy_source,
                quality=decoy_quality,
                freshness=0.30,
                production_cost=0.08,
                correlation_risk=0.80,
                is_true=False,
                is_decoy=True,
                kind="policy",
                valid_scope="superseded",
                opportunity_cost=0.0,
                access_risk=0.70,
            )
        )
        steps.append(
            CollabStep(
                id=step_id,
                episode_id=episode_id,
                index=i,
                question=question,
                gold_answer=true_step,
                buyer_id="buyer0" if i % 2 == 0 else "buyer1",
                facet=facet,
                required_memory_ids=frozenset({true_id}),
                decoy_memory_ids=frozenset({decoy_id}),
                depends_on=(f"{episode_id}_s{i-1:02d}",) if i > 0 else (),
                family=family,
                label=label,
                unit=unit,
                op=op,
            )
        )
        prev_true = true_step
        prev_decoy = decoy_step

    sellers = []
    for seller_id in ("sellerA", "sellerB"):
        inventory = tuple(m for m in memories if m.seller_id == seller_id)
        sellers.append(SellerAgent(seller_id, inventory))
    return CollabEpisode(
        id=episode_id,
        family=family,
        variant=variant,
        steps=tuple(steps),
        final_step_id=steps[-1].id,
        sellers=tuple(sellers),
        buyer_ids=("buyer0", "buyer1"),
    )


def build_world_split(seed: int = 0, *, n_steps: int = 10) -> WorldSplit:
    """Build a cross-family split.

    fit:        vendor_pricing / incident_scoring  (variant 0)
    gate:       vendor_pricing                    (variant 1, held-out episode)
    validation: budget_approval                   (variant 0, held-out family)
    test_seen:  budget_approval                   (variant 1, held-out episode)
    test_unseen: compliance_penalty               (variant 0, unseen family)

    ``seed`` shifts the constant bank without changing the task structure.
    """
    offset = int(seed) * 17
    fit = (
        build_episode(META_FAMILIES[0], offset + 0, n_steps=n_steps),
        build_episode(META_FAMILIES[1], offset + 1, n_steps=n_steps),
    )
    gate = (build_episode(META_FAMILIES[0], offset + 2, n_steps=n_steps),)
    validation = (build_episode(VALIDATION_FAMILY, offset + 3, n_steps=n_steps),)
    test_seen = (build_episode(VALIDATION_FAMILY, offset + 4, n_steps=n_steps),)
    test_unseen = (build_episode(UNSEEN_FAMILY, offset + 5, n_steps=n_steps),)
    return WorldSplit(
        fit=fit,
        gate=gate,
        validation=validation,
        test_seen=test_seen,
        test_unseen=test_unseen,
    )


# ----------------------------------------------------------------------
# Verification and state
# ----------------------------------------------------------------------


def verify_number(answer: str, gold: float, *, tolerance: float = 0.03) -> bool:
    text = str(answer).replace(",", "").replace("−", "-")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return False
    try:
        return abs(float(numbers[-1]) - float(gold)) <= tolerance
    except ValueError:
        return False


@dataclass
class CollabState:
    results: dict[str, tuple[str, float]] = field(default_factory=dict)

    def context(self, limit: int = 14) -> str:
        lines = [f"{sid} result: {answer}" for sid, (answer, _) in self.results.items()]
        return "\n".join(lines[-limit:])


def gold_context(episode: CollabEpisode, upto: int) -> str:
    """Oracle prior context used for counterfactual memory probes.

    This is the credit-assignment device: it isolates the marginal effect of a
    seller offer memory while holding the rest of the long-horizon task
    correct.  The final test does *not* use this context.
    """
    lines = [
        f"{step.id} result: {step.gold_answer:.2f}" for step in episode.steps[:upto]
    ]
    return "\n".join(lines)


def make_contract(step: CollabStep, ops: Sequence[DemandOp] = ()) -> DemandContract:
    contract = DemandContract(
        task_id=step.id,
        goal=step.question,
        facets=(step.facet,),
        facet_weights={step.facet: 1.0},
        budget=1.0,
        max_bundle=1,
        source_limit=2,
        acceptance=(f"{step.gold_answer:.2f}",),
    )
    return apply_demand_ops(contract, ops)


# ----------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------


def seed_buyer_policies(
    episode: CollabEpisode, *, seed_q: str = "llm_utility"
) -> dict[str, NLBuyerPolicy]:
    if seed_q == "constant":
        # Deliberately uninformative seed Q: every offer has value 1.0, so the
        # buyer selects the cheapest affordable offer (the stale decoy).  This
        # forces the task-collaboration signal to update Q, not just filters.
        tokens: tuple[Any, ...] = ("facet_match",)
        name = "seed_constant_facet"
    else:
        tokens = ("llm_utility", "facet_match", "MUL")
        name = "seed_llm_facet"
    return {
        bid: NLBuyerPolicy(
            DEFAULT_DEMAND_INSTRUCTION,
            RPNAST(tokens, version=0, name=name),
            QuoteProgram(price_rule="ask"),
        )
        for bid in episode.buyer_ids
    }


def seed_seller_policies(episode: CollabEpisode) -> dict[str, NLSellerPolicy]:
    return {
        seller.id: NLSellerPolicy(
            DEFAULT_RETRIEVAL_INSTRUCTION,
            top_k=max(1, len(seller.inventory)),
            min_quality=0.5,
        )
        for seller in episode.sellers
    }


# ----------------------------------------------------------------------
# Counterfactual collaboration traces
# ----------------------------------------------------------------------


@dataclass
class TraceStep:
    step: CollabStep
    demand: str
    offers: tuple[MemoryOffer, ...]
    single_success: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step.id,
            "buyer_id": self.step.buyer_id,
            "demand": self.demand,
            "offers": [offer.to_dict() for offer in self.offers],
            "single_success": {
                key: round(value, 4) for key, value in self.single_success.items()
            },
        }


@dataclass
class EpisodeTrace:
    episode_id: str
    family: str
    steps: tuple[TraceStep, ...]
    sellers: tuple[SellerAgent, ...] = ()

    def sellers_map(self) -> dict[str, SellerAgent]:
        return {seller.id: seller for seller in self.sellers}

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "family": self.family,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class TraceBundle:
    fit: tuple[EpisodeTrace, ...]
    gate: tuple[EpisodeTrace, ...]
    validation: tuple[EpisodeTrace, ...]

    def all(self) -> tuple[EpisodeTrace, ...]:
        return self.fit + self.gate + self.validation


def _permissive_seller_policy(seller: SellerAgent) -> NLSellerPolicy:
    return NLSellerPolicy(
        DEFAULT_RETRIEVAL_INSTRUCTION,
        top_k=max(1, len(seller.inventory)),
        min_quality=0.0,
    )


def _collect_offer_pool(
    step: CollabStep,
    demand: str,
    sellers: Sequence[SellerAgent],
    workflow: Any,
) -> list[MemoryOffer]:
    contract = make_contract(step, ())
    quote = QuoteProgram(price_rule="ask")
    pool: list[MemoryOffer] = []
    for seller in sellers:
        pool.extend(
            _seller_offers(
                seller,
                _permissive_seller_policy(seller),
                demand,
                contract,
                quote,
                workflow,
            )
        )
    # Scoped pool: only this step's facet.  Both the true and the stale decoy
    # are retained so the counterfactual labels can separate them.
    pool = [offer for offer in pool if offer.record.facet == step.facet]
    by_id: dict[str, MemoryOffer] = {}
    for offer in pool:
        by_id.setdefault(offer.record.id, offer)
    pool = list(by_id.values())
    return _buyer_value(pool, demand, workflow)


def collect_episode_trace(episode: CollabEpisode, workflow: Any) -> EpisodeTrace:
    rows: list[TraceStep] = []
    for index, step in enumerate(episode.ordered()):
        context = gold_context(episode, index)
        demand = workflow.demand(step, DEFAULT_DEMAND_INSTRUCTION, context=context)
        pool = _collect_offer_pool(step, demand, episode.sellers, workflow)
        single: dict[str, float] = {}
        for offer in pool:
            answer = workflow.answer_general(
                step.question,
                [offer],
                context=context,
                answer_format=ANSWER_FORMAT,
            )
            covered = offer.record.id in step.required_memory_ids
            single[offer.record.id] = (
                1.0 if covered and verify_number(answer, step.gold_answer) else 0.0
            )
        rows.append(TraceStep(step, demand, tuple(pool), single))
    return EpisodeTrace(episode.id, episode.family, tuple(rows), episode.sellers)


def collect_trace_bundle(
    split: WorldSplit,
    workflow: Any,
    *,
    limit: int | None = None,
    progress: bool = False,
) -> TraceBundle:
    def _collect(
        episodes: Sequence[CollabEpisode], name: str
    ) -> tuple[EpisodeTrace, ...]:
        selected = list(episodes)[:limit] if limit else list(episodes)
        traces: list[EpisodeTrace] = []
        for episode in selected:
            trace = collect_episode_trace(episode, workflow)
            traces.append(trace)
            if progress:
                true_ok = 0
                true_n = 0
                decoy_ok = 0
                decoy_n = 0
                for step in trace.steps:
                    for offer in step.offers:
                        value = step.single_success.get(offer.record.id, 0.0)
                        if offer.record.is_decoy:
                            decoy_n += 1
                            decoy_ok += int(value >= 1.0)
                        else:
                            true_n += 1
                            true_ok += int(value >= 1.0)
                print(
                    f"[trace:{name}] {episode.id} steps={len(trace.steps)} "
                    f"true_probe={true_ok}/{true_n} decoy_probe={decoy_ok}/{decoy_n}",
                    flush=True,
                )
        return tuple(traces)

    return TraceBundle(
        fit=_collect(split.fit, "fit"),
        gate=_collect(split.gate, "gate"),
        validation=_collect(split.validation, "validation"),
    )


# ----------------------------------------------------------------------
# Replay evaluation (no LLM calls)
# ----------------------------------------------------------------------


@dataclass
class ReplayStep:
    episode_id: str
    step_id: str
    buyer_id: str
    family: str
    selected: tuple[str, ...]
    success: float
    pollution: float
    spend: float
    surplus: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "buyer_id": self.buyer_id,
            "family": self.family,
            "selected": list(self.selected),
            "success": self.success,
            "pollution": round(self.pollution, 4),
            "spend": round(self.spend, 4),
            "surplus": round(self.surplus, 4),
        }


@dataclass
class ReplayEval:
    rows: tuple[ReplayStep, ...]
    success: float
    final_success: float
    pollution: float
    spend: float
    surplus: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": round(self.success, 4),
            "final_success": round(self.final_success, 4),
            "pollution": round(self.pollution, 4),
            "spend": round(self.spend, 4),
            "surplus": round(self.surplus, 4),
            "n_steps": len(self.rows),
            "rows": [row.to_dict() for row in self.rows],
        }


def _scarcity(record: MemoryRecord, seller: SellerAgent) -> float:
    count = sum(1 for memory in seller.inventory if memory.facet == record.facet)
    if len(seller.inventory) <= 1:
        return 1.0
    return 1.0 - (count - 1) / max(1, len(seller.inventory) - 1)


def prepare_offers(
    trace_step: TraceStep,
    buyer_policy: NLBuyerPolicy,
    seller_policies: dict[str, NLSellerPolicy],
    sellers: dict[str, SellerAgent],
) -> tuple[DemandContract, list[MemoryOffer]]:
    """Apply the current demand contract and seller policy to the fixed pool.

    Ask prices are recomputed from the seller valuation and the buyer quote so
    seller-valuation and quote-stage edits are replayable without new LLM calls.
    """
    step = trace_step.step
    contract = make_contract(step, buyer_policy.demand_ops)
    offers: list[MemoryOffer] = []
    for offer in trace_step.offers:
        seller_policy = seller_policies[offer.seller_id]
        seller = sellers[offer.seller_id]
        record = offer.record
        if record.facet != step.facet:
            continue
        if record.quality < seller_policy.min_quality:
            continue
        if record.quality < contract.min_quality:
            continue
        if contract.require_provenance and record.source not in ("internal", "regulator"):
            continue
        value_score = seller_policy.valuation.evaluate(
            record,
            demand_match=float(offer.relevance),
            scarcity=_scarcity(record, seller),
        )
        ask = buyer_policy.quote.seller_ask(record, value_score=value_score)
        offers.append(
            MemoryOffer(
                record=record,
                seller_id=offer.seller_id,
                ask=float(ask),
                relevance=float(offer.relevance),
                preview=record.content[:120],
                llm_utility=float(offer.llm_utility),
            )
        )
    return contract, offers


def replay_step(
    trace_step: TraceStep,
    buyer_policy: NLBuyerPolicy,
    seller_policies: dict[str, NLSellerPolicy],
    sellers: dict[str, SellerAgent],
) -> ReplayStep:
    contract, offers = prepare_offers(trace_step, buyer_policy, seller_policies, sellers)
    selected = select_offers(contract, offers, buyer_policy.valuation, buyer_policy.quote)
    ids = tuple(purchase.offer.record.id for purchase in selected)
    if not ids:
        success = 0.0
    elif len(ids) == 1:
        success = float(trace_step.single_success.get(ids[0], 0.0))
    else:
        # max_bundle is one in this world; keep a conservative fallback.
        success = float(max(trace_step.single_success.get(mid, 0.0) for mid in ids))
    decoys = sum(
        1 for purchase in selected if purchase.offer.record.is_decoy
    ) / max(1, len(selected))
    spend = float(sum(purchase.price for purchase in selected))
    surplus = float(
        sum(purchase.value_score - purchase.price for purchase in selected)
    )
    return ReplayStep(
        episode_id=trace_step.step.episode_id,
        step_id=trace_step.step.id,
        buyer_id=trace_step.step.buyer_id,
        family=trace_step.step.family,
        selected=ids,
        success=success,
        pollution=decoys,
        spend=spend,
        surplus=surplus,
    )


def replay_eval(
    traces: Sequence[EpisodeTrace],
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
) -> ReplayEval:
    rows: list[ReplayStep] = []
    finals: list[float] = []
    for trace in traces:
        sellers = trace.sellers_map()
        episode_rows = [
            replay_step(
                trace_step,
                buyer_policies[trace_step.step.buyer_id],
                seller_policies,
                sellers,
            )
            for trace_step in trace.steps
        ]
        rows.extend(episode_rows)
        finals.append(episode_rows[-1].success if episode_rows else 0.0)
    return ReplayEval(
        rows=tuple(rows),
        success=_mean([row.success for row in rows]),
        final_success=_mean(finals),
        pollution=_mean([row.pollution for row in rows]),
        spend=_mean([row.spend for row in rows]),
        surplus=_mean([row.surplus for row in rows]),
    )


def _paired_values(
    candidate: ReplayEval,
    base: ReplayEval,
    *,
    key: str = "success",
    buyer_id: str | None = None,
) -> list[float]:
    base_by = {(row.episode_id, row.step_id): row for row in base.rows}
    gains: list[float] = []
    for row in candidate.rows:
        if buyer_id is not None and row.buyer_id != buyer_id:
            continue
        reference = base_by.get((row.episode_id, row.step_id))
        if reference is None:
            continue
        gains.append(float(getattr(row, key) - getattr(reference, key)))
    return gains


def _gate_value(
    gains: Sequence[float], mode: str, delta: float
) -> tuple[bool, float, float]:
    if not gains:
        return False, 0.0, 0.0
    mean = _mean(gains)
    lcb = _lcb(gains)
    if mode == "lcb":
        return lcb > delta, lcb, mean
    if mode == "mean":
        return mean > delta, lcb, mean
    if mode == "strict":
        return (mean > delta and min(gains) >= 0.0), lcb, mean
    if mode == "none":
        return True, lcb, mean
    return mean > delta, lcb, mean


# ----------------------------------------------------------------------
# Q calibration: demand valuation learned from task-completion gains
# ----------------------------------------------------------------------

Q_FEATURES = (
    "llm_utility",
    "quality",
    "provenance",
    "freshness",
    "source_corr",
    "redundancy",
    "complementarity",
    "bundle_coverage",
)


def q_features(offer: MemoryOffer, contract: DemandContract) -> np.ndarray:
    context = EvalContext(contract, offer, ())
    return np.asarray(
        [_factor_value(name, context) for name in Q_FEATURES], dtype=float
    )


@dataclass
class QFit:
    weights: dict[str, float]
    bias: float
    n_rows: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {key: round(value, 6) for key, value in self.weights.items()},
            "bias": round(self.bias, 6),
            "n_rows": self.n_rows,
            "label": self.label,
        }


@dataclass
class _QRow:
    x: np.ndarray
    y: float


def _label_for_offer(
    label: str,
    offer: MemoryOffer,
    trace_step: TraceStep,
    selected_id: str | None,
) -> float:
    if label == "single_gain":
        return float(trace_step.single_success.get(offer.record.id, 0.0))
    if label in ("reference_gain", "selected_gain"):
        if selected_id is not None and offer.record.id == selected_id:
            return float(trace_step.single_success.get(offer.record.id, 0.0))
        return 0.0
    raise ValueError(f"unknown Q label: {label}")


def fit_q(
    traces: Sequence[EpisodeTrace],
    buyer_id: str,
    buyer_policy: NLBuyerPolicy,
    seller_policies: dict[str, NLSellerPolicy],
    *,
    label: str = "single_gain",
    ridge_lambda: float = 1.0,
    max_terms: int = 6,
    min_abs_weight: float = 1e-4,
) -> QFit:
    rows: list[_QRow] = []
    for trace in traces:
        sellers = trace.sellers_map()
        for trace_step in trace.steps:
            if trace_step.step.buyer_id != buyer_id:
                continue
            contract = make_contract(trace_step.step, ())
            selected_id: str | None = None
            if label in ("reference_gain", "selected_gain"):
                contract_sel, offers = prepare_offers(
                    trace_step, buyer_policy, seller_policies, sellers
                )
                purchases = select_offers(
                    contract_sel, offers, buyer_policy.valuation, buyer_policy.quote
                )
                if purchases:
                    selected_id = purchases[0].offer.record.id
            for offer in trace_step.offers:
                rows.append(
                    _QRow(
                        q_features(offer, contract),
                        _label_for_offer(label, offer, trace_step, selected_id),
                    )
                )
    if not rows:
        return QFit({}, 0.0, 0, label)
    x = np.stack([row.x for row in rows])
    y = np.asarray([row.y for row in rows], dtype=float)
    # Drop constant columns: they carry no ranking information and make the
    # design matrix singular.
    keep = x.std(axis=0) > 1e-9
    names = [name for name, use in zip(Q_FEATURES, keep) if use]
    if not names:
        return QFit({}, 0.0, len(rows), label)
    z = x[:, keep]
    mean = z.mean(axis=0)
    scale = z.std(axis=0) + 1e-9
    z = (z - mean) / scale
    design = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    reg = ridge_lambda * np.eye(design.shape[1])
    reg[0, 0] = 0.0  # do not regularize the intercept
    try:
        solution = np.linalg.solve(design.T @ design + reg, design.T @ y)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(design, y, rcond=None)[0]
    bias = float(solution[0] - np.sum(solution[1:] * mean / scale))
    raw = {
        name: float(weight / scale[idx])
        for idx, (name, weight) in enumerate(zip(names, solution[1:]))
    }
    ranked = sorted(raw.items(), key=lambda pair: abs(pair[1]), reverse=True)
    terms = {name: weight for name, weight in ranked[:max_terms] if abs(weight) >= min_abs_weight}
    return QFit(terms, bias, len(rows), label)


def ast_from_qfit(fit: QFit, *, lineage: Sequence[str] = ()) -> RPNAST:
    tokens: list[Any] = []
    if abs(fit.bias) > 1e-6:
        tokens.append(round(fit.bias, 6))
    for name, weight in sorted(fit.weights.items(), key=lambda pair: abs(pair[1]), reverse=True):
        if abs(weight) < 1e-6:
            continue
        tokens.extend([name, round(abs(weight), 6), "MUL"])
        tokens.append("ADD" if weight > 0 else "SUB")
    if not tokens:
        tokens = ["llm_utility", 1.0, "MUL"]
    return RPNAST(
        tuple(tokens),
        version=0,
        name=f"q_ridge_{fit.label}",
        lineage=tuple(lineage),
    )


def qfit_detail(fit: QFit) -> str:
    pieces = [f"{name}={weight:+.3f}" for name, weight in fit.weights.items()]
    return f"Q ridge {fit.label}: bias={fit.bias:+.3f}, " + ", ".join(pieces)


# ----------------------------------------------------------------------
# Update programs: the object of RSI
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateStage:
    kind: str
    params: tuple[tuple[str, Any], ...] = ()

    def param(self, name: str, default: Any = None) -> Any:
        for key, value in self.params:
            if key == name:
                return value
        return default

    def with_params(self, **changes: Any) -> "UpdateStage":
        data = dict(self.params)
        data.update(changes)
        return UpdateStage(self.kind, tuple(sorted(data.items())))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}


def stage(kind: str, **params: Any) -> UpdateStage:
    return UpdateStage(kind, tuple(sorted(params.items())))


@dataclass
class UpdateProgram:
    version: int
    stages: tuple[UpdateStage, ...]
    parent_version: int | None = None
    name: str = "U0_task_collab"
    lineage: tuple[str, ...] = ()

    def copy_with(
        self,
        stages: Sequence[UpdateStage],
        *,
        name: str | None = None,
        lineage: Sequence[str] = (),
    ) -> "UpdateProgram":
        return UpdateProgram(
            version=self.version + 1,
            stages=tuple(stages),
            parent_version=self.version,
            name=name or f"{self.name}_v{self.version + 1}",
            lineage=tuple(self.lineage) + tuple(lineage),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "parent_version": self.parent_version,
            "stages": [stage.to_dict() for stage in self.stages],
            "lineage": list(self.lineage),
        }


def seed_update_program() -> UpdateProgram:
    """Deliberately weak seed U.

    It labels Q from the *selected* offer only (self-confirming, sparse) and
    tunes the quote bridge.  It has no demand-contract repair and no seller
    repair, so the meta-RSI has real structural work to do.
    """
    return UpdateProgram(
        version=0,
        name="U0_task_collab",
        stages=(
            stage(
                "calibrate_q",
                label="reference_gain",
                ridge_lambda=2.0,
                max_terms=4,
                gate="mean",
            ),
            stage("repair_quote", shadings=(0.0, 0.1, 0.2), gate="mean"),
        ),
        lineage=("seed: selected-only Q + quote bridge",),
    )


@dataclass
class StageRecord:
    kind: str
    params: dict[str, Any]
    edits: list[dict[str, Any]]
    accepted: bool
    gain: float
    lcb: float
    n_candidates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "params": self.params,
            "edits": self.edits,
            "accepted": self.accepted,
            "gain": round(self.gain, 4),
            "lcb": round(self.lcb, 4),
            "n_candidates": self.n_candidates,
        }


@dataclass
class UpdateRun:
    buyer_policies: dict[str, NLBuyerPolicy]
    seller_policies: dict[str, NLSellerPolicy]
    records: list[StageRecord]
    n_candidate_evaluations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "n_candidate_evaluations": self.n_candidate_evaluations,
        }


DEMAND_OP_LIBRARY: dict[str, tuple[DemandOp, ...]] = {
    "none": (),
    "prov": (DemandOp("require_provenance", value=True),),
    "q05": (DemandOp("set_min_quality", value=0.5),),
    "prov_q05": (
        DemandOp("require_provenance", value=True),
        DemandOp("set_min_quality", value=0.5),
    ),
    "q07": (DemandOp("set_min_quality", value=0.7),),
    "prov_q07": (
        DemandOp("require_provenance", value=True),
        DemandOp("set_min_quality", value=0.7),
    ),
    "budget15": (DemandOp("set_budget", value=1.5),),
    "prov_budget15": (
        DemandOp("require_provenance", value=True),
        DemandOp("set_budget", value=1.5),
    ),
    "budget20": (DemandOp("set_budget", value=2.0),),
}


@dataclass
class UpdateContext:
    fit_traces: tuple[EpisodeTrace, ...]
    gate_traces: tuple[EpisodeTrace, ...]
    settings: dict[str, Any]
    diagnosis: dict[str, float] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return str(self.settings.get("mode", "meta_rsi"))

    @property
    def delta(self) -> float:
        return float(self.settings.get("delta", 0.0))

    @property
    def gate_signal(self) -> str:
        return str(self.settings.get("gate_signal", "success"))


def _gate_key(ctx: UpdateContext) -> str:
    return "surplus" if ctx.gate_signal == "surplus" else "success"


def _gate_rows(
    candidate: ReplayEval,
    base: ReplayEval,
    ctx: UpdateContext,
    *,
    mode: str,
    buyer_id: str | None = None,
) -> tuple[bool, float, float]:
    gains = _paired_values(candidate, base, key=_gate_key(ctx), buyer_id=buyer_id)
    return _gate_value(gains, mode, ctx.delta)


def _stage_diagnose(
    stage_obj: UpdateStage,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    ctx: UpdateContext,
) -> StageRecord:
    base = replay_eval(ctx.fit_traces, buyer_policies, seller_policies)
    pollution = _mean([row.pollution for row in base.rows])
    q_best_gain = 0.0
    q_errors = 0
    coverage_gaps = 0
    for trace in ctx.fit_traces:
        for trace_step in trace.steps:
            best = max(trace_step.single_success.values(), default=0.0)
            if best <= 0.0:
                coverage_gaps += 1
            selected = next(
                (row for row in base.rows if row.step_id == trace_step.step.id), None
            )
            if selected is not None and selected.success < best:
                q_errors += 1
            q_best_gain += best
    total = max(1, len(base.rows))
    ctx.diagnosis = {
        "pollution_rate": pollution,
        "coverage_gap_rate": coverage_gaps / total,
        "q_rank_error_rate": q_errors / total,
        "best_available_gain": q_best_gain / total,
    }
    return StageRecord(
        kind="diagnose",
        params=dict(stage_obj.params),
        edits=[],
        accepted=False,
        gain=0.0,
        lcb=0.0,
        n_candidates=1,
    )


def _stage_calibrate_q(
    stage_obj: UpdateStage,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    ctx: UpdateContext,
) -> StageRecord:
    label = str(stage_obj.param("label", "single_gain"))
    if ctx.mode == "no_probe":
        # Ablation: no counterfactual per-memory labels; only the selected
        # offer's outcome is visible.
        label = "reference_gain"
    ridge_lambda = float(stage_obj.param("ridge_lambda", 1.0))
    max_terms = int(stage_obj.param("max_terms", 6))
    gate_mode = str(stage_obj.param("gate", "lcb"))
    edits: list[dict[str, Any]] = []
    gain_total = 0.0
    lcb_min = 0.0
    n_candidates = 0
    accepted_any = False
    for buyer_id, policy in list(buyer_policies.items()):
        base = replay_eval(ctx.gate_traces, buyer_policies, seller_policies)
        fit = fit_q(
            ctx.fit_traces,
            buyer_id,
            policy,
            seller_policies,
            label=label,
            ridge_lambda=ridge_lambda,
            max_terms=max_terms,
        )
        n_candidates += 1
        if fit.n_rows == 0:
            continue
        candidate_policy = policy.copy_with(
            valuation=ast_from_qfit(fit, lineage=[qfit_detail(fit)]),
            lineage=[qfit_detail(fit)],
        )
        trial = dict(buyer_policies)
        trial[buyer_id] = candidate_policy
        candidate = replay_eval(ctx.gate_traces, trial, seller_policies)
        accepted, lcb, mean = _gate_rows(
            candidate, base, ctx, mode=gate_mode, buyer_id=buyer_id
        )
        if accepted:
            buyer_policies[buyer_id] = candidate_policy
            accepted_any = True
            gain_total += mean
            lcb_min = min(lcb_min, lcb) if edits else lcb
            edits.append(
                {
                    "buyer": buyer_id,
                    "detail": qfit_detail(fit),
                    "gain": round(mean, 4),
                    "lcb": round(lcb, 4),
                }
            )
    return StageRecord(
        kind="calibrate_q",
        params=dict(stage_obj.params),
        edits=edits,
        accepted=accepted_any,
        gain=gain_total / max(1, len(edits)),
        lcb=lcb_min,
        n_candidates=n_candidates,
    )


def _stage_repair_demand(
    stage_obj: UpdateStage,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    ctx: UpdateContext,
) -> StageRecord:
    requested = str(stage_obj.param("ops", "all"))
    gate_mode = str(stage_obj.param("gate", "lcb"))
    if requested == "all":
        keys = list(DEMAND_OP_LIBRARY)
    else:
        keys = [key for key in str(requested).split(",") if key in DEMAND_OP_LIBRARY]
    if ctx.diagnosis.get("coverage_gap_rate", 0.0) > 0.5:
        # If no memory helps at all, a stricter filter cannot increase
        # coverage; keep only relaxations and budget relief.
        keys = [
            key
            for key in keys
            if key in ("none", "budget15", "budget20")
        ]
    edits: list[dict[str, Any]] = []
    n_candidates = 0
    accepted_any = False
    gain_total = 0.0
    lcb_min = 0.0
    for buyer_id, policy in list(buyer_policies.items()):
        base = replay_eval(ctx.gate_traces, buyer_policies, seller_policies)
        best: tuple[float, float, str, NLBuyerPolicy] | None = None
        for key in keys:
            n_candidates += 1
            candidate_policy = policy.copy_with(
                demand_ops=DEMAND_OP_LIBRARY[key],
                lineage=[f"demand ops <- {key}"],
            )
            trial = dict(buyer_policies)
            trial[buyer_id] = candidate_policy
            candidate = replay_eval(ctx.gate_traces, trial, seller_policies)
            accepted, lcb, mean = _gate_rows(
                candidate, base, ctx, mode=gate_mode, buyer_id=buyer_id
            )
            if accepted and (best is None or (lcb, mean) > (best[0], best[1])):
                best = (lcb, mean, key, candidate_policy)
        if best is not None:
            lcb, mean, key, candidate_policy = best
            buyer_policies[buyer_id] = candidate_policy
            accepted_any = True
            gain_total += mean
            lcb_min = min(lcb_min, lcb) if edits else lcb
            edits.append(
                {
                    "buyer": buyer_id,
                    "detail": f"demand ops <- {key}",
                    "gain": round(mean, 4),
                    "lcb": round(lcb, 4),
                }
            )
    return StageRecord(
        kind="repair_demand",
        params=dict(stage_obj.params),
        edits=edits,
        accepted=accepted_any,
        gain=gain_total / max(1, len(edits)),
        lcb=lcb_min,
        n_candidates=n_candidates,
    )


def _stage_repair_quote(
    stage_obj: UpdateStage,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    ctx: UpdateContext,
) -> StageRecord:
    shadings = stage_obj.param("shadings", (0.0, 0.05, 0.1, 0.2))
    gate_mode = str(stage_obj.param("gate", "lcb"))
    edits: list[dict[str, Any]] = []
    n_candidates = 0
    accepted_any = False
    gain_total = 0.0
    lcb_min = 0.0
    for buyer_id, policy in list(buyer_policies.items()):
        base = replay_eval(ctx.gate_traces, buyer_policies, seller_policies)
        best: tuple[float, float, float, NLBuyerPolicy] | None = None
        for shading in tuple(float(value) for value in shadings):
            n_candidates += 1
            candidate_policy = policy.copy_with(
                quote=policy.quote.copy_with(
                    buyer_shading=shading,
                    lineage=[f"quote shading -> {shading:.2f}"],
                ),
                lineage=[f"quote shading -> {shading:.2f}"],
            )
            trial = dict(buyer_policies)
            trial[buyer_id] = candidate_policy
            candidate = replay_eval(ctx.gate_traces, trial, seller_policies)
            accepted, lcb, mean = _gate_rows(
                candidate, base, ctx, mode=gate_mode, buyer_id=buyer_id
            )
            if accepted and (best is None or (lcb, mean, shading) > (best[0], best[1], best[2])):
                best = (lcb, mean, shading, candidate_policy)
        if best is not None:
            lcb, mean, shading, candidate_policy = best
            buyer_policies[buyer_id] = candidate_policy
            accepted_any = True
            gain_total += mean
            lcb_min = min(lcb_min, lcb) if edits else lcb
            edits.append(
                {
                    "buyer": buyer_id,
                    "detail": f"quote shading -> {shading:.2f}",
                    "gain": round(mean, 4),
                    "lcb": round(lcb, 4),
                }
            )
    return StageRecord(
        kind="repair_quote",
        params=dict(stage_obj.params),
        edits=edits,
        accepted=accepted_any,
        gain=gain_total / max(1, len(edits)),
        lcb=lcb_min,
        n_candidates=n_candidates,
    )


def _stage_repair_seller(
    stage_obj: UpdateStage,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    ctx: UpdateContext,
) -> StageRecord:
    floors = stage_obj.param("floors", (0.0, 0.3, 0.5, 0.7))
    gate_mode = str(stage_obj.param("gate", "lcb"))
    mutator = SellerRPNMutator(int(ctx.settings.get("seed", 0)) + 31)
    edits: list[dict[str, Any]] = []
    n_candidates = 0
    accepted_any = False
    gain_total = 0.0
    lcb_min = 0.0
    for seller_id, policy in list(seller_policies.items()):
        base = replay_eval(ctx.gate_traces, buyer_policies, seller_policies)
        candidates: list[tuple[NLSellerPolicy, str]] = []
        for floor in tuple(float(value) for value in floors):
            candidates.append(
                (
                    policy.copy_with(
                        min_quality=floor,
                        lineage=[f"seller floor -> {floor:.2f}"],
                    ),
                    f"seller floor -> {floor:.2f}",
                )
            )
        for ast, detail in mutator.propose_from(policy.valuation, limit=3):
            candidates.append(
                (
                    policy.copy_with(valuation=ast, lineage=[detail]),
                    detail,
                )
            )
        best: tuple[float, float, str, NLSellerPolicy] | None = None
        for candidate_policy, detail in candidates:
            n_candidates += 1
            trial = dict(seller_policies)
            trial[seller_id] = candidate_policy
            candidate = replay_eval(ctx.gate_traces, buyer_policies, trial)
            accepted, lcb, mean = _gate_rows(candidate, base, ctx, mode=gate_mode)
            if accepted and (best is None or (lcb, mean) > (best[0], best[1])):
                best = (lcb, mean, detail, candidate_policy)
        if best is not None:
            lcb, mean, detail, candidate_policy = best
            seller_policies[seller_id] = candidate_policy
            accepted_any = True
            gain_total += mean
            lcb_min = min(lcb_min, lcb) if edits else lcb
            edits.append(
                {
                    "seller": seller_id,
                    "detail": detail,
                    "gain": round(mean, 4),
                    "lcb": round(lcb, 4),
                }
            )
    return StageRecord(
        kind="repair_seller",
        params=dict(stage_obj.params),
        edits=edits,
        accepted=accepted_any,
        gain=gain_total / max(1, len(edits)),
        lcb=lcb_min,
        n_candidates=n_candidates,
    )


def _stage_joint_seller_quote(
    stage_obj: UpdateStage,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    ctx: UpdateContext,
) -> StageRecord:
    """Atomically coordinate the seller floor and the buyer quote bridge.

    Neither stage is sufficient alone when a high-value memory is hidden by the
    seller floor *and* unaffordable under the current buyer shading.  This
    stage is the U-level answer to that coordination failure.
    """
    shadings = tuple(float(value) for value in stage_obj.param("shadings", (0.0, 0.05, 0.1, 0.2)))
    floors = tuple(float(value) for value in stage_obj.param("floors", (0.0, 0.3, 0.5, 0.7)))
    gate_mode = str(stage_obj.param("gate", "lcb"))
    base = replay_eval(ctx.gate_traces, buyer_policies, seller_policies)
    seller_ids = list(seller_policies.keys())
    best: tuple[float, float, float, tuple[float, ...], dict[str, NLBuyerPolicy], dict[str, NLSellerPolicy]] | None = None
    n_candidates = 0
    for shading in shadings:
        trial_buyers = {
            buyer_id: policy.copy_with(
                quote=policy.quote.copy_with(
                    buyer_shading=shading,
                    lineage=[f"joint: quote shading -> {shading:.2f}"],
                ),
                lineage=[f"joint: quote shading -> {shading:.2f}"],
            )
            for buyer_id, policy in buyer_policies.items()
        }
        for floor_combo in itertools.product(floors, repeat=len(seller_ids)):
            n_candidates += 1
            trial_sellers = {
                seller_id: seller_policies[seller_id].copy_with(
                    min_quality=floor,
                    lineage=[f"joint: seller floor -> {floor:.2f}"],
                )
                for seller_id, floor in zip(seller_ids, floor_combo)
            }
            candidate = replay_eval(ctx.gate_traces, trial_buyers, trial_sellers)
            accepted, lcb, mean = _gate_value(
                _paired_values(candidate, base, key=_gate_key(ctx)),
                gate_mode,
                ctx.delta,
            )
            if accepted and (best is None or (lcb, mean) > (best[0], best[1])):
                best = (lcb, mean, shading, floor_combo, trial_buyers, trial_sellers)
    edits: list[dict[str, Any]] = []
    if best is not None:
        lcb, mean, shading, floor_combo, trial_buyers, trial_sellers = best
        buyer_policies.update(trial_buyers)
        seller_policies.update(trial_sellers)
        edits.append(
            {
                "detail": (
                    f"joint: shading -> {shading:.2f}, floors -> "
                    + ", ".join(f"{sid}={floor:.2f}" for sid, floor in zip(seller_ids, floor_combo))
                ),
                "gain": round(mean, 4),
                "lcb": round(lcb, 4),
            }
        )
    return StageRecord(
        kind="joint_seller_quote",
        params=dict(stage_obj.params),
        edits=edits,
        accepted=bool(edits),
        gain=edits[0]["gain"] if edits else 0.0,
        lcb=edits[0]["lcb"] if edits else 0.0,
        n_candidates=n_candidates,
    )


def _stage_joint_demand_q(
    stage_obj: UpdateStage,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    ctx: UpdateContext,
) -> StageRecord:
    requested = str(stage_obj.param("ops", "all"))
    gate_mode = str(stage_obj.param("gate", "lcb"))
    label = "reference_gain" if ctx.mode == "no_probe" else str(
        stage_obj.param("label", "single_gain")
    )
    keys = (
        list(DEMAND_OP_LIBRARY)
        if requested == "all"
        else [key for key in str(requested).split(",") if key in DEMAND_OP_LIBRARY]
    )
    edits: list[dict[str, Any]] = []
    n_candidates = 0
    accepted_any = False
    gain_total = 0.0
    lcb_min = 0.0
    for buyer_id, policy in list(buyer_policies.items()):
        base = replay_eval(ctx.gate_traces, buyer_policies, seller_policies)
        best: tuple[float, float, str, NLBuyerPolicy] | None = None
        for key in keys:
            n_candidates += 1
            fit = fit_q(
                ctx.fit_traces,
                buyer_id,
                policy,
                seller_policies,
                label=label,
                ridge_lambda=float(stage_obj.param("ridge_lambda", 1.0)),
                max_terms=int(stage_obj.param("max_terms", 6)),
            )
            candidate_policy = policy.copy_with(
                demand_ops=DEMAND_OP_LIBRARY[key],
                valuation=ast_from_qfit(fit, lineage=[qfit_detail(fit)]),
                lineage=[f"joint demand ops <- {key}", qfit_detail(fit)],
            )
            trial = dict(buyer_policies)
            trial[buyer_id] = candidate_policy
            candidate = replay_eval(ctx.gate_traces, trial, seller_policies)
            accepted, lcb, mean = _gate_rows(
                candidate, base, ctx, mode=gate_mode, buyer_id=buyer_id
            )
            if accepted and (best is None or (lcb, mean) > (best[0], best[1])):
                best = (lcb, mean, key, candidate_policy)
        if best is not None:
            lcb, mean, key, candidate_policy = best
            buyer_policies[buyer_id] = candidate_policy
            accepted_any = True
            gain_total += mean
            lcb_min = min(lcb_min, lcb) if edits else lcb
            edits.append(
                {
                    "buyer": buyer_id,
                    "detail": f"joint demand ops <- {key} + ridge Q",
                    "gain": round(mean, 4),
                    "lcb": round(lcb, 4),
                }
            )
    return StageRecord(
        kind="joint_demand_q",
        params=dict(stage_obj.params),
        edits=edits,
        accepted=accepted_any,
        gain=gain_total / max(1, len(edits)),
        lcb=lcb_min,
        n_candidates=n_candidates,
    )


_STAGE_HANDLERS: dict[str, Callable[..., StageRecord]] = {
    "diagnose": _stage_diagnose,
    "calibrate_q": _stage_calibrate_q,
    "repair_demand": _stage_repair_demand,
    "repair_quote": _stage_repair_quote,
    "repair_seller": _stage_repair_seller,
    "joint_seller_quote": _stage_joint_seller_quote,
    "joint_demand_q": _stage_joint_demand_q,
}


def run_update_program(
    program: UpdateProgram,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    fit_traces: Sequence[EpisodeTrace],
    gate_traces: Sequence[EpisodeTrace],
    *,
    settings: dict[str, Any] | None = None,
) -> UpdateRun:
    buyers = {key: value.copy_with() for key, value in buyer_policies.items()}
    sellers_policies = {key: value.copy_with() for key, value in seller_policies.items()}
    ctx = UpdateContext(
        fit_traces=tuple(fit_traces),
        gate_traces=tuple(gate_traces),
        settings=settings or {},
    )
    records: list[StageRecord] = []
    total_candidates = 0
    for stage_obj in program.stages:
        handler = _STAGE_HANDLERS.get(stage_obj.kind)
        if handler is None:
            records.append(
                StageRecord(stage_obj.kind, dict(stage_obj.params), [], False, 0.0, 0.0, 0)
            )
            continue
        record = handler(stage_obj, buyers, sellers_policies, ctx)
        records.append(record)
        total_candidates += record.n_candidates
    return UpdateRun(buyers, sellers_policies, records, total_candidates)


# ----------------------------------------------------------------------
# Meta-level RSI: edit U itself
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MetaDiff:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    replaced: tuple[str, ...]
    params_changed: tuple[str, ...]
    description: str

    @property
    def is_structural(self) -> bool:
        return bool(self.added or self.removed or self.replaced)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "replaced": list(self.replaced),
            "params_changed": list(self.params_changed),
            "description": self.description,
            "is_structural": self.is_structural,
        }


def _diff_programs(old: UpdateProgram, new: UpdateProgram, description: str) -> MetaDiff:
    old_kinds = [stage_obj.kind for stage_obj in old.stages]
    new_kinds = [stage_obj.kind for stage_obj in new.stages]
    # Multiset difference, preserving order.
    remaining_new = list(new_kinds)
    removed: list[str] = []
    for kind in old_kinds:
        if kind in remaining_new:
            remaining_new.remove(kind)
        else:
            removed.append(kind)
    remaining_old = list(old_kinds)
    added: list[str] = []
    for kind in new_kinds:
        if kind in remaining_old:
            remaining_old.remove(kind)
        else:
            added.append(kind)
    replaced: list[str] = []
    if len(old.stages) == len(new.stages):
        for old_stage, new_stage in zip(old.stages, new.stages):
            if old_stage.kind != new_stage.kind:
                replaced.append(f"{old_stage.kind}->{new_stage.kind}")
    params_changed: list[str] = []
    for old_stage, new_stage in zip(old.stages, new.stages):
        if old_stage.kind == new_stage.kind and old_stage.params != new_stage.params:
            params_changed.append(old_stage.kind)
    return MetaDiff(
        added=tuple(added),
        removed=tuple(removed),
        replaced=tuple(replaced),
        params_changed=tuple(params_changed),
        description=description,
    )


class MetaMutator:
    """Typed structural edits over the update procedure U."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, program: UpdateProgram) -> list[tuple[UpdateProgram, MetaDiff]]:
        stages = list(program.stages)
        kinds = [stage_obj.kind for stage_obj in stages]
        out: list[tuple[UpdateProgram, MetaDiff]] = []

        def _add(new_stages: Sequence[UpdateStage], description: str) -> None:
            candidate = program.copy_with(new_stages, lineage=[description])
            out.append((candidate, _diff_programs(program, candidate, description)))

        # 1. Param: switch Q labels from selected-only to counterfactual gain.
        for index, stage_obj in enumerate(stages):
            if stage_obj.kind == "calibrate_q" and stage_obj.param("label") != "single_gain":
                modified = list(stages)
                modified[index] = stage_obj.with_params(label="single_gain")
                _add(modified, "calibrate_q: label -> single_gain")
                break

        # 2. Structural: insert demand-contract repair after a Q stage.
        if "repair_demand" not in kinds:
            for index, stage_obj in enumerate(stages):
                if stage_obj.kind in ("calibrate_q", "joint_demand_q"):
                    modified = list(stages)
                    modified.insert(
                        index + 1,
                        stage("repair_demand", ops="all", gate="strict"),
                    )
                    _add(modified, "insert repair_demand after Q calibration")
                    break

        # 3. Structural: insert seller-side repair.
        if "repair_seller" not in kinds:
            modified = list(stages)
            modified.append(stage("repair_seller", floors=(0.0, 0.3, 0.5, 0.7), gate="strict"))
            _add(modified, "insert repair_seller at end")

        # 4. Structural: replace sequential Q with an atomic joint update.
        for index, stage_obj in enumerate(stages):
            if stage_obj.kind == "calibrate_q":
                modified = list(stages)
                modified[index] = stage(
                    "joint_demand_q", ops="all", label="single_gain", gate="lcb"
                )
                _add(modified, "replace calibrate_q with joint_demand_q")
                break

        # 5. Structural: insert a diagnosis stage before demand repair.
        if "diagnose" not in kinds:
            for index, stage_obj in enumerate(stages):
                if stage_obj.kind == "repair_demand":
                    modified = list(stages)
                    modified.insert(index, stage("diagnose"))
                    _add(modified, "insert diagnose before repair_demand")
                    break

        # 6. Structural: swap adjacent stages (order changes the data flow).
        if len(stages) >= 2:
            for index in range(len(stages) - 1):
                if stages[index].kind != stages[index + 1].kind:
                    modified = list(stages)
                    modified[index], modified[index + 1] = (
                        modified[index + 1],
                        modified[index],
                    )
                    _add(
                        modified,
                        f"swap {stages[index].kind} <-> {stages[index + 1].kind}",
                    )
                    break

        # 6b. Structural: coordinate seller floor and quote bridge atomically.
        if "joint_seller_quote" not in kinds and (
            "repair_seller" in kinds or "repair_quote" in kinds
        ):
            modified = [
                stage_obj
                for stage_obj in stages
                if stage_obj.kind not in ("repair_seller", "repair_quote")
            ]
            modified.append(
                stage(
                    "joint_seller_quote",
                    shadings=(0.0, 0.05, 0.1, 0.2),
                    floors=(0.0, 0.3, 0.5, 0.7),
                    gate="strict",
                )
            )
            _add(modified, "replace quote/seller repair with joint_seller_quote")

        # 7. Structural: remove the quote bridge stage.
        if "repair_quote" in kinds and len(stages) > 1:
            modified = [stage_obj for stage_obj in stages if stage_obj.kind != "repair_quote"]
            _add(modified, "delete repair_quote")

        # 8. Structural: add the quote bridge if absent.
        if "repair_quote" not in kinds:
            modified = list(stages)
            modified.append(stage("repair_quote", shadings=(0.0, 0.1, 0.2), gate="strict"))
            _add(modified, "insert repair_quote at end")

        # 9. Param: tighten the acceptance gate.
        for index, stage_obj in enumerate(stages):
            if stage_obj.param("gate") == "mean":
                modified = list(stages)
                modified[index] = stage_obj.with_params(gate="lcb")
                _add(modified, f"{stage_obj.kind}: gate mean -> lcb")
                break

        # 10. Param: Q regularization strength.
        for index, stage_obj in enumerate(stages):
            if stage_obj.kind in ("calibrate_q", "joint_demand_q"):
                modified = list(stages)
                modified[index] = stage_obj.with_params(ridge_lambda=1.0)
                _add(modified, f"{stage_obj.kind}: ridge_lambda -> 1.0")
                break

        # 11. Param: allow more Q terms.
        for index, stage_obj in enumerate(stages):
            if stage_obj.kind in ("calibrate_q", "joint_demand_q"):
                modified = list(stages)
                modified[index] = stage_obj.with_params(max_terms=6)
                _add(modified, f"{stage_obj.kind}: max_terms -> 6")
                break

        return out


@dataclass
class MetaCandidate:
    program: UpdateProgram
    diff: MetaDiff
    accepted: bool
    gain: float
    lcb: float
    source: str = "meta_rsi"

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program.to_dict(),
            "diff": self.diff.to_dict(),
            "accepted": self.accepted,
            "gain": round(self.gain, 4),
            "lcb": round(self.lcb, 4),
            "source": self.source,
        }


@dataclass
class MetaRecord:
    adopted: UpdateProgram
    diff: MetaDiff | None
    meta_gain: float
    meta_lcb: float
    val_success_before: float
    val_success_after: float
    n_candidates: int
    candidates: list[MetaCandidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adopted_version": self.adopted.version,
            "adopted_name": self.adopted.name,
            "diff": self.diff.to_dict() if self.diff else None,
            "meta_gain": round(self.meta_gain, 4),
            "meta_lcb": round(self.meta_lcb, 4),
            "val_success_before": round(self.val_success_before, 4),
            "val_success_after": round(self.val_success_after, 4),
            "n_candidates": self.n_candidates,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def meta_step(
    program: UpdateProgram,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    *,
    fit_traces: Sequence[EpisodeTrace],
    gate_traces: Sequence[EpisodeTrace],
    validation_traces: Sequence[EpisodeTrace],
    settings: dict[str, Any],
) -> MetaRecord:
    """Propose, validate, and adopt an edit to the update procedure U.

    The acceptance signal is *verified task completion* of the policies that
    the candidate U produces, measured on held-out episodes/families.
    """
    mode = str(settings.get("mode", "meta_rsi"))
    delta = float(settings.get("meta_delta", settings.get("delta", 0.0)))
    base_run = run_update_program(
        program, buyer_policies, seller_policies, fit_traces, gate_traces,
        settings=settings,
    )
    base_val = replay_eval(validation_traces, base_run.buyer_policies, base_run.seller_policies)

    if mode == "u_fixed":
        return MetaRecord(
            adopted=program,
            diff=None,
            meta_gain=0.0,
            meta_lcb=0.0,
            val_success_before=base_val.success,
            val_success_after=base_val.success,
            n_candidates=0,
            candidates=[],
        )

    mutator = MetaMutator(int(settings.get("seed", 0)) + 77)
    proposals = mutator.propose(program)
    if mode == "meta_param":
        proposals = [
            item for item in proposals if not item[1].is_structural and item[1].params_changed
        ]
    elif mode == "random_meta":
        structural = [item for item in proposals if item[1].is_structural]
        proposals = structural or proposals

    evaluated: list[MetaCandidate] = []
    n_candidates = 0
    for candidate_program, diff in proposals:
        n_candidates += 1
        run = run_update_program(
            candidate_program, buyer_policies, seller_policies, fit_traces, gate_traces,
            settings=settings,
        )
        if mode == "self_validated":
            eval_traces = fit_traces
        else:
            eval_traces = validation_traces
        candidate_eval = replay_eval(eval_traces, run.buyer_policies, run.seller_policies)
        if mode == "self_validated":
            base_eval = replay_eval(
                fit_traces,
                base_run.buyer_policies,
                base_run.seller_policies,
            )
        else:
            base_eval = base_val
        gains = _paired_values(candidate_eval, base_eval, key="success")
        lcb = _lcb(gains)
        mean = _mean(gains)
        meta_gate = str(settings.get("meta_gate", "strict"))
        gate_ok, _, _ = _gate_value(gains, meta_gate, delta)
        accepted = (mode == "random_meta") or gate_ok
        evaluated.append(
            MetaCandidate(
                program=candidate_program,
                diff=diff,
                accepted=accepted,
                gain=mean,
                lcb=lcb,
                source=mode,
            )
        )

    accepted = [candidate for candidate in evaluated if candidate.accepted]
    chosen: MetaCandidate | None = None
    if accepted:
        if mode == "random_meta":
            chosen = mutator.rng.choice(accepted)
        else:
            # Prefer structural edits on ties: this is the N2 requirement, but
            # only among candidates that already pass held-out validation.
            accepted.sort(
                key=lambda candidate: (
                    candidate.diff.is_structural,
                    candidate.lcb,
                    candidate.gain,
                ),
                reverse=True,
            )
            chosen = accepted[0]
    adopted = chosen.program if chosen else program
    adopted_run = (
        run_update_program(
            adopted, buyer_policies, seller_policies, fit_traces, gate_traces,
            settings=settings,
        )
        if chosen
        else base_run
    )
    adopted_val = replay_eval(
        validation_traces, adopted_run.buyer_policies, adopted_run.seller_policies
    )
    val_gains = _paired_values(adopted_val, base_val, key="success")
    return MetaRecord(
        adopted=adopted,
        diff=chosen.diff if chosen else None,
        meta_gain=_mean(val_gains),
        meta_lcb=_lcb(val_gains),
        val_success_before=base_val.success,
        val_success_after=adopted_val.success,
        n_candidates=n_candidates,
        candidates=evaluated,
    )


# ----------------------------------------------------------------------
# Live long-horizon rollout (no oracle context)
# ----------------------------------------------------------------------


@dataclass
class LiveStep:
    episode_id: str
    step_id: str
    buyer_id: str
    family: str
    answer: str
    selected: tuple[str, ...]
    success: float
    pollution: float
    spend: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "buyer_id": self.buyer_id,
            "family": self.family,
            "answer": self.answer,
            "selected": list(self.selected),
            "success": self.success,
            "pollution": round(self.pollution, 4),
            "spend": round(self.spend, 4),
        }


@dataclass
class LiveEval:
    rows: tuple[LiveStep, ...]
    success: float
    final_success: float
    pollution: float
    spend: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": round(self.success, 4),
            "final_success": round(self.final_success, 4),
            "pollution": round(self.pollution, 4),
            "spend": round(self.spend, 4),
            "n_steps": len(self.rows),
            "rows": [row.to_dict() for row in self.rows],
        }


def run_episode_live(
    episode: CollabEpisode,
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    workflow: Any,
) -> LiveEval:
    """A real long-horizon rollout using the policy's own context.

    This is the end-to-end check: earlier mistakes propagate into later
    prompts, unlike the counterfactual replay used for U selection.
    """
    state = CollabState()
    sellers = {seller.id: seller for seller in episode.sellers}
    rows: list[LiveStep] = []
    for step in episode.ordered():
        context = state.context()
        buyer_policy = buyer_policies[step.buyer_id]
        demand = workflow.demand(step, buyer_policy.demand_instruction, context=context)
        contract = make_contract(step, buyer_policy.demand_ops)
        offers: list[MemoryOffer] = []
        for seller_id, seller in sellers.items():
            offers.extend(
                _seller_offers(
                    seller,
                    seller_policies[seller_id],
                    demand,
                    contract,
                    buyer_policy.quote,
                    workflow,
                )
            )
        offers = [offer for offer in offers if offer.record.facet == step.facet]
        by_id: dict[str, MemoryOffer] = {}
        for offer in offers:
            by_id.setdefault(offer.record.id, offer)
        offers = _buyer_value(list(by_id.values()), demand, workflow)
        selected = select_offers(
            contract, offers, buyer_policy.valuation, buyer_policy.quote
        )
        chosen = tuple(purchase.offer for purchase in selected)
        answer = workflow.answer_general(
            step.question,
            chosen,
            context=context,
            answer_format=ANSWER_FORMAT,
        )
        selected_ids = {offer.record.id for offer in chosen}
        covered = step.required_memory_ids & selected_ids
        coverage = len(covered) / max(1, len(step.required_memory_ids))
        success = 1.0 if coverage >= 1.0 and verify_number(answer, step.gold_answer) else 0.0
        pollution = sum(1 for offer in chosen if offer.record.is_decoy) / max(1, len(chosen))
        spend = float(sum(purchase.price for purchase in selected))
        rows.append(
            LiveStep(
                episode_id=episode.id,
                step_id=step.id,
                buyer_id=step.buyer_id,
                family=step.family,
                answer=answer,
                selected=tuple(selected_ids),
                success=success,
                pollution=pollution,
                spend=spend,
            )
        )
        state.results[step.id] = (answer, success)
    finals = [rows[-1].success] if rows else [0.0]
    return LiveEval(
        rows=tuple(rows),
        success=_mean([row.success for row in rows]),
        final_success=_mean(finals),
        pollution=_mean([row.pollution for row in rows]),
        spend=_mean([row.spend for row in rows]),
    )


def evaluate_live(
    episodes: Sequence[CollabEpisode],
    buyer_policies: dict[str, NLBuyerPolicy],
    seller_policies: dict[str, NLSellerPolicy],
    workflow: Any,
) -> LiveEval:
    rows: list[LiveStep] = []
    finals: list[float] = []
    for episode in episodes:
        evaluation = run_episode_live(episode, buyer_policies, seller_policies, workflow)
        rows.extend(evaluation.rows)
        finals.append(evaluation.final_success)
    return LiveEval(
        rows=tuple(rows),
        success=_mean([row.success for row in rows]),
        final_success=_mean(finals),
        pollution=_mean([row.pollution for row in rows]),
        spend=_mean([row.spend for row in rows]),
    )


def combine_live(evaluations: Sequence[LiveEval]) -> LiveEval:
    rows: list[LiveStep] = []
    finals: list[float] = []
    for evaluation in evaluations:
        rows.extend(evaluation.rows)
        finals.append(evaluation.final_success)
    return LiveEval(
        rows=tuple(rows),
        success=_mean([row.success for row in rows]),
        final_success=_mean(finals),
        pollution=_mean([row.pollution for row in rows]),
        spend=_mean([row.spend for row in rows]),
    )


# ----------------------------------------------------------------------
# Runner and N4 certificate
# ----------------------------------------------------------------------


@dataclass
class RoundRecord:
    round: int
    u_version: int
    update: dict[str, Any]
    val_before: float
    val_after: float
    meta: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "u_version": self.u_version,
            "update": self.update,
            "val_before": round(self.val_before, 4),
            "val_after": round(self.val_after, 4),
            "meta": self.meta,
        }


class TaskCollabRSIRunner:
    def __init__(self, settings: dict[str, Any], workflow: Any):
        self.settings = settings
        self.workflow = workflow
        self.traces: TraceBundle | None = None
        self.split: WorldSplit | None = None

    # -- per-mode trajectory -------------------------------------------
    def _run_mode(
        self,
        mode: str,
        split: WorldSplit,
        traces: TraceBundle,
    ) -> dict[str, Any]:
        seed = int(self.settings.get("seed", 0))
        rounds = int(self.settings.get("rounds", 4))
        calls_start = int(getattr(self.workflow, "calls", 0))
        failures_start = int(getattr(self.workflow, "failures", 0))
        first_episode = split.fit[0]
        buyers = seed_buyer_policies(
            first_episode, seed_q=str(self.settings.get("seed_q", "llm_utility"))
        )
        sellers = seed_seller_policies(first_episode)
        program = seed_update_program()
        records: list[RoundRecord] = []
        structural_edits: list[dict[str, Any]] = []
        u_lineage: list[dict[str, Any]] = [program.to_dict()]

        if mode == "fixed":
            seen = evaluate_live(split.test_seen, buyers, sellers, self.workflow)
            unseen = evaluate_live(split.test_unseen, buyers, sellers, self.workflow)
            final = combine_live([seen, unseen])
            return {
                "mode": mode,
                "seed": seed,
                "rounds": [],
                "u_versions": [program.to_dict()],
                "structural_edits": [],
                "final_test": final.to_dict(),
                "final_test_seen": seen.to_dict(),
                "final_test_unseen": unseen.to_dict(),
                "llm_calls": int(getattr(self.workflow, "calls", 0)) - calls_start,
                "llm_failures": int(getattr(self.workflow, "failures", 0)) - failures_start,
                "llm_usage": _usage_snapshot(self.workflow),
            }

        for round_index in range(rounds):
            settings = dict(self.settings)
            settings["mode"] = mode
            settings["seed"] = seed + round_index
            val_before = replay_eval(
                traces.validation, buyers, sellers
            ).success
            update_run = run_update_program(
                program,
                buyers,
                sellers,
                traces.fit,
                traces.gate,
                settings=settings,
            )
            val_after = replay_eval(
                traces.validation,
                update_run.buyer_policies,
                update_run.seller_policies,
            ).success
            meta_record = meta_step(
                program,
                buyers,
                sellers,
                fit_traces=traces.fit,
                gate_traces=traces.gate,
                validation_traces=traces.validation,
                settings=settings,
            )
            if meta_record.diff is not None and meta_record.diff.is_structural:
                structural_edits.append(meta_record.diff.to_dict())
            if meta_record.adopted.version != program.version:
                u_lineage.append(meta_record.adopted.to_dict())
            records.append(
                RoundRecord(
                    round=round_index,
                    u_version=program.version,
                    update=update_run.to_dict(),
                    val_before=val_before,
                    val_after=val_after,
                    meta=meta_record.to_dict(),
                )
            )
            buyers = update_run.buyer_policies
            sellers = update_run.seller_policies
            program = meta_record.adopted

        seen = evaluate_live(split.test_seen, buyers, sellers, self.workflow)
        unseen = evaluate_live(split.test_unseen, buyers, sellers, self.workflow)
        final = combine_live([seen, unseen])
        val_final = replay_eval(
            traces.validation, buyers, sellers
        ).success
        return {
            "mode": mode,
            "seed": seed,
            "rounds": [record.to_dict() for record in records],
            "u_versions": u_lineage,
            "structural_edits": structural_edits,
            "val_final": round(val_final, 4),
            "final_test": final.to_dict(),
            "final_test_seen": seen.to_dict(),
            "final_test_unseen": unseen.to_dict(),
            "buyer_policies": {key: value.to_dict() for key, value in buyers.items()},
            "seller_policies": {key: value.to_dict() for key, value in sellers.items()},
            "llm_calls": int(getattr(self.workflow, "calls", 0)) - calls_start,
            "llm_failures": int(getattr(self.workflow, "failures", 0)) - failures_start,
            "llm_usage": _usage_snapshot(self.workflow),
        }

    # -- public --------------------------------------------------------
    def run_seed(self, seed: int) -> dict[str, Any]:
        self.settings["seed"] = seed
        split = build_world_split(seed, n_steps=int(self.settings.get("steps", 10)))
        self.split = split
        traces = collect_trace_bundle(
            split,
            self.workflow,
            progress=bool(self.settings.get("verbose", True)),
        )
        self.traces = traces
        requested = list(self.settings.get("modes", ["fixed", "u_fixed", "meta_rsi"]))
        # The N4 claim needs both baselines; always run them.
        modes = list(dict.fromkeys(["fixed", "u_fixed"] + requested))
        results: dict[str, Any] = {}
        for mode in modes:
            results[mode] = self._run_mode(mode, split, traces)
            row = results[mode]["final_test"]
            print(
                f"[mode:{mode}] test_success={row['success']:.3f} "
                f"final={row['final_success']:.3f} "
                f"unseen={results[mode]['final_test_unseen']['success']:.3f} "
                f"u_v={len(results[mode]['u_versions']) - 1} "
                f"struct={len(results[mode]['structural_edits'])}",
                flush=True,
            )

        # Paired comparisons on the same test episodes.
        fixed = results["fixed"]
        u_fixed = results["u_fixed"]
        rsi = results.get("meta_rsi", u_fixed)
        gain_vs_fixed = _live_gains(rsi["final_test"], fixed["final_test"])
        gain_vs_u_fixed = _live_gains(rsi["final_test"], u_fixed["final_test"])
        unseen_vs_u_fixed = _live_gains(
            rsi["final_test_unseen"], u_fixed["final_test_unseen"]
        )
        meta_gains = [
            float(record["meta"]["meta_gain"])
            for record in rsi["rounds"]
            if record.get("meta") is not None
        ]
        structural = len(rsi["structural_edits"])
        validation_advantage = float(rsi.get("val_final", 0.0)) - float(
            u_fixed.get("val_final", 0.0)
        )
        certificate = {
            "n1_persistence": bool(rsi["u_versions"] and len(rsi["u_versions"]) >= 2),
            "n2_meta_level_u_changed": bool(structural >= 2),
            "structural_u_edits": int(structural),
            "n3_independent_validation": bool(
                gain_vs_u_fixed and _lcb(gain_vs_u_fixed) >= 0.05
            ),
            "n4_compounding": bool(
                meta_gains
                and _mean(meta_gains) > 0.0
                and validation_advantage > 0.0
            ),
            "n4_cross_family_transfer": bool(
                unseen_vs_u_fixed and _lcb(unseen_vs_u_fixed) > 0.0
            ),
            "gain_vs_fixed": round(_mean(gain_vs_fixed), 4),
            "gain_vs_fixed_lcb": round(_lcb(gain_vs_fixed), 4),
            "gain_vs_u_fixed": round(_mean(gain_vs_u_fixed), 4),
            "gain_vs_u_fixed_lcb": round(_lcb(gain_vs_u_fixed), 4),
            "unseen_gain_vs_u_fixed": round(_mean(unseen_vs_u_fixed), 4),
            "unseen_gain_vs_u_fixed_lcb": round(_lcb(unseen_vs_u_fixed), 4),
            "meta_gains": [round(value, 4) for value in meta_gains],
            "meta_gain_mean": round(_mean(meta_gains), 4),
            "meta_gain_slope": round(_slope(meta_gains), 4),
            "validation_advantage": round(validation_advantage, 4),
        }
        certificate["n4"] = bool(
            certificate["n4_compounding"] or certificate["n4_cross_family_transfer"]
        )
        certificate["passed"] = bool(
            certificate["n1_persistence"]
            and certificate["n2_meta_level_u_changed"]
            and certificate["n3_independent_validation"]
            and certificate["n4"]
        )
        return {
            "seed": seed,
            "settings": dict(self.settings),
            "results": results,
            "rsi_certificate": certificate,
            "trace_stats": _trace_stats(traces),
            "llm_calls": getattr(self.workflow, "calls", 0),
            "llm_failures": getattr(self.workflow, "failures", 0),
        }

    def run(self) -> dict[str, Any]:
        seeds = [int(seed) for seed in self.settings.get("seeds", [0])]
        results = [self.run_seed(seed) for seed in seeds]
        summary: dict[str, Any] = {}
        mode_names = list(results[0]["results"].keys()) if results else []
        for mode in mode_names:
            rows = [result["results"][mode] for result in results]
            summary[mode] = {
                "n_seeds": len(rows),
                "test_success": _mean([row["final_test"]["success"] for row in rows]),
                "test_final_success": _mean(
                    [row["final_test"]["final_success"] for row in rows]
                ),
                "test_pollution": _mean(
                    [row["final_test"]["pollution"] for row in rows]
                ),
                "seen_success": _mean(
                    [row["final_test_seen"]["success"] for row in rows]
                ),
                "unseen_success": _mean(
                    [row["final_test_unseen"]["success"] for row in rows]
                ),
                "val_final": _mean([row.get("val_final", 0.0) for row in rows]),
                "u_version": _mean([len(row["u_versions"]) - 1 for row in rows]),
                "structural_u_edits": _mean(
                    [len(row["structural_edits"]) for row in rows]
                ),
                "llm_calls": _mean([row["llm_calls"] for row in rows]),
                "llm_failures": _mean([row["llm_failures"] for row in rows]),
            }
        certs = [result["rsi_certificate"] for result in results]
        evidence = {
            "n_seeds": len(certs),
            "pass_rate": _mean([1.0 if cert["passed"] else 0.0 for cert in certs]),
            "n2_rate": _mean(
                [1.0 if cert["n2_meta_level_u_changed"] else 0.0 for cert in certs]
            ),
            "n3_rate": _mean(
                [1.0 if cert["n3_independent_validation"] else 0.0 for cert in certs]
            ),
            "n4_rate": _mean([1.0 if cert["n4"] else 0.0 for cert in certs]),
            "mean_gain_vs_fixed": _mean([cert["gain_vs_fixed"] for cert in certs]),
            "mean_gain_vs_u_fixed": _mean(
                [cert["gain_vs_u_fixed"] for cert in certs]
            ),
            "mean_unseen_gain_vs_u_fixed": _mean(
                [cert["unseen_gain_vs_u_fixed"] for cert in certs]
            ),
            "mean_meta_gain": _mean([cert["meta_gain_mean"] for cert in certs]),
        }
        if evidence["pass_rate"] >= 0.8 and evidence["n4_rate"] >= 0.8:
            evidence["claim"] = "n4_task_collaboration_rsi"
        elif evidence["n2_rate"] >= 0.8 and evidence["n3_rate"] >= 0.8:
            evidence["claim"] = "bounded_task_collaboration_rsi_without_n4"
        else:
            evidence["claim"] = "insufficient_evidence"
        evidence["single_seed_caveat"] = evidence["n_seeds"] < 2
        return {
            "settings": dict(self.settings),
            "results": results,
            "summary": summary,
            "rsi_evidence": evidence,
        }

    # -- persistence ---------------------------------------------------
    def save(self, payload: dict[str, Any], output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "seed", "mode", "test_success", "test_final_success",
                    "seen_success", "unseen_success", "val_final", "u_version",
                    "structural_u_edits", "n1", "n2", "n3", "n4",
                ]
            )
            for result in payload["results"]:
                cert = result["rsi_certificate"]
                for mode, row in result["results"].items():
                    writer.writerow(
                        [
                            result["seed"], mode, row["final_test"]["success"],
                            row["final_test"]["final_success"],
                            row["final_test_seen"]["success"],
                            row["final_test_unseen"]["success"],
                            row.get("val_final", 0.0),
                            len(row["u_versions"]) - 1,
                            len(row["structural_edits"]),
                            cert.get("n1_persistence", False),
                            cert.get("n2_meta_level_u_changed", False),
                            cert.get("n3_independent_validation", False),
                            cert.get("n4", False),
                        ]
                    )
        lines = [
            "# Task-collaboration-grounded N4 RSI",
            "",
            "Downstream signal: verified task completion gain from seller offer memory.",
            "Trade price/spend is a bridge, not the acceptance objective.",
            "",
            "| mode | test success | final success | seen | unseen | val (replay) | U version | structural U edits | LLM calls |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, row in payload["summary"].items():
            lines.append(
                f"| {mode} | {row['test_success']:.3f} | {row['test_final_success']:.3f} | "
                f"{row['seen_success']:.3f} | {row['unseen_success']:.3f} | "
                f"{row['val_final']:.3f} | {row['u_version']:.1f} | "
                f"{row['structural_u_edits']:.1f} | {row['llm_calls']:.0f} |"
            )
        evidence = payload["rsi_evidence"]
        lines += [
            "",
            "## RSI evidence",
            "",
            f"- claim: `{evidence['claim']}`",
            f"- N2 rate: {evidence['n2_rate']:.2f}",
            f"- N3 rate: {evidence['n3_rate']:.2f}",
            f"- N4 rate: {evidence['n4_rate']:.2f}",
            f"- mean gain vs fixed U: {evidence['mean_gain_vs_fixed']:+.4f}",
            f"- mean gain vs fixed-U trajectory: {evidence['mean_gain_vs_u_fixed']:+.4f}",
            f"- mean unseen-family gain vs fixed U: {evidence['mean_unseen_gain_vs_u_fixed']:+.4f}",
            f"- mean meta gain: {evidence['mean_meta_gain']:+.4f}",
            "",
            "The final test is a live long-horizon rollout.  Replay numbers are used only",
            "for inner policy selection and U-level validation.",
            "",
        ]
        lines += ["", "## U lineage (meta_rsi)", ""]
        for result in payload["results"]:
            rsi = result["results"].get("meta_rsi")
            if not rsi:
                continue
            lines.append(f"### seed {result['seed']}")
            lines.append("")
            for version in rsi.get("u_versions", []):
                kinds = " -> ".join(stage["kind"] for stage in version["stages"])
                lines.append(f"- U v{version['version']}: {kinds}")
            for round_record in rsi.get("rounds", []):
                meta = round_record.get("meta")
                if meta and meta.get("diff"):
                    lines.append(
                        f"- round {round_record['round']}: {meta['diff']['description']} "
                        f"(meta gain {meta['meta_gain']:+.3f}; "
                        f"val {meta['val_success_before']:.2f} -> {meta['val_success_after']:.2f})"
                    )
            for buyer, policy in rsi.get("buyer_policies", {}).items():
                ops = ", ".join(
                    f"{item['kind']}={item['value']}" for item in policy.get("demand_ops", [])
                ) or "none"
                lines.append(
                    f"- {buyer}: demand_ops=[{ops}]; "
                    f"valuation={policy['valuation']['tokens']}"
                )
            for seller, policy in rsi.get("seller_policies", {}).items():
                lines.append(
                    f"- {seller}: min_quality={policy['min_quality']}; "
                    f"valuation={policy['valuation']['tokens']}"
                )
            lines.append("")
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        (out / "u_lineage.json").write_text(
            json.dumps(
                [
                    {
                        "seed": result["seed"],
                        "u_versions": result["results"].get("meta_rsi", {}).get("u_versions", []),
                        "structural_edits": result["results"].get("meta_rsi", {}).get("structural_edits", []),
                    }
                    for result in payload["results"]
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return out


# ----------------------------------------------------------------------
# Small statistics helpers
# ----------------------------------------------------------------------


def _usage_snapshot(workflow: Any) -> dict[str, Any]:
    client = getattr(workflow, "client", None)
    usage = getattr(client, "usage", None)
    snapshot: dict[str, Any] = {}
    if usage is not None and hasattr(usage, "to_dict"):
        snapshot.update(usage.to_dict())
    failure_kinds = getattr(workflow, "failure_kinds", None)
    if failure_kinds:
        snapshot["failure_kinds"] = dict(failure_kinds)
    return snapshot


def _live_gains(candidate: dict[str, Any], base: dict[str, Any]) -> list[float]:
    base_by = {(row["episode_id"], row["step_id"]): row for row in base["rows"]}
    gains: list[float] = []
    for row in candidate["rows"]:
        reference = base_by.get((row["episode_id"], row["step_id"]))
        if reference is not None:
            gains.append(float(row["success"] - reference["success"]))
    return gains


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denominator = float(((x - x_mean) ** 2).sum())
    if denominator <= 1e-12:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denominator)


def _trace_stats(traces: TraceBundle) -> dict[str, Any]:
    all_traces = list(traces.all())

    def _stats(items: Sequence[EpisodeTrace]) -> dict[str, Any]:
        true_success = 0.0
        decoy_success = 0.0
        n_true = 0
        n_decoy = 0
        for trace in items:
            for step in trace.steps:
                for offer in step.offers:
                    value = step.single_success.get(offer.record.id, 0.0)
                    if offer.record.is_decoy:
                        n_decoy += 1
                        decoy_success += value
                    else:
                        n_true += 1
                        true_success += value
        return {
            "n_episodes": len(items),
            "n_steps": sum(len(trace.steps) for trace in items),
            "true_memory_probe_success": round(true_success / max(1, n_true), 4),
            "decoy_memory_probe_success": round(decoy_success / max(1, n_decoy), 4),
            "n_true_probes": n_true,
            "n_decoy_probes": n_decoy,
        }

    return {
        "fit": _stats(traces.fit),
        "gate": _stats(traces.gate),
        "validation": _stats(traces.validation),
        "all": _stats(all_traces),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task-collaboration-grounded N4 RSI pilot"
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--modes",
        default="fixed,u_fixed,meta_rsi",
        help="comma-separated: fixed,u_fixed,meta_rsi,meta_param,random_meta,self_validated,no_probe,surplus_gate",
    )
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--meta-delta", type=float, default=0.0)
    parser.add_argument(
        "--meta-gate",
        default="strict",
        choices=("strict", "mean", "lcb", "none"),
        help="U-level acceptance: strict = mean>0 and no step regressions",
    )
    parser.add_argument("--gate-signal", default="success", choices=("success", "surplus"))
    parser.add_argument(
        "--seed-q",
        default="llm_utility",
        choices=("llm_utility", "constant"),
        help="seed buyer Q: LLM utility (default) or constant (forces Q learning)",
    )
    parser.add_argument("--max-calls", type=int, default=3000)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "task_collab_n4_rsi"),
    )
    args = parser.parse_args(argv)
    _load_dotenv()
    if args.offline:
        workflow = NLWorkflow(None, use_llm=False)
    else:
        client = DeepSeekClient(
            LLMConfig(
                model=args.model,
                max_tokens=2500,
                temperature=0.0,
                max_calls=args.max_calls,
                cache_path=str(DEFAULT_RESULTS_DIR / "llm_cache_task_collab.json"),
            )
        )
        workflow = NLWorkflow(client, use_llm=True, json_retries=1)
    settings = {
        "seeds": [int(seed) for seed in args.seeds.split(",") if seed.strip()],
        "rounds": args.rounds,
        "steps": args.steps,
        "modes": [mode.strip() for mode in args.modes.split(",") if mode.strip()],
        "delta": args.delta,
        "meta_delta": args.meta_delta,
        "meta_gate": args.meta_gate,
        "gate_signal": args.gate_signal,
        "seed_q": args.seed_q,
        "model": "offline" if args.offline else args.model,
        "dataset": "task_collaboration_long_horizon",
        "decomposition": "fixed_gold_dag",
    }
    runner = TaskCollabRSIRunner(settings, workflow)
    payload = runner.run()
    out = runner.save(payload, args.output_dir)
    print(f"Saved: {out}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["rsi_evidence"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
