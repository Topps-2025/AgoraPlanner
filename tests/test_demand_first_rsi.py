"""Tests for the demand-first memory procurement RSI framework."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.demand_first_rsi import (  # noqa: E402
    DemandFirstRSIRunner,
    DemandFirstSettings,
    DemandOp,
    DemandProgram,
    DemandMutator,
    ASTMutator,
    ProgramBundle,
    ValuationAST,
    QuoteProgram,
    Task,
    evaluate_bundle,
    make_world,
    seed_bundle,
    seed_valuation_ast,
)


def _task() -> Task:
    return Task(
        id="t0",
        goal="test",
        required_facets=("compliance", "supply", "risk"),
        facet_weights={"compliance": 1.0, "supply": 0.8, "risk": 0.7},
        budget=8.0,
        max_bundle=3,
        source_limit=2,
        required_memory_ids=frozenset({"true_compliance", "true_supply", "true_risk"}),
        decoy_memory_ids=frozenset({"decoy_compliance", "decoy_supply", "decoy_risk"}),
    )


def test_demand_relative_ops_generalize_across_tasks():
    program = DemandProgram(
        ops=(
            DemandOp("include_all_required"),
            DemandOp("set_max_bundle_required"),
            DemandOp("set_source_limit_required"),
        )
    )
    demand = program.apply(_task())
    assert set(demand.facets) == {"compliance", "supply", "risk"}
    assert demand.max_bundle == 3
    assert demand.source_limit == 2


def test_seller_only_offers_requested_facets():
    world = make_world(0, 2)
    demand = DemandProgram(
        ops=(DemandOp("include_all_required"), DemandOp("set_max_bundle_required"))
    ).apply(world.tasks[0])
    offers = []
    for seller in world.sellers:
        offers.extend(seller.offers(demand, QuoteProgram()))
    assert offers
    assert {o.record.facet for o in offers}.issubset(set(demand.facets))


def test_ast_quality_and_provenance_prefer_true_memory():
    world = make_world(0, 1)
    task = world.tasks[0]
    demand = DemandProgram(
        ops=(DemandOp("include_all_required"), DemandOp("set_max_bundle_required"))
    ).apply(task)
    # Seed AST only sees facet match; it should often pick the cheaper decoy.
    seed_bundle_ = ProgramBundle(
        demand=DemandProgram(
            ops=(DemandOp("include_all_required"), DemandOp("set_max_bundle_required"))
        ),
        valuation=seed_valuation_ast(),
        quote=QuoteProgram(),
    )
    seed_eval = evaluate_bundle(seed_bundle_, [task], world.sellers)
    # Wrap with quality and provenance factors.
    mutator = ASTMutator()
    node = mutator._add_term(seed_valuation_ast().root, "quality", 0.8)
    node = mutator._add_term(node, "provenance", 0.6)
    better = ProgramBundle(seed_bundle_.demand, ValuationAST(node), QuoteProgram())
    better_eval = evaluate_bundle(better, [task], world.sellers)
    assert better_eval.coverage >= seed_eval.coverage
    assert better_eval.pollution <= seed_eval.pollution


def test_joint_rsi_beats_single_axis_on_pilot():
    runner = DemandFirstRSIRunner(
        DemandFirstSettings(
            seeds=(0,),
            rounds=2,
            meta_tasks=3,
            test_tasks=3,
            candidates=6,
            delta=0.01,
        )
    )
    payload = runner.run(
        modes=("fixed", "demand_only", "valuation_only", "joint")
    )
    summary = payload["summary"]
    assert summary["joint"]["task_success"] > summary["fixed"]["task_success"]
    assert summary["joint"]["task_success"] >= summary["demand_only"]["task_success"]
    assert summary["joint"]["task_success"] >= summary["valuation_only"]["task_success"]
    joint = next(r for r in payload["results"] if r["mode"] == "joint")
    assert joint["rsi_certificate"]["n4_synergy"] is True
    assert joint["rsi_certificate"]["passed"] is True
