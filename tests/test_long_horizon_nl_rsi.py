"""Smoke tests for the long-horizon / multi-hop diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.long_horizon_nl_rsi import (  # noqa: E402
    LongHorizonRSIRunner,
    NLWorkflow,
    RoundRobinAllocator,
    apply_allocator,
    build_math_diagnostic_world,
    evaluate_graphs,
    verify_answer,
)
from agora_planner.demand_first_rsi import DemandContract, DemandOp, QuoteProgram, RPNAST  # noqa: E402
from agora_planner.nl_demand_first_rsi import (  # noqa: E402
    DEFAULT_DEMAND_INSTRUCTION,
    DEFAULT_RETRIEVAL_INSTRUCTION,
    NLBuyerPolicy,
    NLSellerPolicy,
    SellerRPNMutator,
    apply_demand_ops,
    seed_seller_valuation,
)


def test_math_world_has_dependent_subtasks_and_balanced_buyers():
    world = build_math_diagnostic_world()
    assert len(world.meta_graph.subtasks) == 6
    assert len(world.test_graph.subtasks) == 6
    assert world.meta_graph.ordered()[0].id.endswith("s1")
    assert world.meta_graph.ordered()[-1].id.endswith("s6")
    for graph in (world.meta_graph, world.test_graph):
        counts = {bid: sum(1 for t in graph.subtasks if t.buyer_id == bid) for bid in world.buyer_ids}
        assert counts == {"buyer0": 3, "buyer1": 3}


def test_round_robin_allocator_is_external_and_balanced():
    world = build_math_diagnostic_world()
    graph = apply_allocator(world.meta_graph, RoundRobinAllocator(), world.buyer_ids)
    counts = {bid: sum(1 for t in graph.subtasks if t.buyer_id == bid) for bid in world.buyer_ids}
    assert counts == {"buyer0": 3, "buyer1": 3}
    assert graph.final_subtask_id == world.meta_graph.final_subtask_id


def test_demand_ops_change_the_contract_not_just_the_instruction():
    contract = DemandContract(
        task_id="x",
        goal="q",
        facets=("f",),
        facet_weights={"f": 1.0},
        budget=6.0,
        max_bundle=2,
        source_limit=2,
    )
    updated = apply_demand_ops(
        contract,
        (
            DemandOp("require_provenance", value=True),
            DemandOp("set_min_quality", value=0.5),
        ),
    )
    assert updated.require_provenance is True
    assert updated.min_quality == 0.5


def test_buyer_candidates_include_demand_edits():
    workflow = NLWorkflow(None, use_llm=False)
    runner = LongHorizonRSIRunner({"seeds": [0], "modes": ["joint"], "delta": 0.0}, workflow)
    policy = NLBuyerPolicy(
        DEFAULT_DEMAND_INSTRUCTION,
        RPNAST(("llm_utility", "facet_match", "MUL"), version=0, name="seed"),
        QuoteProgram(price_rule="ask"),
    )
    scopes = {scope for _, scope, _ in runner._buyer_candidates(policy)}
    assert "valuation_rpn" in scopes
    assert "demand_ops" in scopes
    assert "demand_rpn" in scopes


def test_verifier_modes():
    assert verify_answer("x**2 + 3*x", "x**2 + 3*x", "exact")
    assert verify_answer("x^2 + 3x", "x**2 + 3*x", "sympy")
    assert verify_answer("2.0", "2", "numeric")
    assert not verify_answer("3", "2", "numeric")


def test_offline_long_horizon_runs_and_certificate_shape():
    world = build_math_diagnostic_world()
    workflow = NLWorkflow(None, use_llm=False)
    buyer_policies = {
        bid: NLBuyerPolicy(
            DEFAULT_DEMAND_INSTRUCTION,
            RPNAST(("llm_utility", "facet_match", "MUL"), version=0, name="seed"),
            QuoteProgram(price_rule="ask"),
        )
        for bid in world.buyer_ids
    }
    seller_policies = {
        s.id: NLSellerPolicy(DEFAULT_RETRIEVAL_INSTRUCTION, top_k=5)
        for s in world.meta_sellers
    }
    sellers = {s.id: s for s in world.meta_sellers}
    evaluation = evaluate_graphs(
        [world.meta_graph], buyer_policies, seller_policies, sellers, workflow
    )
    assert len(evaluation.rows) == 6
    assert 0.0 <= evaluation.success <= 1.0
    assert 0.0 <= evaluation.pollution <= 1.0
    runner = LongHorizonRSIRunner({"seeds": [0], "modes": ["fixed"], "delta": 0.0}, workflow)
    assert runner is not None


def test_run_seed_fixed_returns_certificate():
    workflow = NLWorkflow(None, use_llm=False)
    runner = LongHorizonRSIRunner({"seeds": [0], "modes": ["fixed"], "delta": 0.0}, workflow)
    result = runner.run_seed(0, "fixed")
    assert "rsi_certificate" in result
    assert result["final_test"]["n_steps"] == 6
    assert result["rsi_certificate"]["passed"] is False


def test_seller_candidates_include_retrieval_and_valuation():
    workflow = NLWorkflow(None, use_llm=False)
    runner = LongHorizonRSIRunner({"seeds": [0], "modes": ["joint"], "delta": 0.0}, workflow)
    policy = NLSellerPolicy(DEFAULT_RETRIEVAL_INSTRUCTION, top_k=6)
    scopes = {scope for _, scope, _ in runner._seller_candidates(policy)}
    assert "seller_valuation_rpn" in scopes
    assert "retrieval_instruction" in scopes
    assert "min_quality" in scopes
    assert "top_k" in scopes


def test_seed_seller_valuation_matches_quality_heuristic():
    world = build_math_diagnostic_world()
    memory = world.meta_sellers[0].inventory[0]
    value = seed_seller_valuation().evaluate(memory)
    assert abs(value - memory.quality) < 1e-9
    candidates = SellerRPNMutator(0).propose_from(seed_seller_valuation())
    assert candidates
    assert all(0.0 <= ast.evaluate(memory) <= 1.0 for ast, _ in candidates)


def test_run_seed_se_parameter_tuning_is_a_baseline():
    workflow = NLWorkflow(None, use_llm=False)
    runner = LongHorizonRSIRunner({"seeds": [0], "modes": ["se"], "delta": 0.0}, workflow)
    result = runner.run_seed(0, "se")
    assert result["rounds"][0]["mode"] == "se_parameter_tuning"
    assert len(result["rounds"][0]["grid"]) == 8
    assert result["final_test"]["n_steps"] == 6
