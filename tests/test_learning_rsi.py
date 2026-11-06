"""Tests for the agent-level bounded-RSI prototype."""

from __future__ import annotations

import math

import pytest

from agora_planner.learning_rsi import (
    AgentRSIRunner,
    LearningRSIController,
    LearningRSISettings,
    MarketConfig,
    ProgramMutator,
    ProgramRegistry,
    UpdateOp,
    UpdateProgram,
    diagnose,
    evaluate_program,
    run_episode,
    seed_program,
)


def _tiny_settings(**overrides) -> LearningRSISettings:
    settings = LearningRSISettings(
        seeds=(0,),
        meta_rounds=1,
        meta_steps=20,
        meta_seeds=2,
        candidates=4,
        test_steps=20,
        test_seeds=1,
        test_shift=0.5,
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


# ----------------------------------------------------------------------
# Program mutation
# ----------------------------------------------------------------------


def test_structural_mutation_generates_a_new_program():
    program = seed_program()
    config = MarketConfig(seed=0, n_steps=20)
    base = evaluate_program(program, seeds=[100], config=config, steps=20)
    diagnosis = diagnose(base.metrics[0], 0)
    mutator = ProgramMutator(seed=7)
    candidate, mutation = mutator.mutate(program, diagnosis, structural=True)
    assert mutation.structural is True
    assert mutation.kind in {"insert", "delete", "replace", "swap", "wrap", "duplicate"}
    assert candidate.version == program.version + 1
    assert candidate.parent_version == program.version
    assert candidate.ops != program.ops
    assert candidate.lineage


def test_parameter_mutation_keeps_structure():
    program = seed_program()
    config = MarketConfig(seed=0, n_steps=20)
    base = evaluate_program(program, seeds=[100], config=config, steps=20)
    diagnosis = diagnose(base.metrics[0], 0)
    mutator = ProgramMutator(seed=3)
    candidate, mutation = mutator.mutate(program, diagnosis, structural=False)
    assert mutation.structural is False
    assert mutation.kind == "param"
    assert candidate.version == program.version + 1
    assert [op.kind for op in candidate.ops] == [op.kind for op in program.ops]


def test_structural_mutation_adds_or_removes_an_operator():
    program = seed_program()
    config = MarketConfig(seed=0, n_steps=20)
    base = evaluate_program(program, seeds=[100], config=config, steps=20)
    diagnosis = diagnose(base.metrics[0], 0)
    mutator = ProgramMutator(seed=0)
    candidate, mutation = mutator.mutate(program, diagnosis, structural=True)
    assert mutation.op_kind
    assert candidate.to_dict()["ops"]
    assert candidate.to_dict()["parent_version"] == program.version


# ----------------------------------------------------------------------
# Market and episode
# ----------------------------------------------------------------------


def test_market_runs_and_produces_transactions():
    config = MarketConfig(seed=0, n_steps=20)
    metrics, transactions = run_episode(seed_program(), config, steps=20)
    assert len(transactions) == 20
    assert 0.0 <= metrics.trade_rate <= 1.0
    assert metrics.buyer_mse >= 0.0
    assert metrics.seller_mse >= 0.0
    assert math.isfinite(metrics.objective)
    assert any(tx.traded for tx in transactions)


def test_episode_metrics_objective_is_finite_and_matches_formula():
    config = MarketConfig(seed=1, n_steps=20)
    metrics, _ = run_episode(seed_program(), config, steps=20)
    expected = (
        -(metrics.buyer_mse_rel + metrics.seller_mse_rel)
        - 0.5 * metrics.regret
        + 0.5 * metrics.surplus
        - 0.1 * metrics.instability
    )
    assert abs(metrics.objective - expected) < 1e-6


def test_trade_and_no_trade_cases_are_representable():
    # A very large shading/markup should produce no-trade; zero shading/markup
    # should produce trade.  We test the market mechanics directly.
    from agora_planner.learning_rsi import TwoSidedMarket

    config = MarketConfig(seed=0, n_steps=2)
    market = TwoSidedMarket(config)
    buyer = market.buyers[0]
    seller = market.sellers[0]
    buyer.w = buyer.theta.copy()
    buyer.bias = buyer.theta_intercept
    seller.z = seller.gamma.copy()
    seller.bias = seller.gamma_intercept
    buyer.shading = 100.0
    seller.markup = 0.0
    tx_no = market.step(seed_program())
    assert tx_no.traded is False
    buyer.shading = 0.0
    seller.markup = 0.0
    tx_yes = market.step(seed_program())
    assert tx_yes.traded is True


# ----------------------------------------------------------------------
# Controller and registry
# ----------------------------------------------------------------------


def test_registry_versions_and_lineage():
    program = seed_program()
    registry = ProgramRegistry(program)
    candidate = program.copy_with(
        program.ops + [UpdateOp("update_seller_valuation", {"rule": "sgd", "lr": 0.02})],
        version=1,
        rationale="add seller valuation",
    )
    registry.register(candidate)
    registry.record_adoption(candidate, round_idx=0, evidence={"delta": 1.0})
    assert registry.current().version == 1
    assert len(registry.adoptions) == 1
    assert registry.lineage()[1]["parent_version"] == 0
    assert "update_seller_valuation" in registry.lineage()[1]["ops"]


def test_controller_adopts_a_structural_edit():
    config = MarketConfig(seed=0, n_steps=20)
    controller = LearningRSIController(
        config,
        meta_seeds=[100, 137],
        meta_steps=20,
        candidates=6,
        delta=0.001,
        mutator_seed=17,
        structural_only=True,
    )
    program = seed_program()
    new_program, diagnosis, results, best = controller.step(program, 0)
    assert diagnosis.tags
    assert results
    assert best is not None
    assert best.mutation.structural is True
    assert best.accepted is True
    assert new_program.version == program.version + 1
    assert new_program.parent_version == program.version


def test_closed_library_is_buyer_only():
    config = MarketConfig(seed=0, n_steps=20)
    allowed = {
        "update_buyer_valuation",
        "update_buyer_quote",
        "normalize_buyer",
        "clamp_quotes",
    }
    controller = LearningRSIController(
        config,
        meta_seeds=[100],
        meta_steps=20,
        candidates=4,
        mutator_seed=5,
        allowed_kinds=tuple(allowed),
        structural_only=True,
    )
    program = seed_program()
    base = evaluate_program(program, seeds=[100], config=config, steps=20)
    diagnosis = diagnose(base.metrics[0], 0)
    for _ in range(10):
        candidate, _ = controller.mutator.mutate(program, diagnosis, structural=True)
        assert all(op.kind in allowed for op in candidate.ops)
        assert "update_seller_valuation" not in {
            op.kind for op in candidate.ops
        }


# ----------------------------------------------------------------------
# End-to-end modes
# ----------------------------------------------------------------------


def test_multiple_modes_run_end_to_end():
    settings = _tiny_settings()
    payload = AgentRSIRunner(settings).run(
        modes=[
            "fixed",
            "meta_structural",
            "meta_param",
            "random_mutation",
            "self_validated",
            "closed_library",
        ]
    )
    assert len(payload["results"]) == 6
    assert set(payload["summary"]) == {
        "fixed",
        "meta_structural",
        "meta_param",
        "random_mutation",
        "self_validated",
        "closed_library",
    }
    for result in payload["results"]:
        assert "initial_program" in result
        assert "final_program" in result
        assert "heldout_test" in result
        assert "registry" in result
        assert result["heldout_test"]["test_shift"] == 0.5
        assert math.isfinite(result["heldout_test"]["objective_gain"])


def test_meta_structural_reuses_adopted_program():
    settings = _tiny_settings(meta_rounds=2, candidates=6)
    payload = AgentRSIRunner(settings).run(modes=["meta_structural"])
    result = payload["results"][0]
    first = result["rounds"][0]
    if first["adopted"]:
        version = int(first["adopted_program_version"])
        assert int(result["rounds"][1]["evaluated_program_version"]) == version
        assert result["reuse"][str(version)] >= 1


def test_meta_structural_gets_bounded_rsi_certificate():
    settings = LearningRSISettings(
        seeds=(0,),
        meta_rounds=3,
        meta_steps=30,
        meta_seeds=3,
        candidates=8,
        test_steps=30,
        test_seeds=5,
        test_shift=0.5,
    )
    payload = AgentRSIRunner(settings).run(modes=["meta_structural"])
    result = payload["results"][0]
    certificate = result["rsi_certificate"]
    assert certificate["passed"] is True
    assert certificate["rsi_level"] == "bounded_rsi"
    assert certificate["n2_meta_level_generated"] is True
    assert certificate["structural_edits"] >= 2
    assert certificate["composition_synergy"] > 0.0
    assert len(result["program_signature"].split(">")) >= 2


def test_rsi_evidence_rejects_a_control_tie():
    summary = {
        "fixed": {"mean_objective_gain": 0.0, "rsi_pass_rate": 0.0},
        "meta_structural": {
            "mean_objective_gain": 10.0,
            "mean_gain_lcb": 5.0,
            "rsi_pass_rate": 1.0,
        },
        "meta_param": {"mean_objective_gain": 5.0, "rsi_pass_rate": 0.0},
        "random_mutation": {"mean_objective_gain": 10.0, "rsi_pass_rate": 0.0},
        "self_validated": {"mean_objective_gain": 1.0, "rsi_pass_rate": 0.0},
        "closed_library": {"mean_objective_gain": 6.0, "rsi_pass_rate": 0.0},
    }
    evidence = AgentRSIRunner.rsi_evidence(summary)
    assert evidence["claim"] == "bounded_adaptation"
    assert evidence["checks"]["beats_random_mutation"] is False
