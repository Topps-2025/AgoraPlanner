"""Tests for the v9 protocol-level bounded-RI implementation."""

from __future__ import annotations

import pytest

from agora_planner.protocol_rsi import (
    BoundedRIRunner,
    ExperimentSettings,
    MetaController,
    OperatorLibrary,
    Protocol,
    ProtocolRegistry,
    ProtocolRunner,
    Transaction,
    TransactionLog,
    counterfactual_replay,
    initial_protocol,
)


def _settings(**overrides) -> ExperimentSettings:
    base = ExperimentSettings(
        seeds=(0,),
        meta_rounds=1,
        replay_seeds=2,
        candidates_per_round=2,
        n_memories=10,
        n_train_tasks=20,
        n_eval_tasks=20,
        redundancy_fraction=0.40,
        common_mode_risk_same_source=0.95,
        evaluation_common_mode_lambda=0.50,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _log_from_run(run, *, round_idx: int = 0) -> TransactionLog:
    log = TransactionLog()
    log.append(
        Transaction(
            round_idx=round_idx,
            protocol_version=run.protocol.version,
            seed=0,
            split="eval",
            gt_welfare=float(run.summary["gt_welfare"]),
            task_success=float(run.summary["task_success"]),
            min_utility=float(run.summary["gt_min_utility"]),
            error_correlation=float(run.summary["between_agent_error_corr"]),
            memory_count=int(run.summary["memory_count"]),
            total_cost=float(run.summary["total_cost"]),
            audit_flags=int(run.audit_flags),
            public_pool_size=int(run.summary["public_pool_size"]),
            transactions=int(run.transactions),
            common_mode_penalty=float(run.summary["common_mode_penalty"]),
            negative_edit_rate=float(run.negative_edit_rate),
        )
    )
    return log


# ----------------------------------------------------------------------
# Protocol / diff / registry
# ----------------------------------------------------------------------


def test_protocol_diff_distinguishes_structural_from_scalar():
    settings = _settings()
    parent = initial_protocol(settings)
    structural = parent.evolve(
        {"pricing_rule": "common_mode", "common_mode_lambda": 0.35},
        rationale="price correlation",
    )
    scalar = parent.evolve(
        {"common_mode_lambda": 0.50},
        rationale="tune lambda",
    )
    assert parent.diff(structural).is_structural
    assert "pricing_rule" in parent.diff(structural).structural_changes
    assert not parent.diff(scalar).is_structural
    assert "common_mode_lambda" in parent.diff(scalar).parameter_changes


def test_protocol_rejects_invalid_configurations():
    with pytest.raises(ValueError):
        Protocol(
            version=0,
            name="bad",
            verification_rule="audit",
            audit_prob=0.0,
        )
    with pytest.raises(ValueError):
        Protocol(
            version=0,
            name="bad",
            access_rule="open",
            one_per_source=True,
        )
    with pytest.raises(ValueError):
        Protocol(version=0, name="bad", payment_rule="unknown")


def test_registry_versions_and_lineage():
    settings = _settings()
    parent = initial_protocol(settings)
    registry = ProtocolRegistry(parent)
    child = parent.evolve(
        {"pricing_rule": "common_mode", "common_mode_lambda": 0.35},
        rationale="price correlation",
    )
    registry.register(child)
    registry.record_adoption(child, round_idx=0, evidence={"delta": 0.1})
    assert registry.current().version == child.version
    assert len(registry.adoptions) == 1
    lineage = registry.lineage()
    assert lineage[1]["parent_version"] == parent.version
    assert lineage[1]["structural"] is True


# ----------------------------------------------------------------------
# Runner / replay
# ----------------------------------------------------------------------


def test_protocol_runner_returns_consistent_metrics():
    settings = _settings()
    protocol = initial_protocol(settings)
    bench = settings.benchmark_factory(0)
    run = ProtocolRunner(
        bench,
        seed=0,
        evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
    ).run(protocol)
    assert set(run.allocation) == {agent.id for agent in bench.agents}
    assert run.summary["protocol_version"] == protocol.version
    assert run.summary["transactions"] == sum(
        len(bundle) for bundle in run.allocation.values()
    )
    assert "gt_welfare" in run.summary
    assert "between_agent_error_corr" in run.summary
    assert run.summary["common_mode_penalty"] >= 0.0


def test_replay_rejects_a_candidate_with_more_compute():
    settings = _settings()
    base = initial_protocol(settings)
    candidate = base.evolve(
        {"validation_seeds": 4},
        rationale="more validation seeds",
    )
    replay = counterfactual_replay(
        settings.benchmark_factory,
        base,
        candidate,
        seeds=[100],
        evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
    )
    assert replay.compute_matched is False
    assert "compute_budget_exceeds_base" in replay.constraint_violations
    assert replay.accepted is False


def test_replay_accepts_correlation_pricing_on_high_redundancy_benchmark():
    settings = _settings(n_memories=14)
    base = initial_protocol(settings)
    candidate = base.evolve(
        {"pricing_rule": "common_mode", "common_mode_lambda": 0.35},
        source="open_source",
        rationale="price common-mode correlation",
    )
    replay = counterfactual_replay(
        settings.benchmark_factory,
        base,
        candidate,
        seeds=[100, 137, 174],
        delta=0.002,
        evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
    )
    assert replay.mean_welfare_delta > 0.0
    assert replay.mean_delta > 0.0
    assert replay.accepted is True
    assert replay.diff.is_structural


def test_audit_flags_overconfident_bids():
    settings = _settings(
        n_memories=8,
        bid_noise=0.0,
        bid_overconfidence=1.0,
    )
    protocol = Protocol(
        version=0,
        name="noisy_audit",
        evidence_rule="noisy",
        payment_rule="vcg",
        access_rule="budget_only",
        verification_rule="audit",
        pricing_rule="none",
        update_rule="validated",
        common_mode_lambda=0.0,
        max_memories_per_agent=2,
        one_per_source=False,
        audit_prob=1.0,
        review_quorum=1,
    )
    bench = settings.benchmark_factory(0)
    run = ProtocolRunner(bench, seed=0).run(protocol)
    assert run.audited == len(bench.agents)
    assert run.audit_flags >= 1


def test_apply_update_runs_the_inner_memory_loop():
    settings = _settings(
        n_memories=10,
        n_train_tasks=16,
        n_eval_tasks=16,
    )
    protocol = initial_protocol(settings)
    bench = settings.benchmark_factory(0)
    run = ProtocolRunner(
        bench,
        seed=0,
        evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
    ).run(protocol, apply_update=True)
    assert run.store is not None
    assert "post_update_memory_count" in run.summary
    assert run.summary["post_update_memory_count"] >= 1
    assert run.negative_edit_rate >= 0.0
    assert run.update_logs


def test_meta_controller_adopts_a_structural_edit():
    settings = _settings(n_memories=14)
    base = initial_protocol(settings)
    bench = settings.benchmark_factory(0)
    run = ProtocolRunner(
        bench,
        seed=0,
        evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
    ).run(base)
    log = _log_from_run(run)
    controller = MetaController(
        settings.benchmark_factory,
        OperatorLibrary(include_open_source=True),
        replay_seeds=[100, 137],
        delta=0.002,
        max_candidates=6,
        structural_only=True,
        evaluation_common_mode_lambda=settings.evaluation_common_mode_lambda,
    )
    new_protocol, diagnosis, replays, best = controller.step(base, log, 0)
    assert diagnosis.tags
    assert replays
    assert best is not None
    assert best.accepted is True
    assert best.diff.is_structural is True
    assert new_protocol.version == base.version + 1


def test_adopted_protocol_is_reused_in_later_rounds():
    settings = _settings(
        meta_rounds=2,
        replay_seeds=2,
        candidates_per_round=4,
        n_memories=14,
    )
    payload = BoundedRIRunner(settings).run(modes=["meta_structural"])
    result = payload["results"][0]
    first = result["rounds"][0]
    if first["adopted"]:
        version = int(first["adopted_protocol_version"])
        assert int(result["rounds"][1]["evaluated_protocol_version"]) == version
        assert result["reuse"][str(version)] >= 1
    assert result["compounding"]["adoptions"] >= 0


def test_scalar_mode_only_adopts_parameter_edits():
    settings = _settings(
        meta_rounds=1,
        replay_seeds=2,
        candidates_per_round=4,
        n_memories=14,
    )
    payload = BoundedRIRunner(settings).run(modes=["meta_scalar"])
    for result in payload["results"]:
        for round_record in result["rounds"]:
            if round_record.get("adopted"):
                assert int(round_record.get("structural_edits", 0)) == 0


def test_multiple_seed_smoke_run():
    settings = ExperimentSettings(
        seeds=(0, 1),
        meta_rounds=1,
        replay_seeds=2,
        candidates_per_round=3,
        n_memories=10,
        n_train_tasks=16,
        n_eval_tasks=16,
        redundancy_fraction=0.40,
        common_mode_risk_same_source=0.95,
        evaluation_common_mode_lambda=0.50,
    )
    payload = BoundedRIRunner(settings).run(modes=["fixed", "meta_structural"])
    assert len(payload["results"]) == 4
    assert set(payload["summary"]) == {"fixed", "meta_structural"}
    for mode_summary in payload["summary"].values():
        assert mode_summary["n_seeds"] == 2
        assert "mean_test_gain" in mode_summary
        assert "mean_improvement_productivity" in mode_summary
