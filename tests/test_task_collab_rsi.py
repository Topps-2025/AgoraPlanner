"""Tests for the task-collaboration-grounded N4 RSI pilot."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.task_collab_rsi import (  # noqa: E402
    MetaMutator,
    TaskCollabRSIRunner,
    build_world_split,
    collect_trace_bundle,
    gold_context,
    replay_eval,
    run_update_program,
    seed_buyer_policies,
    seed_seller_policies,
    seed_update_program,
    stage,
    verify_number,
)


class ScriptedWorkflow:
    """Deterministic offline stand-in for the natural-language LLM workflow.

    It deliberately makes the seed LLM utility *prefer the decoy* (0.85 vs
    0.45), so the seed policy has a real memory-selection failure to fix.  The
    reader returns the gold number iff the selected memory is the required
    true memory, which makes task completion exactly a function of memory
    selection under the probe's gold prior context.
    """

    def __init__(self, episodes):
        self.registry = {step.id: step for episode in episodes for step in episode.steps}
        self.calls = 0
        self.failures = 0

    def demand(self, task, instruction, context: str = "") -> str:
        self.calls += 1
        return f"{task.question} Policy label: {task.label}."

    def retrieve(self, seller_id, demand, inventory, instruction, top_k):
        self.calls += 1
        ranked = sorted(inventory, key=lambda record: record.quality, reverse=True)
        return {
            "ranked_ids": [record.id for record in ranked[:top_k]],
            "relevance": {record.id: 0.9 for record in inventory},
        }

    def value(self, demand, offers):
        self.calls += 1
        return {
            offer.record.id: {
                "utility": 0.85 if offer.record.is_decoy else 0.45,
            }
            for offer in offers
        }

    def answer_general(
        self, question, selected, context: str = "", answer_format: str = ""
    ) -> str:
        self.calls += 1
        match = re.search(r"\[([^\]]+)\]", question)
        step = self.registry.get(match.group(1)) if match else None
        if step is None:
            return ""
        ids = {offer.record.id for offer in selected}
        if step.required_memory_ids & ids:
            return f"{step.gold_answer:.2f}"
        if step.decoy_memory_ids & ids:
            return f"{step.gold_answer + 100.0:.2f}"
        return f"{step.gold_answer + 50.0:.2f}"


@pytest.fixture(scope="module")
def split_and_workflow():
    split = build_world_split(0)
    workflow = ScriptedWorkflow(split.all_trace_episodes() + split.test_episodes())
    return split, workflow


@pytest.fixture(scope="module")
def traces(split_and_workflow):
    split, workflow = split_and_workflow
    return collect_trace_bundle(split, workflow)


def test_world_split_has_long_dependent_chain_and_heldout_families():
    split = build_world_split(0)
    assert len(split.fit) == 2
    assert len(split.gate) == 1
    assert len(split.validation) == 1
    assert len(split.test_seen) == 1
    assert len(split.test_unseen) == 1
    episode = split.fit[0]
    assert len(episode.steps) == 10
    assert episode.steps[0].depends_on == ()
    assert episode.steps[1].depends_on == (episode.steps[0].id,)
    assert split.validation[0].family == "budget_approval"
    assert split.test_unseen[0].family == "compliance_penalty"
    assert gold_context(episode, 2).count("result") == 2


def test_verify_number_handles_commas_and_tolerance():
    assert verify_number("1,234.56", 1234.56)
    assert verify_number("4.59", 4.59)
    assert not verify_number("4.60", 4.59, tolerance=0.001)
    assert not verify_number("no number", 1.0)


def test_trace_probes_separate_true_and_decoy(traces):
    stats = []
    for trace in traces.all():
        for step in trace.steps:
            for offer in step.offers:
                stats.append((offer.record.is_decoy, step.single_success[offer.record.id]))
    assert stats
    true_values = [value for is_decoy, value in stats if not is_decoy]
    decoy_values = [value for is_decoy, value in stats if is_decoy]
    assert sum(true_values) / len(true_values) == 1.0
    assert sum(decoy_values) / max(1, len(decoy_values)) == 0.0


def test_seed_policy_fails_and_q_calibration_improves(split_and_workflow, traces):
    split, _ = split_and_workflow
    buyers = seed_buyer_policies(split.fit[0])
    sellers = seed_seller_policies(split.fit[0])
    seed_run = run_update_program(
        seed_update_program(), buyers, sellers, traces.fit, traces.gate,
        settings={"seed": 0, "mode": "meta_rsi"},
    )
    seed_success = replay_eval(
        traces.gate, seed_run.buyer_policies, seed_run.seller_policies
    ).success
    assert seed_success == 0.0

    calibrated = seed_update_program().copy_with(
        [
            stage("calibrate_q", label="single_gain", ridge_lambda=1.0, max_terms=6, gate="strict"),
            stage("repair_quote", shadings=(0.0, 0.1), gate="lcb"),
        ],
        lineage=["test: counterfactual Q"],
    )
    run = run_update_program(
        calibrated, buyers, sellers, traces.fit, traces.gate,
        settings={"seed": 0, "mode": "meta_rsi"},
    )
    success = replay_eval(traces.gate, run.buyer_policies, run.seller_policies).success
    assert success >= 0.3
    q = run.buyer_policies["buyer0"].valuation.to_dict()
    assert "provenance" in q["tokens"]


def test_joint_seller_quote_unlocks_hidden_memory(split_and_workflow, traces):
    split, _ = split_and_workflow
    buyers = seed_buyer_policies(split.fit[0])
    sellers = seed_seller_policies(split.fit[0])
    calibrated = seed_update_program().copy_with(
        [
            stage("calibrate_q", label="single_gain", ridge_lambda=1.0, max_terms=6, gate="strict"),
            stage(
                "joint_seller_quote",
                shadings=(0.0, 0.05, 0.1),
                floors=(0.0, 0.3, 0.5, 0.7),
                gate="strict",
            ),
        ],
        lineage=["test: joint bridge"],
    )
    run = run_update_program(
        calibrated, buyers, sellers, traces.fit, traces.gate,
        settings={"seed": 0, "mode": "meta_rsi"},
    )
    success = replay_eval(traces.gate, run.buyer_policies, run.seller_policies).success
    assert success >= 0.7
    assert all(policy.min_quality < 0.5 for policy in run.seller_policies.values())


def test_meta_mutator_proposes_structural_u_edits():
    program = seed_update_program()
    proposals = MetaMutator(0).propose(program)
    structural = [diff for _, diff in proposals if diff.is_structural]
    assert structural
    assert any("joint_seller_quote" in diff.added for diff in structural)
    assert any("repair_demand" in diff.added for diff in structural)


def test_full_offline_run_reaches_n4_certificate(split_and_workflow):
    split, workflow = split_and_workflow
    runner = TaskCollabRSIRunner(
        {
            "seeds": [0],
            "rounds": 4,
            "steps": 10,
            "modes": ["fixed", "u_fixed", "meta_rsi"],
            "delta": 0.0,
            "meta_delta": 0.0,
            "gate_signal": "success",
            "model": "scripted",
        },
        workflow,
    )
    payload = runner.run()
    result = payload["results"][0]
    cert = result["rsi_certificate"]
    assert cert["n1_persistence"]
    assert cert["n2_meta_level_u_changed"]
    assert cert["structural_u_edits"] >= 2
    assert cert["n3_independent_validation"]
    assert cert["n4_cross_family_transfer"]
    assert cert["passed"]

    fixed = result["results"]["fixed"]["final_test"]
    rsi = result["results"]["meta_rsi"]["final_test"]
    unseen_fixed = result["results"]["fixed"]["final_test_unseen"]
    unseen_rsi = result["results"]["meta_rsi"]["final_test_unseen"]
    assert fixed["success"] == 0.0
    assert rsi["success"] == 1.0
    assert unseen_fixed["success"] == 0.0
    assert unseen_rsi["success"] == 1.0
    assert len(result["results"]["meta_rsi"]["u_versions"]) >= 2
