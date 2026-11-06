import numpy as np

from agora_planner.agent_market import (
    AgentBid,
    BidBook,
    BidBookValuator,
    BidEvidence,
    CalibratedValueModel,
    MarketRegulator,
    agreement,
    common_mode_risk,
    compute_consensus_metrics,
    error_correlation,
)
from agora_planner.auction import MemoryAuction
from agora_planner.benchmark import MemoryBenchmark
from agora_planner.config import BenchmarkConfig
from agora_planner.valuation import GroundTruthValuator
class _ConstantModel:
    def __init__(self, value: float):
        self._value = value

    def value(self, agent_id, bundle, *, split="eval"):
        return self._value if bundle else 0.0

    def marginal(self, agent_id, memory_id, bundle, *, split="eval"):
        return self.value(agent_id, bundle | {memory_id}, split=split) - self.value(
            agent_id, bundle, split=split
        )


def _bid(agent_id, bundle, value, *, probe=None):
    evidence = BidEvidence(probe_trace=probe or [], sources=["doc"])
    return AgentBid(agent_id=agent_id, bundle=frozenset(bundle), value=value, evidence=evidence)


def test_bid_book_and_valuator():
    book = BidBook()
    book.submit(_bid("a", {"m1"}, 2.0))
    book.submit(_bid("a", {"m1", "m2"}, 3.0))
    assert book.value("a", frozenset({"m1"})) == 2.0
    assert book.value("a", frozenset({"m2"})) == 0.0

    # fallback is optional; the core lookup must work without it
    valuator = BidBookValuator(book)
    assert valuator.value("a", frozenset({"m1"})) == 2.0
    assert valuator.marginal("a", "m2", frozenset({"m1"})) == 1.0
    assert valuator.bundle_values("a", [frozenset({"m1"})]) == {frozenset({"m1"}): 2.0}
    assert len(book) == 2


def test_regulator_budget_and_consistency():
    reg = MarketRegulator({"a": 2.5}, tolerance=0.01)
    ok, reason = reg.verify_budget(_bid("a", {"m1"}, 2.0))
    assert ok and reason == ""
    ok, reason = reg.verify_budget(_bid("a", {"m1"}, 3.0))
    assert not ok and "exceeds" in reason

    bids = [
        _bid("a", {"m1"}, 3.0),
        _bid("a", {"m1", "m2"}, 2.0),
    ]
    violations = reg.check_consistency(bids)
    assert len(violations) == 1
    assert "superset" in violations[0]


def test_regulator_audit_flags_inflation():
    reg = MarketRegulator({"a": 10.0}, tolerance=0.1)
    bid = _bid("a", {"m1"}, 5.0)
    report = reg.audit(bid, true_value_fn=lambda agent_id, bundle: 2.0)
    assert report.flagged
    assert report.inflated_by == 3.0
    assert reg.penalty_for(report) == 3.0

    honest = reg.audit(_bid("a", {"m1"}, 2.0), true_value_fn=lambda agent_id, bundle: 2.0)
    assert not honest.flagged
    assert reg.penalty_for(honest) == 0.0


def test_calibrated_value_model_updates():
    model = CalibratedValueModel(base=_ConstantModel(2.0))
    assert model.value("a", frozenset({"m1"})) == 2.0
    model.update(2.0, 1.0)
    model.update(4.0, 3.0)
    # y = x - 1
    assert abs(model.slope - 1.0) < 1e-6
    assert abs(model.intercept + 1.0) < 1e-6
    assert abs(model.value("a", frozenset({"m1"})) - 1.0) < 1e-6
    evidence = model.evidence("a", frozenset({"m1"}))
    assert evidence.probe_trace[0]["raw"] == 2.0


def test_consensus_metrics():
    answers = {
        "a": np.array([1.0, 1.0, 0.0]),
        "b": np.array([1.0, 0.0, 0.0]),
        "c": np.array([1.0, 1.0, 0.0]),
    }
    assert 0.0 <= agreement(answers) <= 1.0

    errors = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    corr = error_correlation(errors)
    assert -1.0 <= corr <= 1.0

    cm = common_mode_risk({"m1": 3, "m2": 1}, {"m1": 0.2, "m2": 0.5})
    assert cm == 0.2 * 3

    metrics = compute_consensus_metrics(answers, errors, {"m1": 2}, {"m1": 0.1})
    assert 0.0 <= metrics.consensus_quality <= 1.0
    assert metrics.common_mode_risk == 0.1


def test_auction_can_clear_on_agent_bids():
    bench = MemoryBenchmark(
        BenchmarkConfig(
            seed=7,
            n_agents=2,
            n_facets=4,
            n_memories=4,
            n_train_tasks=8,
            n_eval_tasks=8,
            max_memories_per_agent=2,
            solver="exact",
        )
    )
    truth = GroundTruthValuator(bench)
    probe_auction = MemoryAuction(bench.memories, bench.agents, truth, bench.config)
    book = BidBook()
    for agent in bench.agents:
        for bundle in probe_auction.feasible_bundles(agent.id):
            book.submit(
                AgentBid(
                    agent_id=agent.id,
                    bundle=bundle,
                    value=truth.value(agent.id, bundle),
                    evidence=BidEvidence(sources=["probe"]),
                )
            )
    auction = MemoryAuction(
        bench.memories, bench.agents, BidBookValuator(book), bench.config
    )
    result = auction.run(split="eval")
    assert result.allocation
    for agent in bench.agents:
        assert result.utilities[agent.id] >= -1e-6
