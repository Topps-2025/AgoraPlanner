"""Tests for the public memory pool / commons layer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.demand_first_rsi import MemoryRecord  # noqa: E402
from agora_planner.nl_demand_first_rsi import NLWorkflow  # noqa: E402
from agora_planner.pool_math_experiment import run_config  # noqa: E402
from agora_planner.public_pool import MemoryPool, PoolPolicy  # noqa: E402


def _memory(mid: str = "m1", *, quality: float = 0.9, corr: float = 0.1) -> MemoryRecord:
    return MemoryRecord(
        id=mid,
        seller_id="sellerA",
        content="lemma",
        facet="f",
        source="internal",
        quality=quality,
        freshness=0.9,
        production_cost=0.1,
        correlation_risk=corr,
        is_true=True,
        is_decoy=False,
    )


def test_pool_requires_quality_and_review_quorum():
    pool = MemoryPool(PoolPolicy(review_quorum=2))
    try:
        pool.publish(_memory(quality=0.1))
    except ValueError:
        pass
    else:
        raise AssertionError("low-quality memory should not be publishable")
    entry = pool.publish(_memory())
    assert entry.review_status == "pending"
    pool.review("m1", "r1", 0.9)
    assert entry.review_status == "pending"
    pool.review("m1", "r2", 0.8)
    assert entry.review_status == "approved"
    assert len(pool.available(0)) == 1


def test_pool_embargo_and_source_cap():
    pool = MemoryPool(PoolPolicy(review_quorum=1, embargo_steps=2, source_cap=1))
    for i in range(2):
        mem = _memory(f"m{i}")
        pool.publish(mem, step=0)
        pool.review(f"m{i}", "r", 0.9)
    assert pool.available(0) == []
    available = pool.available(2)
    assert len(available) == 1


def test_pool_reputation_updates():
    pool = MemoryPool(PoolPolicy(review_quorum=1))
    pool.publish(_memory())
    pool.review("m1", "r", 0.9)
    before = pool.entries["m1"].reputation
    pool.record_usage(["m1"], True)
    assert pool.entries["m1"].reputation > before
    stats = pool.stats()
    assert stats["n_approved"] == 1


def test_ols_pool_vs_market_offline_shape():
    workflow = NLWorkflow(None, use_llm=False)
    market = run_config("market", 1, workflow)
    pool = run_config("pool", 1, workflow)
    assert market["n_steps"] == 10
    assert pool["n_steps"] == 10
    assert market["market_usage"] > 0
    assert pool["pool_usage"] > 0
    assert pool["pool_stats"]["n_approved"] == 10
