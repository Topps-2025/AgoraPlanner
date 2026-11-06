"""Smoke tests for the natural-language demand-first pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.demand_first_rsi import QuoteProgram, seed_rpn_ast  # noqa: E402
from agora_planner.nl_demand_first_rsi import (  # noqa: E402
    DEFAULT_DEMAND_INSTRUCTION,
    DEFAULT_RETRIEVAL_INSTRUCTION,
    NLBuyerPolicy,
    NLWorkflow,
    NLSellerPolicy,
    build_nl_world,
    evaluate_nl,
)


def test_nl_world_has_tasks_and_sellers():
    world = build_nl_world(0, 2)
    assert world.tasks
    assert world.sellers
    assert world.buyer_ids


def test_nl_offline_pipeline_runs():
    world = build_nl_world(0, 1)
    workflow = NLWorkflow(None, use_llm=False)
    buyer_policies = {
        bid: NLBuyerPolicy(DEFAULT_DEMAND_INSTRUCTION, seed_rpn_ast(), QuoteProgram())
        for bid in world.buyer_ids
    }
    seller_policies = {
        s.id: NLSellerPolicy(DEFAULT_RETRIEVAL_INSTRUCTION, top_k=3)
        for s in world.sellers
    }
    sellers = {s.id: s for s in world.sellers}
    result = evaluate_nl(
        world.tasks, buyer_policies, seller_policies, sellers, workflow
    )
    assert 0.0 <= result.success <= 1.0
    assert result.rows
