"""Tests for the LLM-backed bilateral buyer--seller RSI prototype.

These tests use a deterministic mock JSON client, so they do not call the
DeepSeek API.  The live API is exercised by
``experiments/run_llm_bilateral_rsi.py``.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from agora_planner.llm_bilateral_rsi import (
    BuyerState,
    Demand,
    LLMBilateralClient,
    LLMBilateralMarket,
    LLMBilateralRSIRunner,
    LLMMarketConfig,
    LLMRSISettings,
    LLMValue,
    ProgramMutator,
    SellerState,
    Transaction,
    UpdateOp,
    UpdateProgram,
    apply_op,
    diagnose,
    evaluate_program,
    run_episode,
    seed_program,
)


class MockJSONClient:
    available = True

    def __init__(self) -> None:
        self.calls = 0

    def json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        self.calls += 1
        text = messages[-1]["content"]
        if "buyer agent" in text:
            score = 6.0 if "compliant" in text else 4.0
            return {
                "semantic_value": score,
                "confidence": 0.7,
                "matched_facets": ["compliance", "supply"],
                "redundancy_risk": 0.2,
                "rationale": "mock buyer",
            }
        if "seller agent" in text:
            return {
                "cost_estimate": 1.1,
                "scarcity": 0.3,
                "quality": 0.6,
                "rationale": "mock seller",
            }
        return {"addendum": "Prefer specific evidence and discount stale sources."}


def _llm() -> LLMBilateralClient:
    return LLMBilateralClient(MockJSONClient(), max_tokens=64)


# ----------------------------------------------------------------------
# Market / valuation chain
# ----------------------------------------------------------------------


def test_market_uses_llm_for_both_sides():
    llm = _llm()
    market = LLMBilateralMarket(LLMMarketConfig(seed=0, n_steps=3), llm)
    txs = market.run(seed_program(), steps=3)
    assert len(txs) == 3
    assert all(tx.buyer_llm.used_llm for tx in txs)
    assert all(tx.seller_llm.used_llm for tx in txs)
    assert llm.calls >= 6


def test_valuation_is_a_function_of_demand_and_quote_follows_valuation():
    llm = _llm()
    market = LLMBilateralMarket(LLMMarketConfig(seed=0, n_steps=2), llm)
    buyer = market.buyers[0]
    memory = market.memories[0]
    llm_value = LLMValue(
        semantic_value=1.0,
        confidence=0.7,
        matched_facets=("compliance", "supply"),
        redundancy_risk=0.1,
    )
    buyer.demand_weights["compliance"] = 1.0
    value_high = buyer.valuation(llm_value)
    buyer.demand_weights["compliance"] = 0.0
    value_low = buyer.valuation(llm_value)
    assert value_high > value_low
    quote = buyer.quote(value_high)
    assert quote == pytest.approx(value_high * (1.0 - buyer.shading))


def test_update_ops_change_valuation_and_quote_parameters():
    llm = _llm()
    market = LLMBilateralMarket(LLMMarketConfig(seed=0, n_steps=1), llm)
    buyer = market.buyers[0]
    seller = market.sellers[0]
    memory = market.memories[0]
    tx = market.step(seed_program())
    old_scale = buyer.scale
    apply_op(UpdateOp("calibrate_buyer", {"lr": 0.2}), market, buyer, seller, tx)
    assert buyer.scale != old_scale
    old_shading = buyer.shading
    apply_op(UpdateOp("update_buyer_quote", {"lr": 0.1}), market, buyer, seller, tx)
    assert buyer.shading != old_shading or tx.traded


def test_portfolio_covers_buyer_and_seller():
    llm = _llm()
    market = LLMBilateralMarket(LLMMarketConfig(seed=0, n_steps=2), llm)
    metrics, _ = run_episode(seed_program(), LLMMarketConfig(seed=0, n_steps=2), llm=llm, steps=2)
    diagnosis = diagnose(metrics, 0)
    mutator = ProgramMutator(seed=3)
    proposals = mutator.portfolio(seed_program(), diagnosis, limit=6)
    kinds = {
        op.kind
        for candidate, _ in proposals
        for op in candidate.ops
        if op.kind != "update_buyer_demand_weights"
    }
    assert any("buyer" in k for k in kinds)
    assert any("seller" in k for k in kinds)


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------


def test_end_to_end_multi_round_with_mock():
    settings = LLMRSISettings(
        seeds=(0,),
        meta_rounds=2,
        meta_steps=3,
        meta_seeds=2,
        candidates=3,
        test_steps=3,
        test_seeds=2,
        test_shift=0.3,
    )
    llm = _llm()
    payload = LLMBilateralRSIRunner(settings, llm=llm).run(
        modes=["fixed", "meta_structural"]
    )
    assert set(payload["summary"]) == {"fixed", "meta_structural"}
    for result in payload["results"]:
        assert "final_program" in result
        assert "rsi_certificate" in result
        assert "heldout_test" in result
        assert math.isfinite(result["heldout_test"]["objective_gain"])
        assert result["rsi_certificate"]["rsi_level"] in {
            "bounded_rsi",
            "bounded_adaptation",
        }
    assert payload["llm_usage"]["calls"] > 0


def test_llm_prompt_revision_is_a_structural_candidate():
    from agora_planner.llm_bilateral_rsi import LLMMetaController

    settings = LLMRSISettings(
        seeds=(0,),
        meta_rounds=1,
        meta_steps=2,
        meta_seeds=1,
        candidates=2,
        test_steps=2,
        test_seeds=1,
    )
    llm = _llm()
    controller = LLMMetaController(
        settings.market_config(0),
        llm=llm,
        meta_seeds=[100],
        meta_steps=2,
        candidates=2,
        mutator_seed=0,
        use_prompt_candidate=True,
    )
    new_program, diagnosis, results, best = controller.step(seed_program(), 0)
    assert any(r.mutation.kind == "prompt_revision" for r in results)
    assert diagnosis.tags
