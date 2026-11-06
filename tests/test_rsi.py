from __future__ import annotations

from agora_planner.benchmark import MemoryBenchmark
from agora_planner.config import BenchmarkConfig, RSIConfig
from agora_planner.memory import MemoryAsset, merge_assets
from agora_planner.rsi import MarketGroundedRSI


def build_bench():
    return MemoryBenchmark(
        BenchmarkConfig(
            seed=5,
            n_agents=2,
            n_facets=4,
            n_memories=4,
            n_train_tasks=8,
            n_eval_tasks=8,
            max_memories_per_agent=2,
            solver="beam",
            beam_width=32,
        )
    )


def test_merge_of_poisoned_memory_stays_poisoned():
    left = MemoryAsset(
        id="a",
        content="good lesson",
        facet="f0",
        source_class="doc",
        quality=0.8,
        scope="research",
        version=1,
        owner="agent0",
        production_cost=0.5,
        metadata={},
    )
    right = MemoryAsset(
        id="b",
        content="bad lesson",
        facet="f0",
        source_class="doc",
        quality=0.9,
        scope="research",
        version=1,
        owner="agent0",
        production_cost=0.5,
        metadata={"poison": True},
    )
    transformation = merge_assets(left, right, new_id="merged")
    assert transformation.output is not None
    assert transformation.output.metadata.get("poison") is True
    assert transformation.output.quality <= 0.25


def test_rsi_step_is_consistent():
    bench = build_bench()
    rsi = MarketGroundedRSI(
        bench,
        RSIConfig(rounds=1, validation_seeds=2, delta=0.001, mdl_lambda=0.01, max_proposals=3),
        auction_config=bench.config,
        valuator_kind="ground_truth",
        seed=0,
    )
    store, record = rsi.step(bench.memories, 0, use_llm=False)
    assert len(store) <= len(bench.memories)
    payload = record.to_dict()
    assert "accepted" in payload
    assert "op" in payload
    if payload["accepted"]:
        assert payload["memory_count"] == len(store)
    else:
        assert payload["memory_count"] == len(bench.memories)


def test_self_validated_rsi_returns_logs():
    bench = build_bench()
    rsi = MarketGroundedRSI(
        bench,
        RSIConfig(rounds=1, validation_seeds=2, delta=0.001, mdl_lambda=0.01, max_proposals=3),
        auction_config=bench.config,
        valuator_kind="ground_truth",
        seed=0,
    )
    payload = rsi.run_self_validated(bench.memories, rounds=1)
    assert payload["mode"] == "self_validated"
    assert len(payload["rounds"]) == 1
    assert payload["final_memory_count"] <= len(bench.memories)
