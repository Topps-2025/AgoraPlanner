"""Tests for the natural-language memory benchmark and memory-system adapters."""

from __future__ import annotations

import pytest

from agora_planner.embeddings import CosineRetriever, HashEmbedder
from agora_planner.memory_systems import (
    CAMAReimpl,
    GovMemReimpl,
    GraphRAGReimpl,
    MemGPTReimpl,
    MemPriceReimpl,
    RAPTORReimpl,
    RCRRouterReimpl,
)
from agora_planner.text_benchmark import TextBenchmarkConfig, TextMemoryBenchmark


def _bench(seed: int = 0) -> TextMemoryBenchmark:
    return TextMemoryBenchmark(
        TextBenchmarkConfig(
            seed=seed,
            n_agents=3,
            n_facets=10,
            facts_per_agent=3,
            n_public_facts=2,
            decoy_fraction=0.4,
            decoy_style="exact",
            n_eval_tasks=12,
            n_train_tasks=8,
            max_memories_per_agent=2,
            solver="beam",
        )
    )


def test_text_benchmark_is_deterministic():
    left = _bench(0)
    right = _bench(0)
    assert left.memories.ids() == right.memories.ids()
    assert [task.id for task in left.eval_tasks] == [task.id for task in right.eval_tasks]
    assert left.facet_codes == right.facet_codes


def test_text_benchmark_has_true_and_decoy_facts():
    bench = _bench(0)
    true_facts = [asset for asset in bench.memories if asset.metadata.get("is_true")]
    decoys = [asset for asset in bench.memories if asset.metadata.get("is_decoy")]
    assert true_facts
    assert decoys
    for asset in true_facts:
        assert asset.metadata["answer_code"] in asset.content
    for asset in decoys:
        assert asset.metadata["answer_code"] in asset.content


def test_retrieval_metrics_distinguish_true_from_decoy():
    bench = _bench(0)
    target = bench.queries[bench.eval_tasks[0].id]
    true_id = sorted(target.required_fact_ids)[0]
    decoy_id = next(iter(target.decoy_fact_ids), None)

    def retrieval_map(use_decoy_for_target: bool) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for task in bench.eval_tasks:
            query = bench.queries[task.id]
            if query.id == target.id and use_decoy_for_target and decoy_id is not None:
                mapping[query.id] = [bench.memories.get(decoy_id).content]
            else:
                gold_id = sorted(query.required_fact_ids)[0]
                mapping[query.id] = [bench.memories.get(gold_id).content]
        return mapping

    good = bench.evaluate_retrieval(retrieval_map(False), k=3)
    assert good["answer_accuracy@1"] == 1.0
    if decoy_id is not None:
        bad = bench.evaluate_retrieval(retrieval_map(True), k=3)
        assert bad["answer_accuracy@1"] < 1.0
        assert bad["decoy_rate@1"] > 0.0


def test_agent_value_rewards_true_facts_and_penalises_decoys():
    bench = _bench(0)
    query = bench.queries[bench.eval_tasks[0].id]
    agent_id = query.agent_id
    true_id = sorted(query.required_fact_ids)[0]
    decoy_id = next(iter(query.decoy_fact_ids), None)
    assert bench.agent_value(agent_id, {true_id}) > 0
    if decoy_id is not None:
        # A decoy cannot answer the query; its value is the negative usage cost.
        assert bench.agent_value(agent_id, {decoy_id}) < 0


@pytest.mark.parametrize(
    "system_factory",
    [
        lambda: MemGPTReimpl(),
        lambda: RAPTORReimpl(),
        lambda: GraphRAGReimpl(),
        lambda: CAMAReimpl(),
        lambda: GovMemReimpl(),
        lambda: RCRRouterReimpl(),
        lambda: MemPriceReimpl(),
    ],
)
def test_memory_systems_return_ranked_context(system_factory):
    bench = _bench(0)
    system = system_factory()
    system.ingest(list(bench.memories))
    if hasattr(system, "set_roles"):
        system.set_roles({agent.id: "researcher" for agent in bench.agents})
    query = bench.queries[bench.eval_tasks[0].id]
    items = system.search(query.agent_id, query.question, top_k=3)
    assert items
    assert len(items) <= 3
    assert all(hasattr(item, "text") for item in items)


def test_govmem_gate_returns_decisions():
    bench = _bench(0)
    gate = GovMemReimpl()
    gate.ingest(list(bench.memories))
    summary = gate.summary()
    assert summary["n_candidates"] == len(bench.memories)
    assert summary["promoted"] + summary["rejected"] + summary["needs_review"] == len(
        bench.memories
    )


def test_hash_embedder_ignores_access_codes():
    embedder = HashEmbedder()
    left = embedder.embed("The vault access code for facet alpha is K7Q2.")
    right = embedder.embed("The vault access code for facet alpha is M3P9.")
    assert (left == right).all()
