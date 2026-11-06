"""Experiment orchestration for AgoraPlanner."""

from __future__ import annotations

import csv
import json
import logging
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .agents import Auctioneer, ConsumerAgent, ProducerAgent, VerifierAgent
from .auction import MemoryAuction
from .baselines import baseline_allocations as build_baseline_allocations
from .benchmark import MemoryBenchmark
from .config import BenchmarkConfig, ExperimentConfig
from .llm import DeepSeekClient, LLMError
from .metrics import oracle_welfare, summarize_allocation, summarize_auction
from .rsi import MarketGroundedRSI
from .valuation import GroundTruthValuator, make_valuator

LOGGER = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config or ExperimentConfig()
        self.client: DeepSeekClient | None = None
        if self.config.llm.enabled:
            self.client = DeepSeekClient(self.config.llm)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def build_benchmark(self, seed: int) -> MemoryBenchmark:
        bench_config = replace(self.config.benchmark, seed=seed)
        return MemoryBenchmark(bench_config)

    def make_valuator(self, bench: MemoryBenchmark, kind: str, seed: int):
        return make_valuator(
            kind, bench, bench.config, client=self.client, seed=seed
        )

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    def _greedy_agent_bundle(
        self,
        bench: MemoryBenchmark,
        agent_id: str,
        *,
        split: str = "eval",
        threshold: float = 1e-4,
    ) -> frozenset[str]:
        """A strong per-agent baseline: greedy marginal-value selection."""

        agent = bench._agent(agent_id)  # noqa: SLF001
        bundle: frozenset[str] = frozenset()
        remaining = set(bench.memories.ids())
        while len(bundle) < agent.max_memories and remaining:
            best_mid = None
            best_gain = threshold
            for mid in sorted(remaining):
                asset = bench.memories.get(mid)
                if bench.config.one_per_source and any(
                    bench.memories.get(x).source_class == asset.source_class
                    for x in bundle
                ):
                    continue
                gain = bench.memory_marginal_value(
                    agent_id, mid, bundle, split=split
                )
                if gain > best_gain:
                    best_gain = gain
                    best_mid = mid
            if best_mid is None:
                break
            bundle = bundle | {best_mid}
            remaining.remove(best_mid)
        return bundle

    def _quality_bundle(
        self, bench: MemoryBenchmark, agent_id: str, k: int
    ) -> frozenset[str]:
        """Top-k by quality, with source diversity (a budget-matched shared pool)."""

        scores = sorted(
            [(asset.quality, asset) for asset in bench.memories],
            key=lambda x: x[0],
            reverse=True,
        )
        bundle: set[str] = set()
        used_sources: set[str] = set()
        for _, asset in scores:
            if len(bundle) >= k:
                break
            if bench.config.one_per_source and asset.source_class in used_sources:
                continue
            bundle.add(asset.id)
            used_sources.add(asset.source_class)
        return frozenset(bundle)

    def _top_k_bundle(
        self, bench: MemoryBenchmark, agent_id: str, k: int, *, split: str = "eval"
    ) -> frozenset[str]:
        tasks = bench._tasks(split)  # noqa: SLF001
        # Relevance of a memory: quality times the task-weight mass of its facet.
        mass: dict[str, float] = {facet: 0.0 for facet in bench.facets}
        for task in tasks:
            for facet, weight in task.required_facets.items():
                mass[facet] = mass.get(facet, 0.0) + weight
        scores = []
        for asset in bench.memories:
            scores.append((asset.quality * (0.2 + mass.get(asset.facet, 0.0)), asset))
        scores.sort(key=lambda x: x[0], reverse=True)
        bundle: set[str] = set()
        used_sources: set[str] = set()
        for _, asset in scores:
            if len(bundle) >= k:
                break
            if bench.config.one_per_source and asset.source_class in used_sources:
                continue
            bundle.add(asset.id)
            used_sources.add(asset.source_class)
        return frozenset(bundle)

    def baseline_allocations(self, bench: MemoryBenchmark) -> dict[str, dict[str, frozenset[str]]]:
        """Run the reproducible baseline suite on one benchmark instance.

        The suite covers allocation baselines (no memory / private / shared-all /
        budget-matched shared / top-k / selective / central planner), a
        composition proxy (RCR + CAMA + GovMem + MemPrice), and lightweight
        Mem0 / MemGPT / RAPTOR / GraphRAG policy proxies.  The proxies are
        labelled as such; official systems require their own adapters.
        """

        return build_baseline_allocations(
            bench,
            seed=bench.config.seed,
            include_proxies=True,
            include_oracle=True,
        )

    # ------------------------------------------------------------------
    # Market
    # ------------------------------------------------------------------
    def run_market(
        self, bench: MemoryBenchmark, *, kind: str | None = None, seed: int = 0
    ) -> tuple[Any, dict[str, Any]]:
        kind = kind or self.config.benchmark.valuation
        valuator = self.make_valuator(bench, kind, seed)
        auction = MemoryAuction(
            bench.memories, bench.agents, valuator, bench.config
        )
        result = auction.run(split="eval")
        summary = summarize_auction(bench, result)
        return result, summary

    def run_seed(self, seed: int, *, run_rsi: bool = True) -> dict[str, Any]:
        bench = self.build_benchmark(seed)
        result: dict[str, Any] = {
            "seed": seed,
            "benchmark": bench.describe(),
            "baselines": {},
        }
        for name, allocation in self.baseline_allocations(bench).items():
            result["baselines"][name] = summarize_allocation(bench, allocation)
        auction_result, auction_summary = self.run_market(bench, seed=seed)
        result["auction"] = auction_summary
        result["auction_result"] = auction_result.to_dict()

        if run_rsi:
            rsi = MarketGroundedRSI(
                bench,
                self.config.rsi,
                auction_config=bench.config,
                valuator_kind=self.config.benchmark.valuation,
                client=self.client,
                seed=seed,
            )
            result["rsi"] = rsi.run(bench.memories)
            if self.config.rsi.self_validated_baseline:
                result["rsi_self_validated"] = rsi.run_self_validated(bench.memories)
        return result

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def _metric(summary: dict[str, Any], key: str) -> float:
        value = summary.get(key, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def aggregate(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        methods: dict[str, list[dict[str, Any]]] = {}
        for item in results:
            for name, summary in item.get("baselines", {}).items():
                methods.setdefault(name, []).append(summary)
            if "auction" in item:
                methods.setdefault("auction", []).append(item["auction"])

        keys = [
            "task_success",
            "gt_welfare",
            "gt_avg_utility",
            "gt_min_utility",
            "memory_count",
            "redundancy_rate",
            "zero_price_rate",
            "independence_index",
            "team_redundancy",
            "between_agent_error_corr",
            "total_cost",
        ]
        aggregate: dict[str, Any] = {}
        for method, summaries in methods.items():
            aggregate[method] = {}
            for key in keys:
                values = [self._metric(s, key) for s in summaries]
                aggregate[method][key] = {
                    "mean": float(np.mean(values)) if values else 0.0,
                    "std": float(np.std(values)) if values else 0.0,
                    "values": [round(v, 6) for v in values],
                }
        return aggregate

    # ------------------------------------------------------------------
    # Public runs
    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        results = []
        for seed in self.config.seeds:
            LOGGER.info("Running seed %s", seed)
            results.append(self.run_seed(seed))
        aggregate = self.aggregate(results)
        payload = {
            "name": self.config.name,
            "config": self.config.to_dict(),
            "results": results,
            "aggregate": aggregate,
            "llm_usage": self.client.usage.to_dict() if self.client else {},
        }
        self.save_json(payload, f"{self.config.name}_results.json")
        self.save_summary_csv(aggregate, f"{self.config.name}_summary.csv")
        self.print_summary(aggregate)
        return payload

    def save_json(self, payload: dict[str, Any], filename: str) -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path

    def save_summary_csv(self, aggregate: dict[str, Any], filename: str) -> Path:
        path = self.output_dir / filename
        keys = [
            "task_success",
            "gt_welfare",
            "gt_avg_utility",
            "gt_min_utility",
            "memory_count",
            "redundancy_rate",
            "zero_price_rate",
            "independence_index",
            "team_redundancy",
            "between_agent_error_corr",
            "total_cost",
        ]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["method"] + keys)
            for method, metrics in aggregate.items():
                row = [method]
                for key in keys:
                    stats = metrics.get(key, {})
                    row.append(
                        f"{stats.get('mean', 0.0):.4f}±{stats.get('std', 0.0):.4f}"
                    )
                writer.writerow(row)
        return path

    @staticmethod
    def print_summary(aggregate: dict[str, Any]) -> None:
        keys = [
            "task_success",
            "gt_welfare",
            "redundancy_rate",
            "zero_price_rate",
            "independence_index",
            "between_agent_error_corr",
            "memory_count",
        ]
        header = "method".ljust(16) + "".join(k[:16].rjust(18) for k in keys)
        print("\n" + header)
        print("-" * len(header))
        for method, metrics in aggregate.items():
            row = method.ljust(16)
            for key in keys:
                stats = metrics.get(key, {})
                row += f"{stats.get('mean', 0.0):>10.4f}±{stats.get('std', 0.0):<7.4f}"
            print(row)
        print()

    # ------------------------------------------------------------------
    # LLM demo
    # ------------------------------------------------------------------
    def run_llm_demo(self, *, max_calls: int = 12, seed: int = 0) -> dict[str, Any]:
        """A small end-to-end demo that actually calls DeepSeek flash.

        The LLM writes memory content, explains one winning bid, and proposes
        one RSI edit.  The auction itself uses the noisy additive bidder so that
        the number of API calls stays small; the ground-truth benchmark remains
        the auditor.
        """

        if self.client is None:
            raise RuntimeError("LLM is disabled in the config")
        self.client.config.max_calls = max_calls
        bench_config = replace(
            self.config.benchmark,
            seed=seed,
            n_agents=3,
            n_facets=8,
            n_memories=9,
            n_train_tasks=16,
            n_eval_tasks=16,
            max_memories_per_agent=2,
        )
        bench = MemoryBenchmark(bench_config)
        producer = ProducerAgent(
            bench.agents[0], bench, bench.config, client=self.client, seed=seed
        )
        produced = []
        for i, task in enumerate(bench.train_tasks[:3]):
            asset = producer.produce(
                task, round_idx=0, memory_id=f"llm_m{i:03d}"
            )
            try:
                bench.memories.add(asset)
                produced.append(asset)
            except ValueError:
                continue
        for asset in produced:
            bench.agents[0].produced.append(asset.id)

        auction_result, auction_summary = self.run_market(
            bench, kind="additive_noisy", seed=seed
        )
        consumer = ConsumerAgent(
            bench.agents[0],
            bench,
            self.make_valuator(bench, "additive_noisy", seed),
            client=self.client,
        )
        rationale = ""
        best_agent = max(
            auction_result.allocation,
            key=lambda aid: len(auction_result.allocation[aid]),
            default=bench.agents[0].id,
        )
        winning_bundle = auction_result.allocation.get(best_agent, frozenset())
        if winning_bundle:
            rationale = consumer.rationale(winning_bundle)

        verifier = VerifierAgent(bench, seed=seed)
        audits = []
        for agent in bench.agents:
            bundle = auction_result.allocation.get(agent.id, frozenset())
            if not bundle:
                continue
            bid = ConsumerAgent(
                agent,
                bench,
                self.make_valuator(bench, "additive_noisy", seed),
            ).bid(bundle)
            audits.append(verifier.audit_bid(bid))

        rsi = MarketGroundedRSI(
            bench,
            replace(self.config.rsi, rounds=1, validation_seeds=2, max_proposals=2),
            auction_config=bench.config,
            valuator_kind="additive_noisy",
            client=self.client,
            seed=seed,
        )
        rsi_payload = rsi.run(bench.memories, rounds=1, use_llm=True)

        payload = {
            "mode": "llm_demo",
            "model": self.client.model,
            "llm_usage": self.client.usage.to_dict(),
            "produced_memories": [asset.to_dict() for asset in produced],
            "auction": auction_summary,
            "auction_result": auction_result.to_dict(),
            "winning_agent": best_agent,
            "winning_bundle": sorted(winning_bundle),
            "bid_rationale": rationale,
            "audits": audits,
            "rsi": rsi_payload,
            "benchmark": bench.describe(),
        }
        self.save_json(payload, "llm_demo_results.json")
        return payload

    def test_llm(self) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("LLM is disabled in the config")
        return self.client.test()
