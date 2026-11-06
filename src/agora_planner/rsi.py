"""Market-grounded recursive self-improvement (RSI) of memory representations.

The RSI controller proposes edits to the memory registry (retire, merge, refresh,
split, abstract).  A proposal is accepted only when it improves **held-out
ground-truth welfare** on fresh task samples and does not violate a Pareto gate.
The market's prices are the *proposal signal*; held-out welfare is the
*validation signal*.  This separation is the core defence against the
self-referential collapse that affects self-validated memory evolution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .auction import MemoryAuction
from .benchmark import MemoryBenchmark
from .config import BenchmarkConfig, RSIConfig
from .llm import DeepSeekClient, LLMError
from .memory import (
    MemoryAsset,
    MemoryStore,
    Transformation,
    apply_transformation_with_children,
    merge_assets,
    refresh_asset,
    split_asset,
)
from .metrics import summarize_allocation, summarize_auction
from .valuation import GroundTruthValuator, make_valuator

LOGGER = logging.getLogger(__name__)


@dataclass
class RSIRound:
    round_idx: int
    accepted: bool
    op: str = ""
    inputs: list[str] = field(default_factory=list)
    rationale: str = ""
    mean_gain: float = 0.0
    lcb: float = 0.0
    baseline_gt_welfare: float = 0.0
    candidate_gt_welfare: float = 0.0
    baseline_min_utility: float = 0.0
    candidate_min_utility: float = 0.0
    memory_count: int = 0
    mode: str = "market"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_idx,
            "accepted": self.accepted,
            "op": self.op,
            "inputs": list(self.inputs),
            "rationale": self.rationale,
            "mean_gain": round(self.mean_gain, 6),
            "lcb": round(self.lcb, 6),
            "baseline_gt_welfare": round(self.baseline_gt_welfare, 6),
            "candidate_gt_welfare": round(self.candidate_gt_welfare, 6),
            "baseline_min_utility": round(self.baseline_min_utility, 6),
            "candidate_min_utility": round(self.candidate_min_utility, 6),
            "memory_count": self.memory_count,
            "mode": self.mode,
            "metadata": self.metadata,
        }


class MarketGroundedRSI:
    def __init__(
        self,
        benchmark: MemoryBenchmark,
        rsi_config: RSIConfig | None = None,
        *,
        auction_config: BenchmarkConfig | None = None,
        valuator_kind: str = "ground_truth",
        client: DeepSeekClient | None = None,
        seed: int = 0,
    ):
        self.benchmark = benchmark
        self.rsi_config = rsi_config or RSIConfig()
        self.auction_config = auction_config or benchmark.config
        self.valuator_kind = valuator_kind
        self.client = client
        self.seed = seed
        self._eval_cache: dict[tuple, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _store_signature(self, store: MemoryStore) -> tuple:
        return tuple(
            (
                asset.id,
                asset.facet,
                asset.source_class,
                round(asset.quality, 6),
                round(asset.production_cost, 6),
                round(asset.correlation_risk, 6),
            )
            for asset in sorted(store, key=lambda a: a.id)
        )

    def _evaluate_store(
        self,
        store: MemoryStore,
        *,
        seed: int,
        fresh_tasks: bool = True,
        fresh_agents: bool = False,
        full_market: bool = False,
        shift: bool | None = None,
    ) -> dict[str, Any]:
        shift = self.rsi_config.validation_shift if shift is None else shift
        key = (
            self._store_signature(store),
            seed,
            fresh_tasks,
            fresh_agents,
            full_market,
            shift,
        )
        if key in self._eval_cache:
            return self._eval_cache[key]

        if fresh_agents:
            bench = self.benchmark.clone_with_fresh_agents(
                seed,
                shifted=shift,
                shift_strength=self.rsi_config.shift_strength,
            )
            bench.memories = store.copy()
            bench._success_cache = {}
            bench._value_cache = {}
            bench._bundle_cache = {}
        else:
            bench = self.benchmark.clone_with_store(
                store, seed=seed, fresh_tasks=fresh_tasks
            )
        valuator = make_valuator(
            self.valuator_kind,
            bench,
            bench.config,
            client=self.client,
            seed=seed,
        )
        auction = MemoryAuction(bench.memories, bench.agents, valuator, bench.config)
        if full_market:
            # The RSI market report is a proposal signal; beam search is fast
            # enough and keeps the (much more expensive) held-out validation
            # loop tractable.  The main experiment can still use exact VCG.
            result = auction.run(split="eval", solver="beam")
            summary = summarize_auction(bench, result)
            payload = {
                "bench": bench,
                "auction": auction,
                "result": result,
                "summary": summary,
                "gt_welfare": summary["gt_welfare"],
                "min_utility": summary["gt_min_utility"],
                "task_success": summary["task_success"],
            }
        else:
            welfare, allocation, solver = auction.solve(split="eval", solver="beam")
            summary = summarize_allocation(bench, allocation)
            payload = {
                "bench": bench,
                "auction": auction,
                "result": None,
                "allocation": allocation,
                "summary": summary,
                "gt_welfare": summary["gt_welfare"],
                "min_utility": summary["gt_min_utility"],
                "task_success": summary["task_success"],
                "solver": solver,
                "bid_welfare": welfare,
            }
        self._eval_cache[key] = payload
        return payload

    def _baseline_on_validation(self, store: MemoryStore) -> dict[str, Any]:
        seeds = self._validation_seeds()
        evals = [
            self._evaluate_store(
                store,
                seed=seed,
                fresh_tasks=True,
                fresh_agents=True,
                full_market=False,
            )
            for seed in seeds
        ]
        return {
            "gt_welfare": float(np.mean([e["gt_welfare"] for e in evals])),
            "min_utility": float(np.mean([e["min_utility"] for e in evals])),
            "task_success": float(np.mean([e["task_success"] for e in evals])),
            "per_seed": [e["gt_welfare"] for e in evals],
        }

    def _validation_seeds(self) -> list[int]:
        return [
            self.seed + 1000 + 137 * i
            for i in range(max(1, self.rsi_config.validation_seeds))
        ]

    # ------------------------------------------------------------------
    # Proposals
    # ------------------------------------------------------------------
    def _max_individual_value(self, store: MemoryStore) -> dict[str, float]:
        """A cheap redundancy signal: best single-memory value across agents."""

        bench = self.benchmark.clone_with_store(
            store, seed=self.benchmark.config.seed, fresh_tasks=False
        )
        values: dict[str, float] = {}
        for asset in store:
            best = 0.0
            for agent in bench.agents:
                best = max(
                    best,
                    bench.agent_value(agent.id, {asset.id}, split="train"),
                )
            values[asset.id] = float(best)
        return values

    def propose_heuristic(
        self,
        store: MemoryStore,
        market_report: dict[str, Any] | None = None,
    ) -> list[Transformation]:
        if len(store) <= 3:
            return []
        proposals: list[Transformation] = []
        values = self._max_individual_value(store)
        prices = (market_report or {}).get("prices", {})
        allocation = (market_report or {}).get("allocation", {})
        admitted = set((market_report or {}).get("admitted", []))

        # 1) Retire common-knowledge / redundant memories with (near) zero value.
        retire_candidates = [
            asset.id
            for asset in store
            if values.get(asset.id, 0.0) <= 0.015
            and (
                asset.origin in {"common_knowledge", "redundant"}
                or asset.id not in admitted
            )
        ]
        for mid in sorted(retire_candidates)[:2]:
            proposals.append(
                Transformation(
                    op="retire",
                    inputs=[mid],
                    rationale="zero marginal value for every agent (common knowledge or substitute)",
                )
            )

        # 2) Merge same-facet, same-source pairs when one is a valuable memory
        # and the other is a weaker duplicate.  Merging two worthless memories
        # is pointless, so rank by value gap and skip zero-value pairs.
        by_key: dict[tuple[str, str], list[MemoryAsset]] = {}
        for asset in store:
            by_key.setdefault((asset.facet, asset.source_class), []).append(asset)
        merge_candidates: list[tuple[float, MemoryAsset, MemoryAsset]] = []
        for assets in by_key.values():
            if len(assets) < 2:
                continue
            assets = sorted(assets, key=lambda a: a.quality)
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    left, right = assets[i], assets[j]
                    if left.metadata.get("poison") or right.metadata.get("poison"):
                        continue
                    v_left = max(0.0, values.get(left.id, 0.0))
                    v_right = max(0.0, values.get(right.id, 0.0))
                    if max(v_left, v_right) <= 0.02:
                        continue
                    gap = abs(v_left - v_right)
                    score = gap + 0.1 * max(v_left, v_right)
                    merge_candidates.append((score, left, right))
        merge_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, left, right in merge_candidates[:2]:
            proposals.append(
                merge_assets(
                    left,
                    right,
                    new_id=f"rsi_merge_{left.id}_{right.id}",
                    rationale="same facet/source; noisy-OR merge of a valuable memory with its weaker duplicate",
                )
            )

        # 3) Refresh an admitted, high-price, low-quality memory.
        refresh_candidates = []
        for asset in store:
            if asset.id not in admitted:
                continue
            price = float(prices.get(asset.id, 0.0))
            if price <= 1e-9:
                continue
            refresh_candidates.append((price / max(0.05, asset.quality), asset))
        refresh_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, asset in refresh_candidates[:1]:
            if asset.quality < 0.75:
                proposals.append(
                    refresh_asset(
                        asset,
                        new_id=f"rsi_refresh_{asset.id}",
                        quality_delta=0.08,
                        cost_delta=0.01,
                        rationale="high demand but low quality; refresh with a small extra verification cost",
                    )
                )

        # 4) Split a global-scope memory that is admitted and valuable.
        split_candidates = [
            asset
            for asset in store
            if asset.scope == "global" and asset.id in admitted and prices.get(asset.id, 0.0) > 0
        ]
        for asset in split_candidates[:1]:
            scopes = self.benchmark.config.task_scopes[:2]
            proposals.append(
                split_asset(
                    asset,
                    new_ids=[f"rsi_split_{asset.id}_{i}" for i in range(len(scopes))],
                    scopes=scopes,
                    rationale="global memory serves multiple scopes; split into scoped copies",
                )
            )
        return proposals[: self.rsi_config.max_proposals]

    def propose_llm(
        self,
        store: MemoryStore,
        market_report: dict[str, Any],
    ) -> list[Transformation]:
        if self.client is None or not self.client.available:
            return []
        rows = []
        for asset in sorted(store, key=lambda a: a.id):
            rows.append(
                {
                    "id": asset.id,
                    "facet": asset.facet,
                    "source": asset.source_class,
                    "quality": round(asset.quality, 3),
                    "cost": round(asset.production_cost, 3),
                    "scope": asset.scope,
                    "origin": asset.origin,
                    "price": round(float(market_report.get("prices", {}).get(asset.id, 0.0)), 4),
                    "admitted": asset.id in set(market_report.get("admitted", [])),
                }
            )
        prompt = (
            "You are the RSI controller of a multi-agent memory market. "
            "Your job is to improve the memory portfolio, not to add more memories for their own sake.\n\n"
            "You may propose up to "
            f"{self.rsi_config.max_proposals} transformations. Allowed ops:\n"
            "- retire: remove a memory that is redundant/common knowledge.\n"
            "- merge: merge two memories with the same facet and source.\n"
            "- refresh: update a high-demand but low-quality memory.\n"
            "- split: split a broad-scope memory into 2 scoped copies.\n"
            "- abstract: replace one or two memories with a more general, lower-cost memory.\n\n"
            "Market report:\n"
            + json.dumps(rows, ensure_ascii=False)
            + "\n\nReturn JSON: {\"transformations\":[{\"op\":\"...\",\"inputs\":[\"...\"],"
            "\"content\":\"optional new content\",\"quality_delta\":0.0,\"cost_delta\":0.0,"
            "\"scopes\":[\"...\"],\"rationale\":\"...\"}]}."
        )
        try:
            data = self.client.json(
                [
                    {
                        "role": "system",
                        "content": "You are a conservative memory-portfolio editor. Return only JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1800,
            )
        except (LLMError, ValueError) as exc:
            LOGGER.warning("LLM RSI proposal failed: %s", exc)
            return []

        proposals: list[Transformation] = []
        for item in data.get("transformations", [])[: self.rsi_config.max_proposals]:
            try:
                op = str(item.get("op", "")).lower()
                inputs = [str(x) for x in item.get("inputs", [])]
                if not inputs or any(mid not in store for mid in inputs):
                    continue
                rationale = str(item.get("rationale", ""))[:300]
                if op == "retire" and len(inputs) == 1:
                    proposals.append(
                        Transformation(op="retire", inputs=inputs, rationale=rationale)
                    )
                elif op == "merge" and len(inputs) == 2:
                    left, right = store.get(inputs[0]), store.get(inputs[1])
                    if left.facet != right.facet:
                        continue
                    if left.metadata.get("poison") or right.metadata.get("poison"):
                        continue
                    proposals.append(
                        merge_assets(
                            left,
                            right,
                            new_id=f"rsi_llm_merge_{left.id}_{right.id}",
                            content=item.get("content"),
                            rationale=rationale,
                        )
                    )
                elif op == "refresh" and len(inputs) == 1:
                    asset = store.get(inputs[0])
                    delta = float(item.get("quality_delta", 0.08))
                    cost_delta = float(item.get("cost_delta", 0.01))
                    delta = float(np.clip(delta, -0.1, 0.15))
                    cost_delta = float(np.clip(cost_delta, -0.03, 0.05))
                    proposals.append(
                        refresh_asset(
                            asset,
                            new_id=f"rsi_llm_refresh_{asset.id}",
                            quality_delta=delta,
                            cost_delta=cost_delta,
                            content=item.get("content"),
                            rationale=rationale,
                        )
                    )
                elif op == "split" and len(inputs) == 1:
                    asset = store.get(inputs[0])
                    scopes = [str(s) for s in item.get("scopes", [])][:3]
                    if len(scopes) < 2:
                        scopes = self.benchmark.config.task_scopes[:2]
                    proposals.append(
                        split_asset(
                            asset,
                            new_ids=[
                                f"rsi_llm_split_{asset.id}_{i}"
                                for i in range(len(scopes))
                            ],
                            scopes=scopes,
                            rationale=rationale,
                        )
                    )
                elif op == "abstract" and len(inputs) in {1, 2}:
                    inputs_assets = [store.get(mid) for mid in inputs]
                    base = inputs_assets[0]
                    quality = max(a.quality for a in inputs_assets)
                    cost = max(a.production_cost for a in inputs_assets) * 0.8
                    output = base.copy(
                        id=f"rsi_llm_abstract_{base.id}",
                        content=str(item.get("content") or base.content),
                        source_class="derived",
                        quality=float(np.clip(quality, 0.05, 0.95)),
                        production_cost=max(0.005, cost),
                        version=max(a.version for a in inputs_assets) + 1,
                        correlation_risk=min(
                            a.correlation_risk for a in inputs_assets
                        )
                        * 0.75,
                        origin="rsi_abstract",
                        metadata={
                            "abstracted_from": inputs,
                            "rationale": rationale,
                        },
                    )
                    proposals.append(
                        Transformation(
                            op="abstract",
                            inputs=inputs,
                            output=output,
                            rationale=rationale,
                        )
                    )
            except (KeyError, ValueError, TypeError):
                continue
        return proposals[: self.rsi_config.max_proposals]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def evaluate_candidate(
        self,
        candidate: Transformation,
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        current_store = self._current_store
        candidate_store = apply_transformation_with_children(current_store, candidate)
        if len(candidate_store) > self.rsi_config.max_memories:
            return {
                "accepted": False,
                "reason": "memory budget exceeded",
                "candidate_store": candidate_store,
                "mean_gain": -1e9,
            }
        seeds = self._validation_seeds()
        candidate_evals = [
            self._evaluate_store(
                candidate_store,
                seed=seed,
                fresh_tasks=True,
                fresh_agents=True,
                full_market=False,
            )
            for seed in seeds
        ]
        gains = np.asarray(
            [
                candidate_eval["gt_welfare"] - baseline_seed
                for candidate_eval, baseline_seed in zip(
                    candidate_evals, baseline["per_seed"]
                )
            ],
            dtype=float,
        )
        mean_gain = float(gains.mean())
        std_gain = float(gains.std(ddof=0))
        mdl_bonus = self.rsi_config.mdl_lambda * (
            len(current_store) - len(candidate_store)
        )
        score = mean_gain + mdl_bonus
        lcb = score - (
            1.96 * std_gain / np.sqrt(len(gains)) if len(gains) > 1 else 0.0
        )
        candidate_min_utility = float(
            np.mean([e["min_utility"] for e in candidate_evals])
        )
        candidate_gt = float(np.mean([e["gt_welfare"] for e in candidate_evals]))
        accepted = score >= self.rsi_config.delta
        if self.rsi_config.require_lower_confidence_bound:
            accepted = accepted and lcb > 0
        accepted = accepted and (
            candidate_min_utility
            >= baseline["min_utility"] - self.rsi_config.pareto_epsilon
        )
        return {
            "accepted": bool(accepted),
            "candidate_store": candidate_store,
            "mean_gain": mean_gain,
            "mdl_bonus": float(mdl_bonus),
            "score": float(score),
            "lcb": float(lcb),
            "candidate_gt_welfare": candidate_gt,
            "candidate_min_utility": candidate_min_utility,
            "baseline_gt_welfare": baseline["gt_welfare"],
            "baseline_min_utility": baseline["min_utility"],
            "per_seed": [e["gt_welfare"] for e in candidate_evals],
        }

    def step(
        self,
        store: MemoryStore,
        round_idx: int,
        *,
        use_llm: bool = False,
    ) -> tuple[MemoryStore, RSIRound]:
        self._current_store = store
        # Full market on the training task sample: prices are the proposal signal.
        train_eval = self._evaluate_store(
            store,
            seed=self.benchmark.config.seed,
            fresh_tasks=False,
            full_market=True,
        )
        market_report = (
            train_eval["result"].to_dict() if train_eval["result"] else {}
        )
        market_report["prices"] = market_report.get("prices", {})
        proposals = self.propose_heuristic(store, market_report)
        if use_llm:
            proposals.extend(self.propose_llm(store, market_report))
        if self.rsi_config.allowed_ops:
            proposals = [
                p for p in proposals if p.op in self.rsi_config.allowed_ops
            ]
        # Deduplicate by (op, inputs).
        seen: set[tuple[str, tuple[str, ...]]] = set()
        unique: list[Transformation] = []
        for proposal in proposals:
            key = (proposal.op, tuple(sorted(proposal.inputs)))
            if key in seen:
                continue
            seen.add(key)
            unique.append(proposal)

        if not unique:
            return store, RSIRound(
                round_idx=round_idx,
                accepted=False,
                rationale="no candidate transformations",
                baseline_gt_welfare=train_eval["gt_welfare"],
                memory_count=len(store),
            )

        baseline = self._baseline_on_validation(store)
        best: dict[str, Any] | None = None
        best_candidate: Transformation | None = None
        for candidate in unique:
            evaluation = self.evaluate_candidate(candidate, baseline)
            if evaluation["accepted"]:
                if best is None or evaluation["score"] > best["score"]:
                    best = evaluation
                    best_candidate = candidate

        if best is None or best_candidate is None:
            return store, RSIRound(
                round_idx=round_idx,
                accepted=False,
                rationale="no candidate passed the held-out welfare / Pareto gate",
                baseline_gt_welfare=baseline["gt_welfare"],
                baseline_min_utility=baseline["min_utility"],
                memory_count=len(store),
                metadata={"proposals": len(unique)},
            )

        return best["candidate_store"], RSIRound(
            round_idx=round_idx,
            accepted=True,
            op=best_candidate.op,
            inputs=list(best_candidate.inputs),
            rationale=best_candidate.rationale,
            mean_gain=best["mean_gain"],
            lcb=best["lcb"],
            baseline_gt_welfare=best["baseline_gt_welfare"],
            candidate_gt_welfare=best["candidate_gt_welfare"],
            baseline_min_utility=best["baseline_min_utility"],
            candidate_min_utility=best["candidate_min_utility"],
            memory_count=len(best["candidate_store"]),
            metadata={
                "proposals": len(unique),
                "mdl_bonus": best.get("mdl_bonus", 0.0),
                "score": best.get("score", best["mean_gain"]),
            },
        )

    def run(
        self,
        store: MemoryStore,
        *,
        rounds: int | None = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        rounds = self.rsi_config.rounds if rounds is None else rounds
        current = store.copy()
        logs: list[dict[str, Any]] = []
        for round_idx in range(rounds):
            current, record = self.step(current, round_idx, use_llm=use_llm)
            logs.append(record.to_dict())
        return {
            "mode": "market_grounded",
            "rounds": logs,
            "final_store": current.to_dicts(),
            "final_memory_count": len(current),
        }

    # ------------------------------------------------------------------
    # Self-validated baseline (for the collapse experiment)
    # ------------------------------------------------------------------
    def run_self_validated(
        self,
        store: MemoryStore,
        *,
        rounds: int | None = None,
    ) -> dict[str, Any]:
        """Accept a proposal if it improves the *training* welfare.

        This is the baseline that lacks an independent validation stream.  It
        can overfit the task sample used to propose the edit; we measure its
        held-out welfare after every accepted round.
        """

        rounds = self.rsi_config.rounds if rounds is None else rounds
        current = store.copy()
        logs: list[dict[str, Any]] = []
        for round_idx in range(rounds):
            self._current_store = current
            train_eval = self._evaluate_store(
                current,
                seed=self.benchmark.config.seed,
                fresh_tasks=False,
                full_market=True,
            )
            report = train_eval["result"].to_dict() if train_eval["result"] else {}
            proposals = self.propose_heuristic(current, report)
            if self.rsi_config.allowed_ops:
                proposals = [
                    p for p in proposals if p.op in self.rsi_config.allowed_ops
                ]
            baseline_train = float(train_eval["gt_welfare"])
            best: tuple[float, float, MemoryStore, Transformation] | None = None
            for proposal in proposals:
                candidate_store = apply_transformation_with_children(current, proposal)
                if len(candidate_store) > self.rsi_config.max_memories:
                    continue
                eval_train = self._evaluate_store(
                    candidate_store,
                    seed=self.benchmark.config.seed,
                    fresh_tasks=False,
                    full_market=False,
                )
                gain = float(eval_train["gt_welfare"] - baseline_train)
                mdl_bonus = self.rsi_config.mdl_lambda * (
                    len(current) - len(candidate_store)
                )
                score = gain + mdl_bonus
                if score >= self.rsi_config.delta and (
                    best is None or score > best[0]
                ):
                    best = (score, gain, candidate_store, proposal)
            if best is None:
                heldout = self._baseline_on_validation(current)
                logs.append(
                    RSIRound(
                        round_idx=round_idx,
                        accepted=False,
                        rationale="no training-welfare improvement",
                        baseline_gt_welfare=heldout["gt_welfare"],
                        memory_count=len(current),
                        mode="self_validated",
                    ).to_dict()
                )
                continue
            score, gain, candidate_store, proposal = best
            heldout_before = self._baseline_on_validation(current)
            heldout_after = self._baseline_on_validation(candidate_store)
            current = candidate_store
            logs.append(
                RSIRound(
                    round_idx=round_idx,
                    accepted=True,
                    op=proposal.op,
                    inputs=list(proposal.inputs),
                    rationale=proposal.rationale,
                    mean_gain=heldout_after["gt_welfare"]
                    - heldout_before["gt_welfare"],
                    baseline_gt_welfare=heldout_before["gt_welfare"],
                    candidate_gt_welfare=heldout_after["gt_welfare"],
                    baseline_min_utility=heldout_before["min_utility"],
                    candidate_min_utility=heldout_after["min_utility"],
                    memory_count=len(current),
                    mode="self_validated",
                    metadata={
                        "training_gain": gain,
                        "training_score": score,
                        "heldout_gain": heldout_after["gt_welfare"]
                        - heldout_before["gt_welfare"],
                    },
                ).to_dict()
            )
        return {
            "mode": "self_validated",
            "rounds": logs,
            "final_store": current.to_dicts(),
            "final_memory_count": len(current),
        }
