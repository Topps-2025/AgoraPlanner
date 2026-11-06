"""Valuation and bidding models.

Three valuation modes are supported:

``ground_truth``
    The benchmark's expected-utility difference.  Used to test the mechanism's
    theoretical properties (efficiency, individual rationality, zero price).

``additive_noisy``
    A bounded-rational bidder that estimates each memory's marginal value with
    noise and overconfidence, then adds a redundancy penalty.  Used to test how
    the auction degrades when LLM agents do not bid truthfully.

``llm``
    A DeepSeek-backed bidder that scores the usefulness of memory content for
    the agent's tasks.  The numeric score is still calibrated by the benchmark
    scale; the LLM supplies a semantic usefulness judgement.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .benchmark import AgentProfile, MemoryBenchmark, Task
from .config import BenchmarkConfig
from .llm import DeepSeekClient, LLMError
from .memory import MemoryAsset, MemoryStore

LOGGER = logging.getLogger(__name__)


class Valuator(Protocol):
    def value(
        self, agent_id: str, bundle: frozenset[str], *, split: str = "eval"
    ) -> float: ...

    def marginal(
        self,
        agent_id: str,
        memory_id: str,
        bundle: frozenset[str],
        *,
        split: str = "eval",
    ) -> float: ...


class GroundTruthValuator:
    """Exact benchmark valuation (the mechanism-design reference)."""

    def __init__(self, benchmark: MemoryBenchmark):
        self.benchmark = benchmark

    def value(
        self, agent_id: str, bundle: frozenset[str], *, split: str = "eval"
    ) -> float:
        return self.benchmark.agent_value(agent_id, bundle, split=split)

    def marginal(
        self,
        agent_id: str,
        memory_id: str,
        bundle: frozenset[str],
        *,
        split: str = "eval",
    ) -> float:
        return self.value(agent_id, bundle | {memory_id}, split=split) - self.value(
            agent_id, bundle, split=split
        )

    def bundle_values(
        self,
        agent_id: str,
        bundles: list[frozenset[str]],
        *,
        split: str = "eval",
    ) -> dict[frozenset[str], float]:
        return {bundle: self.value(agent_id, bundle, split=split) for bundle in bundles}


@dataclass
class NoisyBidder:
    """Additive noisy valuation with a same-source redundancy penalty."""

    benchmark: MemoryBenchmark
    config: BenchmarkConfig
    seed: int = 0
    _cache: dict[tuple[str, frozenset[str], str], float] = field(default_factory=dict)
    _marginals: dict[tuple[str, str, str], float] = field(default_factory=dict)
    _estimated: dict[tuple[str, str, str], float] = field(default_factory=dict)

    def _rng(self, agent_id: str, split: str) -> np.random.Generator:
        offset = sum(ord(c) for c in agent_id) + (0 if split == "train" else 7919)
        return np.random.default_rng(self.seed + offset)

    def _true_marginal(self, agent_id: str, memory_id: str, split: str) -> float:
        key = (agent_id, memory_id, split)
        if key not in self._marginals:
            self._marginals[key] = self.benchmark.agent_value(
                agent_id, {memory_id}, split=split
            )
        return self._marginals[key]

    def _estimated_marginal(
        self, agent_id: str, memory_id: str, split: str
    ) -> float:
        key = (agent_id, memory_id, split)
        if key in self._estimated:
            return self._estimated[key]
        true = max(0.0, self._true_marginal(agent_id, memory_id, split))
        rng = self._rng(agent_id, split)
        noise = float(rng.normal(0.0, self.config.bid_noise))
        overconfidence = 1.0 + self.config.bid_overconfidence
        value = max(0.0, true * overconfidence + noise)
        self._estimated[key] = value
        return value

    def value(
        self, agent_id: str, bundle: frozenset[str], *, split: str = "eval"
    ) -> float:
        key = (agent_id, frozenset(bundle), split)
        if key in self._cache:
            return self._cache[key]
        if not bundle:
            return 0.0
        total = sum(self._estimated_marginal(agent_id, mid, split) for mid in bundle)
        # Same-source, same-facet memories are substitutes: subtract a fraction
        # of the smaller estimated value for each redundant pair.
        assets = [self.benchmark.memories.get(mid) for mid in bundle]
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                left, right = assets[i], assets[j]
                if (
                    left.facet == right.facet
                    and left.source_class == right.source_class
                ):
                    total -= 0.65 * min(
                        self._estimated_marginal(agent_id, left.id, split),
                        self._estimated_marginal(agent_id, right.id, split),
                    )
        value = max(0.0, total)
        self._cache[key] = value
        return value

    def marginal(
        self,
        agent_id: str,
        memory_id: str,
        bundle: frozenset[str],
        *,
        split: str = "eval",
    ) -> float:
        return self.value(agent_id, bundle | {memory_id}, split=split) - self.value(
            agent_id, bundle, split=split
        )

    def bundle_values(
        self,
        agent_id: str,
        bundles: list[frozenset[str]],
        *,
        split: str = "eval",
    ) -> dict[frozenset[str], float]:
        return {bundle: self.value(agent_id, bundle, split=split) for bundle in bundles}


class LLMValuator:
    """A DeepSeek-backed semantic bidder.

    The LLM scores how useful each memory's content is for the agent's task
    stream on a 0–1 scale.  We calibrate the score to the benchmark's value
    scale using a single ground-truth anchor, then apply the same redundancy
    penalty as :class:`NoisyBidder`.  This keeps the demo grounded while still
    exercising LLM judgement.
    """

    def __init__(
        self,
        benchmark: MemoryBenchmark,
        config: BenchmarkConfig,
        client: DeepSeekClient,
        *,
        seed: int = 0,
    ):
        self.benchmark = benchmark
        self.config = config
        self.client = client
        self.seed = seed
        self._scores: dict[tuple[str, str], float] = {}
        self._ground = NoisyBidder(benchmark, config, seed=seed)

    def _score_memory(
        self, agent: AgentProfile, asset: MemoryAsset, tasks: list[Task]
    ) -> float:
        key = (agent.id, asset.id)
        if key in self._scores:
            return self._scores[key]
        task_summary = "\n".join(
            f"- {task.question} (scope={task.scope}, difficulty={task.difficulty:.2f})"
            for task in tasks[:8]
        )
        prompt = (
            "You are a consumer agent in a multi-agent memory market. "
            "You must decide how useful one memory asset is for your task stream.\n\n"
            f"Your skill: {agent.skill:.2f}. "
            f"Facets you already know: "
            f"{json.dumps({k: round(v, 2) for k, v in agent.private_knowledge.items() if v > 0.05})}\n\n"
            f"Task stream sample:\n{task_summary}\n\n"
            f"Memory asset:\n- content: {asset.content}\n"
            f"- scope: {asset.scope}\n- source: {asset.source_class}\n"
            f"- quality hint: {asset.quality:.2f}\n\n"
            "Return JSON: {\"usefulness\": 0.0-1.0, \"rationale\": \"one sentence\"}. "
            "Usefulness should be low if the memory is about a facet you already know, "
            "is redundant, or is too generic; high if it fills a missing facet and is scoped."
        )
        try:
            data = self.client.json(
                [
                    {
                        "role": "system",
                        "content": "You are a careful, calibrated bidder. Return only JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=320,
            )
            score = float(data.get("usefulness", 0.0))
        except (LLMError, ValueError, TypeError) as exc:
            LOGGER.warning("LLM valuation failed for %s/%s: %s", agent.id, asset.id, exc)
            score = 0.0
        score = float(np.clip(score, 0.0, 1.0))
        self._scores[key] = score
        return score

    def value(
        self, agent_id: str, bundle: frozenset[str], *, split: str = "eval"
    ) -> float:
        if not bundle:
            return 0.0
        agent = self.benchmark._agent(agent_id)  # noqa: SLF001 - benchmark helper
        tasks = self.benchmark._tasks(split)  # noqa: SLF001
        # Calibrate the 0–1 score to the benchmark value scale.  The anchor is
        # the ground-truth value of the best single memory for this agent.
        anchor = max(
            [
                max(0.0, self.benchmark.agent_value(agent_id, {mid}, split=split))
                for mid in self.benchmark.memories.ids()
            ]
            or [0.05]
        )
        total = 0.0
        assets: list[MemoryAsset] = []
        for mid in bundle:
            if mid not in self.benchmark.memories:
                continue
            asset = self.benchmark.memories.get(mid)
            score = self._score_memory(agent, asset, tasks)
            total += score * anchor
            assets.append(asset)
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                left, right = assets[i], assets[j]
                if (
                    left.facet == right.facet
                    and left.source_class == right.source_class
                ):
                    total -= 0.65 * min(
                        self._score_memory(agent, left, tasks) * anchor,
                        self._score_memory(agent, right, tasks) * anchor,
                    )
        return max(0.0, total)

    def marginal(
        self,
        agent_id: str,
        memory_id: str,
        bundle: frozenset[str],
        *,
        split: str = "eval",
    ) -> float:
        return self.value(agent_id, bundle | {memory_id}, split=split) - self.value(
            agent_id, bundle, split=split
        )

    def bundle_values(
        self,
        agent_id: str,
        bundles: list[frozenset[str]],
        *,
        split: str = "eval",
    ) -> dict[frozenset[str], float]:
        return {bundle: self.value(agent_id, bundle, split=split) for bundle in bundles}


def make_valuator(
    kind: str,
    benchmark: MemoryBenchmark,
    config: BenchmarkConfig,
    *,
    client: DeepSeekClient | None = None,
    seed: int = 0,
) -> GroundTruthValuator | NoisyBidder | LLMValuator:
    if kind == "ground_truth":
        return GroundTruthValuator(benchmark)
    if kind == "additive_noisy":
        return NoisyBidder(benchmark, config, seed=seed)
    if kind == "llm":
        if client is None or not client.available:
            LOGGER.warning("LLM valuator requested but no client; falling back to noisy bids")
            return NoisyBidder(benchmark, config, seed=seed)
        return LLMValuator(benchmark, config, client, seed=seed)
    raise ValueError(f"Unknown valuation kind: {kind}")
