"""Agent roles in the memory market.

The prototype separates four roles:

* **producer** — solves a task and writes a reusable memory asset;
* **consumer** — bids for memory licenses based on a valuation model;
* **verifier** — audits a sample of bids against probe/ground-truth evidence;
* **auctioneer** — clears the market and publishes prices.

The LLM is used as a *proposer* (write a memory, explain a bid), not as the
source of ground truth.  Ground truth stays in the benchmark, so the experiment
can separate "the LLM wrote a plausible memory" from "the memory actually helps".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .auction import AuctionResult, Bid
from .benchmark import AgentProfile, MemoryBenchmark, Task
from .config import BenchmarkConfig
from .llm import DeepSeekClient, LLMError
from .memory import MemoryAsset
from .valuation import Valuator

LOGGER = logging.getLogger(__name__)


@dataclass
class ProducerAgent:
    profile: AgentProfile
    benchmark: MemoryBenchmark
    config: BenchmarkConfig
    client: DeepSeekClient | None = None
    seed: int = 0

    def _fallback_content(self, task: Task) -> str:
        facet = next(iter(task.required_facets))
        return (
            f"When handling {task.scope} tasks that involve {facet}, "
            f"check the relevant source before committing; record the scope and "
            f"the observed outcome."
        )

    def produce(self, task: Task, *, round_idx: int, memory_id: str) -> MemoryAsset:
        facet = next(iter(task.required_facets))
        source = self.benchmark.facet_sources.get(facet, "derived")
        quality = float(np.clip(0.45 + 0.45 * self.profile.skill, 0.2, 0.95))
        content = self._fallback_content(task)
        scope = task.scope

        if self.client is not None and self.client.available:
            prompt = (
                "You are a producer agent in a multi-agent memory market. "
                "You solved the task below and want to sell one reusable memory.\n\n"
                f"Task: {task.question}\n"
                f"Scope: {task.scope}\n"
                f"Observed evidence: {task.answer_hint}\n\n"
                "Write ONE concise, specific, scoped memory (<=80 words). "
                "It must be actionable and must not simply restate the question. "
                "Return JSON: {\"content\": \"...\", \"scope\": \"...\", "
                "\"confidence\": 0.0-1.0}."
            )
            try:
                data = self.client.json(
                    [
                        {
                            "role": "system",
                            "content": "You write compact, reusable agent memories. Return only JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=420,
                )
                content = str(data.get("content") or content)[:600]
                scope = str(data.get("scope") or scope)
                confidence = float(data.get("confidence", self.profile.skill))
                quality = float(
                    np.clip(0.35 + 0.55 * confidence * (0.6 + 0.4 * self.profile.skill), 0.15, 0.97)
                )
            except (LLMError, ValueError, TypeError) as exc:
                LOGGER.warning("Producer LLM failed for %s: %s", self.profile.id, exc)

        return MemoryAsset(
            id=memory_id,
            content=content,
            facet=facet,
            source_class=source,
            quality=quality,
            scope=scope,
            version=1,
            owner=self.profile.id,
            production_cost=float(self.config.memory_cost),
            created_round=round_idx,
            correlation_risk=float(self.config.common_mode_risk_same_source),
            origin="llm_producer" if self.client and self.client.available else "producer",
            metadata={"producer": self.profile.id, "task_id": task.id},
        )


@dataclass
class ConsumerAgent:
    profile: AgentProfile
    benchmark: MemoryBenchmark
    valuator: Valuator
    client: DeepSeekClient | None = None

    def bid(self, bundle: frozenset[str], *, split: str = "eval") -> Bid:
        value = float(self.valuator.value(self.profile.id, bundle, split=split))
        return Bid(
            agent_id=self.profile.id,
            bundle=bundle,
            value=value,
            confidence=1.0,
            evidence={"source": "valuator"},
        )

    def rationale(self, bundle: frozenset[str], *, split: str = "eval") -> str:
        """Ask the LLM for a one-sentence bid justification (optional)."""

        if self.client is None or not self.client.available or not bundle:
            return ""
        memories = []
        for mid in sorted(bundle):
            if mid in self.benchmark.memories:
                asset = self.benchmark.memories.get(mid)
                memories.append(
                    f"- {asset.id}: {asset.content} (source={asset.source_class}, "
                    f"scope={asset.scope}, quality={asset.quality:.2f})"
                )
        if not memories:
            return ""
        value = self.valuator.value(self.profile.id, bundle, split=split)
        known = {
            facet: round(coverage, 2)
            for facet, coverage in self.profile.private_knowledge.items()
            if coverage > 0.05
        }
        prompt = (
            "You are a consumer agent in a memory market. "
            "You are considering buying the following memory bundle.\n\n"
            f"Your known facets: {json.dumps(known)}\n"
            f"Your skill: {self.profile.skill:.2f}\n"
            f"Bundle:\n" + "\n".join(memories) + "\n\n"
            f"Your estimated value (0-1 scale) is {value:.3f}.\n"
            "Important: your known facets are what you already know. A bundle is "
            "valuable when it covers facets you do NOT already know and that are "
            "relevant to your task stream; it is less valuable when it duplicates "
            "knowledge you already have.\n"
            "In one sentence, explain the bundle's value in terms of missing facets "
            "and task relevance. Do not contradict the numeric estimate without a reason. "
            "Return JSON: {\"rationale\": \"...\"}."
        )
        try:
            data = self.client.json(
                [
                    {
                        "role": "system",
                        "content": "You are a terse, calibrated bidder. Return only JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=640,
            )
            return str(data.get("rationale", ""))[:400]
        except (LLMError, ValueError, TypeError) as exc:
            LOGGER.warning("Consumer rationale failed for %s: %s", self.profile.id, exc)
            return ""


@dataclass
class VerifierAgent:
    benchmark: MemoryBenchmark
    tolerance: float = 0.05
    audit_noise: float = 0.02
    seed: int = 0

    def audit_bid(
        self,
        bid: Bid,
        *,
        tasks: list[Task] | None = None,
        split: str = "eval",
    ) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + sum(ord(c) for c in bid.agent_id))
        true_value = self.benchmark.agent_value(
            bid.agent_id, bid.bundle, tasks, split=split
        )
        observed = true_value + float(rng.normal(0.0, self.audit_noise))
        inflated = bid.value - observed
        flagged = inflated > self.tolerance
        if flagged:
            for agent in self.benchmark.agents:
                if agent.id == bid.agent_id:
                    agent.credibility = max(0.1, agent.credibility * 0.9)
                    break
        return {
            "agent_id": bid.agent_id,
            "bundle": sorted(bid.bundle),
            "bid": round(float(bid.value), 6),
            "estimated_true_value": round(float(observed), 6),
            "inflated_by": round(float(inflated), 6),
            "flagged": bool(flagged),
        }


@dataclass
class Auctioneer:
    """Thin wrapper that records the market report."""

    history: list[dict[str, Any]] = field(default_factory=list)

    def publish(
        self,
        result: AuctionResult,
        *,
        round_idx: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        report = result.to_dict()
        report["round"] = round_idx
        if extra:
            report.update(extra)
        self.history.append(report)
        return report
