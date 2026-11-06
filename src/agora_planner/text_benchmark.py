"""A small natural-language memory benchmark for official memory systems.

The latent-facet benchmark in :mod:`agora_planner.benchmark` is deliberately
abstract: a memory is a vector of facet coverages, so an allocation policy can
be evaluated without a retriever.  Official memory systems (Mem0, MemGPT,
RAPTOR, GraphRAG, ...) instead consume text, embed it, and return retrieved
passages.  This module provides the missing text layer while keeping the same
allocation interface:

* every memory is a short sentence containing a unique access code;
* every query asks for one access code;
* a query is answered correctly iff the retrieved context contains the code
  from a *true* fact (a decoy fact contains a wrong code);
* the benchmark exposes ``agent_value`` / ``expected_success`` so the existing
  VCG auction and baseline policies can run unchanged.

The benchmark is intentionally small and synthetic.  It is a *mechanism and
adapter* testbed, not a replacement for LongMemEval / LoCoMo / MultiAgentBench.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .benchmark import AgentProfile, Task
from .config import BenchmarkConfig
from .memory import MemoryAsset, MemoryStore


@dataclass
class TextBenchmarkConfig(BenchmarkConfig):
    """Configuration for the text-memory benchmark.

    The base fields (seed, n_agents, max_memories_per_agent, one_per_source,
    solver, ...) keep the class compatible with ``MemoryAuction`` and the
    allocation baselines.  The text-specific fields control fact/query
    generation.
    """

    n_facets: int = 12
    facts_per_agent: int = 3
    n_public_facts: int = 4
    decoy_fraction: float = 0.35
    cross_agent_fraction: float = 0.5
    n_train_tasks: int = 24
    n_eval_tasks: int = 18
    # Duplicate true facts with the same facet are allowed; the source matroid
    # still constrains a single agent's bundle.
    source_classes: list[str] = field(
        default_factory=lambda: ["doc", "tool", "human", "derived"]
    )
    topics: list[str] = field(
        default_factory=lambda: [
            "vault",
            "archive",
            "gateway",
            "ledger",
            "beacon",
            "relay",
            "cipher",
            "harbor",
            "atlas",
            "matrix",
            "prism",
            "vector",
            "orchard",
            "lantern",
            "compass",
            "foundry",
        ]
    )
    code_alphabet: str = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    code_length: int = 4
    # ``exact`` makes a decoy lexically indistinguishable from the true fact
    # (same template, different code), so only provenance/quality or conflict
    # arbitration can resolve it.  ``hedged`` adds an unverified-note prefix.
    decoy_style: str = "exact"


@dataclass
class TextFact:
    """A natural-language fact with ground-truth provenance."""

    id: str
    facet: str
    topic: str
    content: str
    code: str
    owner: str
    source_class: str
    quality: float
    production_cost: float
    correlation_risk: float
    is_true: bool
    is_decoy: bool
    created_round: int = 0

    def to_memory_asset(self) -> MemoryAsset:
        return MemoryAsset(
            id=self.id,
            content=self.content,
            facet=self.facet,
            source_class=self.source_class,
            quality=float(self.quality),
            scope="team",
            version=1,
            owner=self.owner,
            production_cost=float(self.production_cost),
            created_round=int(self.created_round),
            correlation_risk=float(self.correlation_risk),
            origin="common_knowledge"
            if self.owner == "team" and self.is_true
            else ("producer" if self.is_true else "redundant"),
            metadata={
                "text": True,
                "is_true": bool(self.is_true),
                "is_decoy": bool(self.is_decoy),
                "answer_code": self.code,
                "topic": self.topic,
                "poison": bool(self.is_decoy and self.quality < 0.35),
            },
        )


@dataclass
class TextQuery:
    id: str
    agent_id: str
    facet: str
    topic: str
    question: str
    gold_answer: str
    required_fact_ids: frozenset[str]
    decoy_fact_ids: frozenset[str]
    difficulty: float
    scope: str
    split: str

    def to_task(self) -> Task:
        return Task(
            id=self.id,
            required_facets={self.facet: 1.0},
            difficulty=float(self.difficulty),
            scope=self.scope,
            question=self.question,
            answer_hint=self.gold_answer,
        )


_CODE_RE = re.compile(r"\b([A-Z0-9]{4})\b")
_WORD_RE = re.compile(r"[a-z0-9]+")


def extract_code(text: str) -> str | None:
    """Return the first access-code-shaped token in ``text``."""

    match = _CODE_RE.search(text or "")
    return match.group(1) if match else None


@dataclass
class TextMemoryBenchmark:
    """Text-memory benchmark compatible with the allocation/auction layer."""

    config: TextBenchmarkConfig = field(default_factory=TextBenchmarkConfig)
    rng: np.random.Generator = field(init=False)
    facets: list[str] = field(default_factory=list, init=False)
    facet_topics: dict[str, str] = field(default_factory=dict, init=False)
    facet_labels: dict[str, str] = field(default_factory=dict, init=False)
    facet_codes: dict[str, str] = field(default_factory=dict, init=False)
    agents: list[AgentProfile] = field(default_factory=list, init=False)
    memories: MemoryStore = field(default_factory=MemoryStore, init=False)
    train_tasks: list[Task] = field(default_factory=list, init=False)
    eval_tasks: list[Task] = field(default_factory=list, init=False)
    queries: dict[str, TextQuery] = field(default_factory=dict, init=False)
    task_agent: dict[str, str] = field(default_factory=dict, init=False)
    _value_cache: dict[tuple[str, frozenset[str], str], float] = field(
        default_factory=dict, init=False
    )
    _success_cache: dict[tuple[str, frozenset[str], str], float] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.seed)
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _new_code(self, used: set[str]) -> str:
        while True:
            code = "".join(
                self.rng.choice(list(self.config.code_alphabet), size=self.config.code_length)
            )
            if code not in used:
                used.add(code)
                return code

    def _quality(self, low: float, high: float) -> float:
        return float(self.rng.uniform(low, high))

    def _source(self) -> str:
        return str(self.rng.choice(self.config.source_classes))

    @staticmethod
    def _corr(source: str) -> float:
        return {
            "doc": 0.30,
            "tool": 0.45,
            "human": 0.65,
            "derived": 0.80,
        }.get(source, 0.55)

    def _build(self) -> None:
        cfg = self.config
        used_codes: set[str] = set()
        greek = list(string.ascii_lowercase)
        for idx in range(cfg.n_facets):
            facet = f"facet_{idx:02d}"
            topic = cfg.topics[idx % len(cfg.topics)]
            label = f"facet {greek[idx % len(greek)]}{idx // len(greek) or ''}"
            self.facets.append(facet)
            self.facet_topics[facet] = topic
            self.facet_labels[facet] = label
            self.facet_codes[facet] = self._new_code(used_codes)

        for idx in range(cfg.n_agents):
            self.agents.append(
                AgentProfile(
                    id=f"agent{idx}",
                    skill=float(np.clip(self.rng.normal(0.55, 0.12), 0.2, 0.9)),
                    private_knowledge={},
                    max_memories=int(cfg.max_memories_per_agent),
                )
            )

        # True facts: each agent owns ``facts_per_agent`` facets; every facet
        # gets at least one true fact (possibly a team-owned public copy).
        owner_facets: dict[str, list[str]] = {agent.id: [] for agent in self.agents}
        shuffled = list(self.facets)
        self.rng.shuffle(shuffled)
        cursor = 0
        for agent in self.agents:
            for _ in range(cfg.facts_per_agent):
                facet = shuffled[cursor % len(shuffled)]
                cursor += 1
                owner_facets[agent.id].append(facet)
                self._add_true_fact(facet, agent.id, used_codes)

        public_facets = list(self.rng.choice(self.facets, size=min(cfg.n_public_facts, len(self.facets)), replace=False))
        for facet in public_facets:
            self._add_true_fact(str(facet), "team", used_codes)

        covered = {
            asset.facet
            for asset in self.memories
            if asset.metadata.get("is_true")
        }
        for facet in self.facets:
            if facet not in covered:
                self._add_true_fact(facet, "team", used_codes)

        # Decoys: a wrong code with the same facet/topic.  They are plausible
        # lexically but low-quality; a quality-aware policy can filter them.
        n_decoys = int(round(cfg.decoy_fraction * len(self.facets)))
        decoy_facets = list(self.rng.choice(self.facets, size=max(0, n_decoys), replace=False))
        for facet in decoy_facets:
            owners = [agent.id for agent in self.agents] + ["team"]
            owner = str(self.rng.choice(owners))
            wrong = self._new_code(used_codes)
            topic = self.facet_topics[facet]
            label = self.facet_labels[facet]
            source = self._source()
            quality = self._quality(0.10, 0.38)
            fid = f"decoy_{facet}_{owner}_{len(self.memories)}"
            if cfg.decoy_style == "hedged":
                content = (
                    f"An unverified cached note claims the {topic} access code "
                    f"for {label} is {wrong}."
                )
            else:
                content = f"The {topic} access code for {label} is {wrong}."
            self.memories.add(
                TextFact(
                    id=fid,
                    facet=facet,
                    topic=topic,
                    content=content,
                    code=wrong,
                    owner=owner,
                    source_class=source,
                    quality=quality,
                    production_cost=float(self.rng.uniform(0.1, 0.4)),
                    correlation_risk=self._corr(source),
                    is_true=False,
                    is_decoy=True,
                    created_round=int(self.rng.integers(0, 6)),
                ).to_memory_asset()
            )

        self.train_tasks = self._make_queries(cfg.n_train_tasks, split="train", seed_offset=101)
        self.eval_tasks = self._make_queries(cfg.n_eval_tasks, split="eval", seed_offset=202)
        for query in self.queries.values():
            self.task_agent[query.id] = query.agent_id

    def _add_true_fact(self, facet: str, owner: str, used_codes: set[str]) -> None:
        topic = self.facet_topics[facet]
        label = self.facet_labels[facet]
        code = self.facet_codes[facet]
        source = self._source()
        quality = self._quality(0.55, 0.96)
        fid = f"true_{facet}_{owner}_{len(self.memories)}"
        self.memories.add(
            TextFact(
                id=fid,
                facet=facet,
                topic=topic,
                content=f"The {topic} access code for {label} is {code}.",
                code=code,
                owner=owner,
                source_class=source,
                quality=quality,
                production_cost=float(self.rng.uniform(0.3, 1.0)),
                correlation_risk=self._corr(source),
                is_true=True,
                is_decoy=False,
                created_round=int(self.rng.integers(0, 6)),
            ).to_memory_asset()
        )

    def _make_queries(self, n: int, *, split: str, seed_offset: int) -> list[Task]:
        cfg = self.config
        if not self.agents:
            return []
        rng = np.random.default_rng(cfg.seed + seed_offset)
        tasks: list[Task] = []
        for q_idx in range(n):
            agent = self.agents[q_idx % len(self.agents)]
            own = [
                facet
                for facet in self.facets
                if any(
                    asset.facet == facet
                    and asset.owner == agent.id
                    and asset.metadata.get("is_true")
                    for asset in self.memories
                )
            ]
            cross = [facet for facet in self.facets if facet not in own]
            use_cross = rng.random() < cfg.cross_agent_fraction and bool(cross)
            pool = cross if use_cross else (own or cross or self.facets)
            facet = str(rng.choice(pool))
            # Half of cross-agent queries deliberately target a facet with a
            # decoy, so sharing without arbitration can be actively harmful.
            decoy_facets = [
                f
                for f in pool
                if any(
                    asset.facet == f and asset.metadata.get("is_decoy")
                    for asset in self.memories
                )
            ]
            if decoy_facets and rng.random() < 0.5:
                facet = str(rng.choice(decoy_facets))
            topic = self.facet_topics[facet]
            label = self.facet_labels[facet]
            true_ids = frozenset(
                asset.id
                for asset in self.memories
                if asset.facet == facet and asset.metadata.get("is_true")
            )
            decoy_ids = frozenset(
                asset.id
                for asset in self.memories
                if asset.facet == facet and asset.metadata.get("is_decoy")
            )
            qid = f"{split}_{agent.id}_{q_idx:03d}"
            query = TextQuery(
                id=qid,
                agent_id=agent.id,
                facet=facet,
                topic=topic,
                question=f"What is the {topic} access code for {label}?",
                gold_answer=self.facet_codes[facet],
                required_fact_ids=true_ids,
                decoy_fact_ids=decoy_ids,
                difficulty=float(rng.uniform(0.2, 0.8)),
                scope="team",
                split=split,
            )
            self.queries[qid] = query
            tasks.append(query.to_task())
        return tasks

    # ------------------------------------------------------------------
    # Benchmark interface used by the auction / baselines
    # ------------------------------------------------------------------
    def _agent(self, agent_id: str) -> AgentProfile:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        raise KeyError(f"Unknown agent: {agent_id}")

    def _tasks(self, split: str) -> list[Task]:
        return self.train_tasks if split == "train" else self.eval_tasks

    def query_success(
        self, agent_id: str, memory_ids: Iterable[str], query: TextQuery
    ) -> float:
        """Probability of returning the gold code from this bundle.

        We deliberately use a simple analytic model rather than the retriever:
        the auction needs a stable ground-truth value, while the empirical
        retrieval metrics are computed separately by each system adapter.
        """

        bundle = frozenset(memory_ids)
        true_qualities: list[float] = []
        decoy_qualities: list[float] = []
        for mid in bundle:
            if mid not in self.memories:
                continue
            asset = self.memories.get(mid)
            if asset.facet != query.facet:
                continue
            if asset.metadata.get("is_true"):
                true_qualities.append(float(asset.quality))
            elif asset.metadata.get("is_decoy"):
                decoy_qualities.append(float(asset.quality))
        if not true_qualities:
            return 0.0
        best_true = max(true_qualities)
        if not decoy_qualities:
            return best_true
        # The retriever ranks true and decoy facts by similarity; the true fact
        # wins with a quality-weighted probability.  This makes a decoy harmful
        # but not automatically fatal.
        conflict = best_true / (best_true + sum(decoy_qualities) + 1e-9)
        return float(best_true * conflict)

    def success_probability(
        self, agent_id: str, memory_ids: Iterable[str], task: Task
    ) -> float:
        query = self.queries.get(task.id)
        if query is None:
            return 0.0
        return self.query_success(agent_id, memory_ids, query)

    def expected_success(
        self, agent_id: str, memory_ids: Iterable[str], tasks: list[Task] | None = None
    ) -> float:
        tasks = self._tasks("eval") if tasks is None else tasks
        own_tasks = [task for task in tasks if self.task_agent.get(task.id) == agent_id]
        if not own_tasks:
            return 0.0
        bundle = frozenset(memory_ids)
        key = (agent_id, bundle, "|".join(task.id for task in own_tasks))
        if key in self._success_cache:
            return self._success_cache[key]
        value = float(
            np.mean([self.success_probability(agent_id, bundle, task) for task in own_tasks])
        )
        self._success_cache[key] = value
        return value

    def agent_value(
        self,
        agent_id: str,
        memory_ids: Iterable[str],
        tasks: list[Task] | None = None,
        *,
        split: str = "eval",
    ) -> float:
        tasks = self._tasks(split) if tasks is None else tasks
        bundle = frozenset(memory_ids)
        own_tasks = tuple(task.id for task in tasks if self.task_agent.get(task.id) == agent_id)
        key = (agent_id, bundle, "|".join(own_tasks) + f"::{split}")
        if key in self._value_cache:
            return self._value_cache[key]
        with_bundle = self.expected_success(agent_id, bundle, tasks)
        without_bundle = 0.0
        value = (
            float(self.config.utility_scale) * (with_bundle - without_bundle)
            - float(self.config.usage_cost) * len(bundle)
        )
        self._value_cache[key] = value
        return value

    def memory_marginal_value(
        self,
        agent_id: str,
        memory_id: str,
        bundle: Iterable[str],
        tasks: list[Task] | None = None,
        *,
        split: str = "eval",
    ) -> float:
        bundle = frozenset(bundle)
        return self.agent_value(agent_id, bundle | {memory_id}, tasks, split=split) - self.agent_value(
            agent_id, bundle, tasks, split=split
        )

    def memory_total_marginal_value(
        self,
        memory_id: str,
        allocation: dict[str, frozenset[str]],
        tasks: list[Task] | None = None,
        *,
        split: str = "eval",
    ) -> float:
        total = 0.0
        for agent_id, bundle in allocation.items():
            if memory_id in bundle:
                total += self.memory_marginal_value(
                    agent_id, memory_id, bundle - {memory_id}, tasks, split=split
                )
        return total

    def error_correlation(
        self, allocation: dict[str, frozenset[str]], tasks: list[Task] | None = None, *, seed: int = 7
    ) -> float:
        """Mean pairwise correlation of per-facet success probabilities.

        Agents answer different query sets, so we compare their success
        probability on the shared facet grid.  This is a diagnostic, not a
        statistical estimate.
        """

        tasks = self._tasks("eval") if tasks is None else tasks
        agent_ids = [agent.id for agent in self.agents]
        vectors: dict[str, list[float]] = {}
        for agent_id in agent_ids:
            values = []
            for facet in self.facets:
                facet_tasks = [
                    task
                    for task in tasks
                    if self.task_agent.get(task.id) == agent_id
                    and task.required_facets.get(facet)
                ]
                if facet_tasks:
                    values.append(
                        float(
                            np.mean(
                                [
                                    self.success_probability(
                                        agent_id, allocation.get(agent_id, frozenset()), task
                                    )
                                    for task in facet_tasks
                                ]
                            )
                        )
                    )
                else:
                    values.append(0.0)
            vectors[agent_id] = values
        correlations: list[float] = []
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                left = np.asarray(vectors[agent_ids[i]], dtype=float)
                right = np.asarray(vectors[agent_ids[j]], dtype=float)
                if left.std() < 1e-9 or right.std() < 1e-9:
                    continue
                correlations.append(float(np.corrcoef(left, right)[0, 1]))
        return float(np.mean(correlations)) if correlations else 0.0

    def feasible_bundles(
        self, agent_id: str, memory_ids: Iterable[str] | None = None
    ) -> list[frozenset[str]]:
        agent = self._agent(agent_id)
        ids = sorted(self.memories.ids() if memory_ids is None else memory_ids)
        budget = max(0, int(agent.max_memories))
        from itertools import combinations

        bundles: list[frozenset[str]] = []
        for size in range(budget + 1):
            for combo in combinations(ids, size):
                sources = [self.memories.get(mid).source_class for mid in combo]
                if self.config.one_per_source and len(set(sources)) != len(sources):
                    continue
                bundles.append(frozenset(combo))
        return bundles

    # ------------------------------------------------------------------
    # Retrieval evaluation
    # ------------------------------------------------------------------
    def texts_for(self, memory_ids: Iterable[str]) -> list[str]:
        return [
            self.memories.get(mid).content
            for mid in memory_ids
            if mid in self.memories
        ]

    def evaluate_retrieval(
        self,
        retrieved: dict[str, list[str]],
        *,
        k: int = 3,
        split: str = "eval",
    ) -> dict:
        """Evaluate a mapping ``query_id -> ranked retrieved texts``.

        ``hit@1`` is exact answer accuracy: the first retrieved passage must
        contain the gold code.  ``recall@k`` is whether any of the top-k
        passages contains it.  ``decoy@1`` flags a wrong code in the top
        passage.  Token counts are whitespace tokens in the retrieved context.
        """

        tasks = self._tasks(split)
        rows: list[dict] = []
        for task in tasks:
            query = self.queries.get(task.id)
            if query is None:
                continue
            texts = [str(text) for text in retrieved.get(query.id, [])][:k]
            gold = query.gold_answer
            first = texts[0] if texts else ""
            parsed = extract_code(first)
            hit1 = bool(parsed == gold)
            hitk = any(gold in text for text in texts)
            rank = 0
            for idx, text in enumerate(texts):
                if gold in text:
                    rank = idx + 1
                    break
            decoy = bool(parsed and parsed != gold)
            rows.append(
                {
                    "query_id": query.id,
                    "agent_id": query.agent_id,
                    "facet": query.facet,
                    "cross_agent": not any(
                        self.memories.get(mid).owner == query.agent_id
                        for mid in query.required_fact_ids
                        if mid in self.memories
                    ),
                    "hit1": hit1,
                    "hitk": hitk,
                    "mrr": 1.0 / rank if rank else 0.0,
                    "decoy1": decoy,
                    "tokens": sum(len(text.split()) for text in texts),
                    "context_chars": sum(len(text) for text in texts),
                }
            )
        if not rows:
            return {
                "n": 0,
                "answer_accuracy@1": 0.0,
                "recall@k": 0.0,
                "mrr": 0.0,
                "decoy_rate@1": 0.0,
                "context_tokens": 0.0,
            }

        def _mean(key: str, subset: list[dict] | None = None) -> float:
            subset = rows if subset is None else subset
            return float(np.mean([row[key] for row in subset])) if subset else 0.0

        cross_rows = [row for row in rows if row["cross_agent"]]
        own_rows = [row for row in rows if not row["cross_agent"]]
        return {
            "n": len(rows),
            "answer_accuracy@1": _mean("hit1"),
            "recall@k": _mean("hitk"),
            "mrr": _mean("mrr"),
            "decoy_rate@1": _mean("decoy1"),
            "context_tokens": _mean("tokens"),
            "context_chars": _mean("context_chars"),
            "own_accuracy@1": _mean("hit1", own_rows),
            "cross_accuracy@1": _mean("hit1", cross_rows),
            "own_n": len(own_rows),
            "cross_n": len(cross_rows),
        }

    def describe(self) -> dict:
        return {
            "n_agents": len(self.agents),
            "n_facets": len(self.facets),
            "n_memories": len(self.memories),
            "n_train_tasks": len(self.train_tasks),
            "n_eval_tasks": len(self.eval_tasks),
            "n_true": sum(1 for asset in self.memories if asset.metadata.get("is_true")),
            "n_decoy": sum(1 for asset in self.memories if asset.metadata.get("is_decoy")),
            "agents": [agent.id for agent in self.agents],
            "memories": self.memories.summary(),
        }


__all__ = [
    "TextBenchmarkConfig",
    "TextFact",
    "TextQuery",
    "TextMemoryBenchmark",
    "extract_code",
]
