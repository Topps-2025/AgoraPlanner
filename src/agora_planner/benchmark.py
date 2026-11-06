"""A controllable latent-facet benchmark for memory markets.

The benchmark has ground-truth structure:

* ``facets`` — the latent pieces of knowledge a task may require;
* ``agents`` — private knowledge and skill;
* ``memories`` — assets covering facets, with source classes and qualities;
* ``tasks`` — weighted subsets of facets plus a difficulty.

A memory's ground-truth value for an agent is the expected task-success
improvement it produces, minus a small usage cost.  This lets us evaluate
whether a market allocates memory to the agents who actually benefit, whether
redundant memories receive zero price, and whether market-grounded RSI improves
held-out welfare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Iterable

import numpy as np

from .config import BenchmarkConfig
from .memory import MemoryAsset, MemoryStore


@dataclass
class AgentProfile:
    id: str
    skill: float
    private_knowledge: dict[str, float] = field(default_factory=dict)
    max_memories: int = 2
    wallet: float = 0.0
    credibility: float = 1.0
    produced: list[str] = field(default_factory=list)

    def copy(self) -> "AgentProfile":
        return replace(
            self,
            private_knowledge=dict(self.private_knowledge),
            produced=list(self.produced),
        )


@dataclass
class Task:
    id: str
    required_facets: dict[str, float]
    difficulty: float
    scope: str
    question: str = ""
    answer_hint: str = ""


def _sigmoid(x: float) -> float:
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class MemoryBenchmark:
    """Latent-facet benchmark with deterministic generation and caches."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.facets: list[str] = []
        self.facet_sources: dict[str, str] = {}
        self.facet_descriptions: dict[str, str] = {}
        self.agents: list[AgentProfile] = []
        self.memories = MemoryStore()
        self.train_tasks: list[Task] = []
        self.eval_tasks: list[Task] = []
        self._success_cache: dict[tuple[str, frozenset[str]], float] = {}
        self._value_cache: dict[tuple[str, frozenset[str], str], float] = {}
        self._bundle_cache: dict[tuple[str, frozenset[str]], list[frozenset[str]]] = {}
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self._build_facets()
        self._build_agents()
        self._build_memories()
        self.train_tasks = self._generate_tasks(
            self.config.n_train_tasks, prefix="train", seed=self.config.seed * 1000 + 1
        )
        self.eval_tasks = self._generate_tasks(
            self.config.n_eval_tasks, prefix="eval", seed=self.config.seed * 1000 + 2
        )

    def _build_facets(self) -> None:
        sources = self.config.source_classes
        for idx in range(self.config.n_facets):
            facet = f"f{idx:02d}"
            source = sources[idx % len(sources)]
            self.facets.append(facet)
            self.facet_sources[facet] = source
            self.facet_descriptions[facet] = (
                f"knowledge facet {facet} (typical source: {source})"
            )

    def _build_agents(self) -> None:
        n_agents = self.config.n_agents
        skills = np.clip(self.rng.normal(0.55, 0.12, size=n_agents), 0.15, 0.9)
        common = {
            str(f)
            for f in self.rng.choice(
                self.facets,
                size=max(1, int(round(self.config.common_knowledge_fraction * len(self.facets)))),
                replace=False,
            ).tolist()
        }
        cold_candidates = [f for f in self.facets if f not in common]
        cold = (
            {
                str(f)
                for f in self.rng.choice(
                    cold_candidates,
                    size=min(
                        len(cold_candidates),
                        max(1, int(round(self.config.cold_start_fraction * len(self.facets)))),
                    ),
                    replace=False,
                ).tolist()
            }
            if cold_candidates
            else set()
        )

        non_common = [f for f in self.facets if f not in common and f not in cold]
        for agent_idx in range(n_agents):
            private: dict[str, float] = {}
            # Each agent has a distinct specialty: a subset of facets it knows
            # well.  This makes the same memory valuable to one agent and
            # redundant to another, which is exactly where per-agent allocation
            # beats a global top-k pool.
            n_specialty = max(1, int(round(0.35 * len(non_common)))) if non_common else 0
            specialty = set(
                str(f)
                for f in self.rng.choice(
                    non_common, size=min(n_specialty, len(non_common)), replace=False
                ).tolist()
            ) if non_common else set()
            for facet in self.facets:
                if facet in common:
                    private[facet] = float(self.rng.uniform(0.72, 0.95))
                elif facet in cold:
                    private[facet] = 0.0
                elif facet in specialty:
                    private[facet] = float(self.rng.uniform(0.65, 0.95))
                elif self.rng.random() < 0.20:
                    private[facet] = float(self.rng.uniform(0.0, 0.30))
                else:
                    private[facet] = 0.0
            self.agents.append(
                AgentProfile(
                    id=f"agent{agent_idx}",
                    skill=float(skills[agent_idx]),
                    private_knowledge=private,
                    max_memories=self.config.max_memories_per_agent,
                )
            )

    def _new_memory(
        self,
        *,
        idx: int,
        facet: str,
        source: str | None,
        quality: float,
        scope: str | None = None,
        owner: str | None = None,
        origin: str = "producer",
        cost_scale: float = 1.0,
        correlation_risk: float | None = None,
        metadata: dict | None = None,
    ) -> MemoryAsset:
        owner = owner or self.agents[int(self.rng.integers(0, len(self.agents)))].id
        source = source or self.facet_sources[facet]
        scope = scope or self.config.task_scopes[
            int(self.rng.integers(0, len(self.config.task_scopes)))
        ]
        if correlation_risk is None:
            correlation_risk = (
                self.config.common_mode_risk_same_source
                if source == self.facet_sources[facet]
                else self.config.common_mode_risk_cross_source
            )
        return MemoryAsset(
            id=f"m{idx:03d}",
            content=f"Reusable lesson about {self.facet_descriptions[facet]}.",
            facet=facet,
            source_class=source,
            quality=float(np.clip(quality, 0.02, 0.99)),
            scope=scope,
            version=1,
            owner=owner,
            production_cost=float(
                max(0.005, self.config.memory_cost * cost_scale)
            ),
            created_round=0,
            correlation_risk=float(np.clip(correlation_risk, 0.0, 1.0)),
            origin=origin,
            metadata=metadata or {},
        )

    def _build_memories(self) -> None:
        # ``n_memories`` counts *base* memories.  Redundant / poisoned / common
        # knowledge variants are added on top, so the base pool can cover every
        # facet even when the benchmark is small.
        n_base = max(1, self.config.n_memories)
        n_redundant = max(0, int(round(self.config.redundancy_fraction * n_base)))
        n_poison = max(0, int(round(self.config.poison_fraction * n_base)))
        n_common = max(2, int(round(0.15 * n_base)))

        facet_order = list(self.facets)
        self.rng.shuffle(facet_order)
        base_assets: list[MemoryAsset] = []
        for idx in range(n_base):
            facet = facet_order[idx % len(facet_order)]
            # 70% of memories come from the facet's canonical source; 30% are
            # alternative-source discoveries, which are more independent.
            source = (
                self.facet_sources[facet]
                if self.rng.random() < 0.70
                else str(self.rng.choice(self.config.source_classes))
            )
            asset = self._new_memory(
                idx=idx,
                facet=facet,
                source=source,
                quality=float(self.rng.uniform(0.55, 0.94)),
                origin="producer",
                cost_scale=float(self.rng.uniform(0.8, 1.25)),
            )
            base_assets.append(asset)
            self.memories.add(asset)

        # Common-knowledge memories: every agent already knows the facet, so the
        # marginal value should be (near) zero.  These are the zero-price test.
        common_facets = [
            facet
            for facet in self.facets
            if all(agent.private_knowledge.get(facet, 0.0) >= 0.7 for agent in self.agents)
        ]
        if not common_facets:
            common_facets = list(self.facets[:n_common])
        for j in range(n_common):
            facet = common_facets[j % len(common_facets)]
            asset = self._new_memory(
                idx=n_base + j,
                facet=facet,
                source=self.facet_sources[facet],
                quality=float(self.rng.uniform(0.85, 0.97)),
                origin="common_knowledge",
                cost_scale=0.5,
                metadata={"common_knowledge": True},
            )
            self.memories.add(asset)

        # Redundant memories: same facet and source as an existing asset, lower
        # quality.  They are near-perfect substitutes and should be priced at 0.
        for j in range(n_redundant):
            parent = base_assets[int(self.rng.integers(0, len(base_assets)))]
            asset = self._new_memory(
                idx=n_base + n_common + j,
                facet=parent.facet,
                source=parent.source_class,
                quality=parent.quality * float(self.rng.uniform(0.55, 0.85)),
                scope=parent.scope,
                origin="redundant",
                cost_scale=0.6,
                metadata={"redundant_of": parent.id},
            )
            self.memories.add(asset)

        # Poisoned memories: look plausible but hurt the agent that uses them.
        for j in range(n_poison):
            parent = base_assets[int(self.rng.integers(0, len(base_assets)))]
            asset = self._new_memory(
                idx=n_base + n_common + n_redundant + j,
                facet=parent.facet,
                source=str(self.rng.choice(self.config.source_classes)),
                quality=float(self.rng.uniform(0.70, 0.95)),
                scope=parent.scope,
                origin="poisoned",
                cost_scale=0.9,
                metadata={"poison": True},
            )
            self.memories.add(asset)

    def _generate_tasks(
        self, n_tasks: int, *, prefix: str, seed: int
    ) -> list[Task]:
        rng = np.random.default_rng(seed)
        tasks: list[Task] = []
        for idx in range(n_tasks):
            k = int(rng.integers(self.config.task_facet_min, self.config.task_facet_max + 1))
            k = min(k, len(self.facets))
            facet_p = None
            if self.config.task_facet_weights:
                raw_p = np.asarray(
                    [
                        max(1e-3, float(self.config.task_facet_weights.get(f, 1.0)))
                        for f in self.facets
                    ],
                    dtype=float,
                )
                if raw_p.sum() > 0:
                    facet_p = raw_p / raw_p.sum()
            chosen = [
                str(f)
                for f in rng.choice(
                    self.facets, size=k, replace=False, p=facet_p
                ).tolist()
            ]
            raw = rng.uniform(0.5, 1.5, size=k)
            weights = raw / raw.sum()
            required = {facet: float(weight) for facet, weight in zip(chosen, weights)}
            difficulty = float(rng.uniform(0.15, 0.85))
            scope = str(rng.choice(self.config.task_scopes))
            question = (
                f"[{scope}] Which facts are required to handle a task involving "
                + ", ".join(chosen)
                + "?"
            )
            answer_hint = "; ".join(
                f"{facet}:{self.facet_descriptions[facet]}" for facet in chosen
            )
            tasks.append(
                Task(
                    id=f"{prefix}_{idx:04d}",
                    required_facets=required,
                    difficulty=difficulty,
                    scope=scope,
                    question=question,
                    answer_hint=answer_hint,
                )
            )
        return tasks

    # ------------------------------------------------------------------
    # Ground-truth model
    # ------------------------------------------------------------------
    def _facet_coverage(
        self, agent_id: str, memory_ids: Iterable[str], facet: str
    ) -> float:
        agent = self._agent(agent_id)
        base = float(agent.private_knowledge.get(facet, 0.0))
        by_source: dict[str, float] = {}
        for memory_id in memory_ids:
            if memory_id not in self.memories:
                continue
            asset = self.memories.get(memory_id)
            if asset.facet != facet or asset.metadata.get("poison"):
                continue
            by_source[asset.source_class] = max(
                by_source.get(asset.source_class, 0.0), max(0.0, asset.quality)
            )
        if not by_source:
            return min(1.0, base)
        # Diminishing returns to additional independent source classes.
        ranked = sorted(by_source.values(), reverse=True)
        extra = sum(q * (0.82**rank) for rank, q in enumerate(ranked))
        return min(1.0, base + extra)

    def _poison_count(self, memory_ids: Iterable[str]) -> int:
        return sum(
            1
            for memory_id in memory_ids
            if memory_id in self.memories
            and self.memories.get(memory_id).metadata.get("poison")
        )

    def success_probability(
        self, agent_id: str, memory_ids: Iterable[str], task: Task
    ) -> float:
        agent = self._agent(agent_id)
        memory_ids = frozenset(memory_ids)
        facet_coverages = {
            facet: self._facet_coverage(agent_id, memory_ids, facet)
            for facet in task.required_facets
        }
        weighted = sum(
            weight * facet_coverages[facet]
            for facet, weight in task.required_facets.items()
        )
        # A soft bottleneck: a task with one completely missing facet is much
        # harder than the weighted average suggests.  This creates genuine
        # complementarity between memories covering different facets.
        bottleneck = min(facet_coverages.values()) if facet_coverages else 0.0
        coverage = weighted * (0.55 + 0.45 * bottleneck)
        poison = self._poison_count(memory_ids)
        z = (
            self.config.success_alpha * coverage
            - self.config.success_beta * task.difficulty * len(task.required_facets)
            + self.config.success_gamma * (agent.skill - 0.5)
            - self.config.poison_penalty * poison
        )
        z = z / max(0.1, self.config.success_temperature)
        return float(np.clip(_sigmoid(z), 0.01, 0.99))

    def expected_success(
        self, agent_id: str, memory_ids: Iterable[str], tasks: list[Task] | None = None
    ) -> float:
        tasks = self.eval_tasks if tasks is None else tasks
        key = (agent_id, frozenset(memory_ids))
        if key in self._success_cache:
            return self._success_cache[key]
        value = float(
            np.mean(
                [
                    self.success_probability(agent_id, memory_ids, task)
                    for task in tasks
                ]
            )
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
        """Expected success improvement minus the agent's usage cost."""

        tasks = self._tasks(split) if tasks is None else tasks
        bundle = frozenset(memory_ids)
        key = (agent_id, bundle, split if tasks is self._tasks(split) else "custom")
        if key in self._value_cache:
            return self._value_cache[key]
        with_bundle = self.expected_success(agent_id, bundle, tasks)
        without_bundle = self.expected_success(agent_id, frozenset(), tasks)
        value = (
            self.config.utility_scale * (with_bundle - without_bundle)
            - self.config.usage_cost * len(bundle)
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
        return self.agent_value(
            agent_id, bundle | {memory_id}, tasks, split=split
        ) - self.agent_value(agent_id, bundle, tasks, split=split)

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

    # ------------------------------------------------------------------
    # Feasible bundles
    # ------------------------------------------------------------------
    def feasible_bundles(
        self, agent_id: str, memory_ids: Iterable[str] | None = None
    ) -> list[frozenset[str]]:
        agent = self._agent(agent_id)
        memory_ids = tuple(sorted(self.memories.ids() if memory_ids is None else memory_ids))
        cache_key = (agent_id, frozenset(memory_ids))
        if cache_key in self._bundle_cache:
            return self._bundle_cache[cache_key]

        bundles: list[frozenset[str]] = [frozenset()]
        max_size = max(0, int(agent.max_memories))
        for size in range(1, max_size + 1):
            for combo in combinations(memory_ids, size):
                if self.config.one_per_source:
                    sources = [self.memories.get(mid).source_class for mid in combo]
                    if len(set(sources)) != len(sources):
                        continue
                bundles.append(frozenset(combo))
        bundles = sorted(bundles, key=lambda b: (len(b), tuple(sorted(b))))
        self._bundle_cache[cache_key] = bundles
        return bundles

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def sampled_success_matrix(
        self,
        allocation: dict[str, frozenset[str]],
        tasks: list[Task] | None = None,
        *,
        seed: int = 7,
    ) -> tuple[list[str], np.ndarray]:
        tasks = self.eval_tasks if tasks is None else tasks
        rng = np.random.default_rng(seed)
        agent_ids = sorted(allocation)
        matrix = np.zeros((len(agent_ids), len(tasks)), dtype=float)
        for i, agent_id in enumerate(agent_ids):
            for j, task in enumerate(tasks):
                p = self.success_probability(agent_id, allocation[agent_id], task)
                matrix[i, j] = 1.0 if rng.random() < p else 0.0
        return agent_ids, matrix

    def error_correlation(
        self,
        allocation: dict[str, frozenset[str]],
        tasks: list[Task] | None = None,
        *,
        seed: int = 7,
    ) -> float:
        agent_ids, matrix = self.sampled_success_matrix(allocation, tasks, seed=seed)
        if len(agent_ids) < 2:
            return 0.0
        correlations = []
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                x, y = matrix[i], matrix[j]
                if x.std() == 0 or y.std() == 0:
                    correlations.append(0.0)
                else:
                    correlations.append(float(np.corrcoef(x, y)[0, 1]))
        return float(np.mean(correlations)) if correlations else 0.0

    # ------------------------------------------------------------------
    # Clones for RSI / held-out validation
    # ------------------------------------------------------------------
    def clone_with_tasks(
        self, seed: int, *, n_train_tasks: int | None = None, n_eval_tasks: int | None = None
    ) -> "MemoryBenchmark":
        clone = object.__new__(MemoryBenchmark)
        clone.config = replace(
            self.config,
            seed=seed,
            n_train_tasks=self.config.n_train_tasks if n_train_tasks is None else n_train_tasks,
            n_eval_tasks=self.config.n_eval_tasks if n_eval_tasks is None else n_eval_tasks,
        )
        clone.rng = np.random.default_rng(seed)
        clone.facets = list(self.facets)
        clone.facet_sources = dict(self.facet_sources)
        clone.facet_descriptions = dict(self.facet_descriptions)
        clone.agents = [agent.copy() for agent in self.agents]
        clone.memories = self.memories.copy()
        clone.train_tasks = clone._generate_tasks(
            clone.config.n_train_tasks, prefix="train", seed=seed * 1000 + 1
        )
        clone.eval_tasks = clone._generate_tasks(
            clone.config.n_eval_tasks, prefix="eval", seed=seed * 1000 + 2
        )
        clone._success_cache = {}
        clone._value_cache = {}
        clone._bundle_cache = {}
        return clone

    def clone_with_shifted_tasks(
        self,
        seed: int,
        *,
        shift_strength: float = 0.7,
        n_eval_tasks: int | None = None,
    ) -> "MemoryBenchmark":
        """Clone with a held-out task distribution that emphasises rare facets.

        The training stream is the original benchmark.  The held-out stream
        upweights the facets that were *rare* in training, which is a controlled
        distribution shift: a memory that looks useless on the training sample
        can be valuable on the shifted stream.  This is the setting in which a
        self-validated RSI is most likely to make a harmful edit.
        """

        counts = {facet: 0 for facet in self.facets}
        for task in self.train_tasks:
            for facet in task.required_facets:
                counts[facet] = counts.get(facet, 0) + 1
        n_facets = max(1, len(self.facets))
        uniform = 1.0 / n_facets
        ranked = sorted(self.facets, key=lambda f: counts.get(f, 0))
        n_edge = max(1, n_facets // 3)
        rare = set(ranked[:n_edge])
        common = set(ranked[-n_edge:])
        raw = {
            facet: (3.0 if facet in rare else 0.3 if facet in common else 1.0)
            for facet in self.facets
        }
        denom = sum(raw.values())
        weights = {
            facet: (1.0 - shift_strength) * uniform
            + shift_strength * raw[facet] / denom
            for facet in self.facets
        }
        clone = self.clone_with_tasks(seed, n_eval_tasks=n_eval_tasks)
        clone.config = replace(clone.config, task_facet_weights=weights)
        clone.train_tasks = clone._generate_tasks(
            clone.config.n_train_tasks, prefix="train", seed=seed * 1000 + 1
        )
        clone.eval_tasks = clone._generate_tasks(
            clone.config.n_eval_tasks, prefix="eval", seed=seed * 1000 + 2
        )
        clone._success_cache = {}
        clone._value_cache = {}
        clone._bundle_cache = {}
        return clone

    def clone_with_fresh_agents(
        self,
        seed: int,
        *,
        n_eval_tasks: int | None = None,
        shifted: bool = False,
        shift_strength: float = 0.7,
    ) -> "MemoryBenchmark":
        if shifted:
            clone = self.clone_with_shifted_tasks(
                seed,
                shift_strength=shift_strength,
                n_eval_tasks=n_eval_tasks,
            )
        else:
            clone = self.clone_with_tasks(seed, n_eval_tasks=n_eval_tasks)
        clone.agents = []
        clone.rng = np.random.default_rng(seed + 991)
        clone._build_agents()
        clone._success_cache = {}
        clone._value_cache = {}
        clone._bundle_cache = {}
        return clone

    def clone_with_store(
        self,
        store: MemoryStore,
        *,
        seed: int | None = None,
        fresh_tasks: bool = True,
    ) -> "MemoryBenchmark":
        """Clone the benchmark with a different memory store.

        ``fresh_tasks=False`` reproduces the original task sample (the original
        seed); ``fresh_tasks=True`` samples a new task seed.  Agents and facets
        are kept fixed, which isolates the effect of the memory portfolio.
        """

        seed = self.config.seed if seed is None else seed
        clone = self.clone_with_tasks(seed) if fresh_tasks else self.clone_with_tasks(
            self.config.seed
        )
        clone.memories = store.copy()
        clone._success_cache = {}
        clone._value_cache = {}
        clone._bundle_cache = {}
        return clone

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _agent(self, agent_id: str) -> AgentProfile:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        raise KeyError(f"Unknown agent: {agent_id}")

    def _tasks(self, split: str) -> list[Task]:
        return self.train_tasks if split == "train" else self.eval_tasks

    def describe(self) -> dict:
        return {
            "n_agents": len(self.agents),
            "n_facets": len(self.facets),
            "n_memories": len(self.memories),
            "n_train_tasks": len(self.train_tasks),
            "n_eval_tasks": len(self.eval_tasks),
            "agents": [
                {
                    "id": agent.id,
                    "skill": round(agent.skill, 3),
                    "known_facets": {
                        facet: round(value, 3)
                        for facet, value in agent.private_knowledge.items()
                        if value > 0.05
                    },
                }
                for agent in self.agents
            ],
            "memories": self.memories.summary(),
        }
