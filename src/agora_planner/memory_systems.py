"""Faithful, clearly-labelled reimplementations of memory-system baselines.

No official code is available in this offline environment for MemGPT/Letta
server (PyPI ``letta`` is Letta Code, not the MemGPT server), RAPTOR, CAMA,
GovMem, RCR-Router, or MemPrice.  These classes implement the *mechanisms
described in the papers* on top of the text-memory benchmark.  They are
reimplementations, not official reproductions; the report labels them as such.

The official systems that *are* runnable here are Mem0 (``mem0_adapter.py``)
and AutoGen (``autogen_adapter.py``).  GraphRAG has an official package; if its
install completes, ``graphrag_official.py`` is preferred over the
reimplementation here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .embeddings import CosineRetriever, HashEmbedder, RetrievedItem
from .memory import MemoryAsset


def _asset_texts(assets: Iterable[MemoryAsset]) -> list[tuple[str, MemoryAsset]]:
    return [(asset.id, asset) for asset in assets]


@dataclass
class _BaseMemorySystem:
    embedder: HashEmbedder = field(default_factory=HashEmbedder)
    assets: dict[str, MemoryAsset] = field(default_factory=dict)

    def ingest(self, assets: Iterable[MemoryAsset]) -> None:
        for asset in assets:
            self.assets[asset.id] = asset

    def search(self, agent_id: str, query: str, *, top_k: int = 3) -> list[RetrievedItem]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------
# MemGPT / Letta reimplementation
# ----------------------------------------------------------------------


class MemGPTReimpl(_BaseMemorySystem):
    """Core / recall / archival memory hierarchy from the MemGPT paper.

    * core memory: a small fixed set of high-importance facts, always in context;
    * recall memory: the most recent facts, searched by recency + relevance;
    * archival memory: all facts, searched by embedding similarity.

    The official MemGPT agent uses an LLM to edit core memory and issue
    retrieval calls.  This deterministic policy uses the same memory tiers and
    ranking signals without the LLM controller.
    """

    def __init__(
        self,
        *,
        core_slots: int = 2,
        recall_slots: int = 3,
        embedder: HashEmbedder | None = None,
    ):
        super().__init__(embedder=embedder or HashEmbedder())
        self.core_slots = int(core_slots)
        self.recall_slots = int(recall_slots)
        self._retriever = CosineRetriever(self.embedder)
        self._core: list[str] = []
        self._recall: list[str] = []

    def ingest(self, assets: Iterable[MemoryAsset]) -> None:
        assets = list(assets)
        super().ingest(assets)
        self._retriever = CosineRetriever(self.embedder)
        self._retriever.extend([(asset.id, asset.content) for asset in assets])
        ranked = sorted(
            assets,
            key=lambda asset: (
                float(asset.quality),
                float(asset.created_round),
            ),
            reverse=True,
        )
        self._core = [asset.id for asset in ranked[: self.core_slots]]
        recent = sorted(assets, key=lambda asset: int(asset.created_round), reverse=True)
        self._recall = [asset.id for asset in recent[: self.recall_slots]]

    def search(self, agent_id: str, query: str, *, top_k: int = 3) -> list[RetrievedItem]:
        candidates: dict[str, RetrievedItem] = {}
        for mid in self._core:
            asset = self.assets.get(mid)
            if asset is not None:
                candidates[mid] = RetrievedItem(
                    text=asset.content,
                    score=1.0 + 0.1 * float(asset.quality),
                    memory_id=mid,
                    metadata={"tier": "core"},
                )
        for tier, ids in (("archival", None), ("recall", self._recall)):
            if ids is None:
                items = self._retriever.search(query, top_k=max(top_k, 4))
            else:
                items = []
                for item in self._retriever.search(query, top_k=max(len(self._recall), top_k)):
                    if item.memory_id in set(ids):
                        items.append(item)
            for item in items:
                mid = item.memory_id or ""
                asset = self.assets.get(mid)
                if asset is None:
                    continue
                recency = 1.0 / (1.0 + max(0, 6 - int(asset.created_round)))
                score = 0.70 * item.score + 0.22 * float(asset.quality) + 0.08 * recency
                existing = candidates.get(mid)
                if existing is None or score > existing.score:
                    candidates[mid] = RetrievedItem(
                        text=asset.content,
                        score=float(score),
                        memory_id=mid,
                        metadata={"tier": tier},
                    )
        return sorted(candidates.values(), key=lambda item: item.score, reverse=True)[: max(1, int(top_k))]


# ----------------------------------------------------------------------
# RAPTOR reimplementation
# ----------------------------------------------------------------------


@dataclass
class _RaptorNode:
    id: str
    text: str
    level: int
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RAPTORReimpl(_BaseMemorySystem):
    """Recursive clustering + summarisation tree retrieval.

    The paper builds a tree by clustering chunks and summarising each cluster
    with an LLM.  This implementation uses facet/topic clusters and *extractive*
    summaries (the highest-quality fact texts in a cluster).  Retrieval is the
    collapsed-tree mode: search all leaf and summary nodes, then return the
    top-k nodes.  This preserves the key property that a summary can answer a
    query without expanding every child, while avoiding an LLM dependency.
    """

    def __init__(
        self,
        *,
        branch_facts: int = 3,
        use_quality: bool = True,
        embedder: HashEmbedder | None = None,
    ):
        super().__init__(embedder=embedder or HashEmbedder())
        self.branch_facts = int(branch_facts)
        self.use_quality = bool(use_quality)
        self.nodes: dict[str, _RaptorNode] = {}
        self._retriever = CosineRetriever(self.embedder)

    def ingest(self, assets: Iterable[MemoryAsset]) -> None:
        assets = list(assets)
        super().ingest(assets)
        self.nodes = {}
        self._retriever = CosineRetriever(self.embedder)

        by_facet: dict[str, list[MemoryAsset]] = defaultdict(list)
        for asset in assets:
            by_facet[asset.facet].append(asset)

        level1_ids: list[str] = []
        for facet, group in sorted(by_facet.items()):
            ranked = sorted(
                group,
                key=lambda asset: (
                    float(asset.quality) if self.use_quality else -int(asset.created_round),
                    asset.id,
                ),
                reverse=True,
            )
            children = [asset.id for asset in ranked[: self.branch_facts]]
            summary_text = " ".join(
                self.assets[mid].content for mid in children if mid in self.assets
            )
            node = _RaptorNode(
                id=f"summary_L1_{facet}",
                text=summary_text,
                level=1,
                children=children,
                metadata={"facet": facet, "is_summary": True, "level": 1},
            )
            self.nodes[node.id] = node
            level1_ids.append(node.id)

        if level1_ids:
            root_children = sorted(
                level1_ids,
                key=lambda nid: len(self.nodes[nid].children),
                reverse=True,
            )[: max(3, self.branch_facts)]
            root_text = " ".join(self.nodes[nid].text for nid in root_children)
            root = _RaptorNode(
                id="summary_L2_root",
                text=root_text,
                level=2,
                children=root_children,
                metadata={"is_summary": True, "level": 2},
            )
            self.nodes[root.id] = root

        for asset in assets:
            self.nodes[asset.id] = _RaptorNode(
                id=asset.id,
                text=asset.content,
                level=0,
                metadata={"facet": asset.facet, "is_summary": False, "quality": asset.quality},
            )
        self._retriever.extend([(node.id, node.text) for node in self.nodes.values()])

    def search(self, agent_id: str, query: str, *, top_k: int = 3) -> list[RetrievedItem]:
        items = self._retriever.search(query, top_k=max(top_k, 4))
        results: list[RetrievedItem] = []
        for item in items:
            node = self.nodes.get(item.memory_id or "")
            if node is None:
                continue
            results.append(
                RetrievedItem(
                    text=node.text,
                    score=item.score,
                    memory_id=node.id,
                    metadata=dict(node.metadata),
                )
            )
        return results[: max(1, int(top_k))]


# ----------------------------------------------------------------------
# GraphRAG reimplementation
# ----------------------------------------------------------------------


class GraphRAGReimpl(_BaseMemorySystem):
    """Entity/facet graph + community summaries + local search.

    GraphRAG's official pipeline uses an LLM for entity/relation extraction,
    Leiden community detection, and community reports.  This implementation
    builds a provenance/facet graph deterministically, uses connected
    components as communities, and creates extractive community summaries.
    Local search expands from embedding seeds through the graph; global search
    returns community summaries.
    """

    def __init__(
        self,
        *,
        expand_neighbors: int = 3,
        use_quality: bool = True,
        embedder: HashEmbedder | None = None,
    ):
        super().__init__(embedder=embedder or HashEmbedder())
        self.expand_neighbors = int(expand_neighbors)
        self.use_quality = bool(use_quality)
        self._retriever = CosineRetriever(self.embedder)
        self.graph: dict[str, set[str]] = defaultdict(set)
        self.communities: dict[str, list[str]] = {}
        self.summaries: dict[str, str] = {}

    def ingest(self, assets: Iterable[MemoryAsset]) -> None:
        assets = list(assets)
        super().ingest(assets)
        self.graph = defaultdict(set)
        self.communities = {}
        self.summaries = {}
        self._retriever = CosineRetriever(self.embedder)
        self._retriever.extend([(asset.id, asset.content) for asset in assets])

        # Bipartite-style graph: facts linked when they share a facet, topic,
        # source class, or owner.  This is the deterministic analogue of an
        # entity co-occurrence graph.
        keys: dict[tuple[str, str], list[str]] = defaultdict(list)
        for asset in assets:
            keys[("facet", asset.facet)].append(asset.id)
            keys[("source", asset.source_class)].append(asset.id)
            keys[("owner", asset.owner)].append(asset.id)
            topic = str(asset.metadata.get("topic", ""))
            if topic:
                keys[("topic", topic)].append(asset.id)
        for ids in keys.values():
            for left in ids:
                for right in ids:
                    if left != right:
                        self.graph[left].add(right)

        # Connected components on the fact graph.
        seen: set[str] = set()
        for asset in assets:
            if asset.id in seen:
                continue
            stack = [asset.id]
            component: list[str] = []
            seen.add(asset.id)
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbor in self.graph.get(node, set()):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            cid = f"community_{len(self.communities):03d}"
            self.communities[cid] = component
            ranked = sorted(
                component,
                key=lambda mid: (
                    float(self.assets[mid].quality)
                    if self.use_quality
                    else -int(self.assets[mid].created_round),
                    mid,
                ),
                reverse=True,
            )
            self.summaries[cid] = " ".join(
                self.assets[mid].content for mid in ranked[: self.expand_neighbors]
            )

    def search(
        self,
        agent_id: str,
        query: str,
        *,
        top_k: int = 3,
        global_search: bool = False,
    ) -> list[RetrievedItem]:
        if global_search:
            scored = []
            for cid, text in self.summaries.items():
                emb = self.embedder.embed(text)
                scored.append((float(emb @ self.embedder.embed(query)), cid, text))
            scored.sort(reverse=True)
            return [
                RetrievedItem(text=text, score=score, memory_id=cid, metadata={"community": cid})
                for score, cid, text in scored[: max(1, int(top_k))]
            ]

        seeds = self._retriever.search(query, top_k=max(2, top_k))
        candidate_ids: set[str] = {item.memory_id for item in seeds if item.memory_id}
        for item in seeds:
            if item.memory_id:
                neighbors = sorted(
                    self.graph.get(item.memory_id, set()),
                    key=lambda mid: (
                        float(self.assets[mid].quality)
                        if self.use_quality
                        else -int(self.assets[mid].created_round),
                        mid,
                    ),
                    reverse=True,
                )[: self.expand_neighbors]
                candidate_ids.update(neighbors)
        scored: list[RetrievedItem] = []
        q = self.embedder.embed(query)
        for mid in candidate_ids:
            asset = self.assets.get(mid)
            if asset is None:
                continue
            relevance = float(self.embedder.embed(asset.content) @ q)
            graph_bonus = 0.05 if any(
                mid in self.graph.get(item.memory_id, set()) for item in seeds if item.memory_id
            ) else 0.0
            score = (
                relevance
                + (0.15 * float(asset.quality) if self.use_quality else 0.0)
                + graph_bonus
            )
            scored.append(
                RetrievedItem(
                    text=asset.content,
                    score=score,
                    memory_id=mid,
                    metadata={"facet": asset.facet, "source": asset.source_class},
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(1, int(top_k))]


# ----------------------------------------------------------------------
# CAMA reimplementation
# ----------------------------------------------------------------------


class CAMAReimpl(_BaseMemorySystem):
    """Correlation-aware memory arbitration (CAMA), reimplemented.

    CAMA's central claim is that repeated memories from the same latent source
    must not be counted as independent votes.  This implementation retrieves a
    candidate pool, groups candidates by ``(facet, answer code)``, discounts
    support from the same source/owner, and chooses the code with the highest
    independence-weighted support.  It optionally retrieves an alternative
    source when the top group is contested.
    """

    def __init__(
        self,
        *,
        candidate_k: int = 8,
        use_quality: bool = True,
        embedder: HashEmbedder | None = None,
    ):
        super().__init__(embedder=embedder or HashEmbedder())
        self.candidate_k = int(candidate_k)
        self.use_quality = bool(use_quality)
        self._retriever = CosineRetriever(self.embedder)

    def ingest(self, assets: Iterable[MemoryAsset]) -> None:
        assets = list(assets)
        super().ingest(assets)
        self._retriever = CosineRetriever(self.embedder)
        self._retriever.extend([(asset.id, asset.content) for asset in assets])

    @staticmethod
    def _code(asset: MemoryAsset) -> str:
        return str(asset.metadata.get("answer_code", ""))

    def search(self, agent_id: str, query: str, *, top_k: int = 3) -> list[RetrievedItem]:
        candidates = self._retriever.search(query, top_k=max(self.candidate_k, top_k))
        groups: dict[tuple[str, str], list[tuple[float, MemoryAsset]]] = defaultdict(list)
        for item in candidates:
            asset = self.assets.get(item.memory_id or "")
            if asset is None:
                continue
            groups[(asset.facet, self._code(asset))].append((item.score, asset))

        best_key: tuple[str, str] | None = None
        best_score = -1.0
        for key, members in groups.items():
            # Independence-weighted support: one vote per (source_class, owner).
            support: dict[tuple[str, str], float] = {}
            for relevance, asset in members:
                vote_key = (asset.source_class, asset.owner)
                vote = relevance * (float(asset.quality) if self.use_quality else 1.0)
                support[vote_key] = max(support.get(vote_key, 0.0), vote)
            total = sum(support.values())
            # A group with many duplicate same-source facts gets no extra vote.
            score = total
            if score > best_score:
                best_score = score
                best_key = key
        if best_key is None:
            return []

        selected: list[RetrievedItem] = []
        # Evidence for the winning code first, then the strongest alternatives.
        ordered_keys = sorted(
            groups,
            key=lambda key: (
                key == best_key,
                max(
                    relevance * (float(asset.quality) if self.use_quality else 1.0)
                    for relevance, asset in groups[key]
                ),
            ),
            reverse=True,
        )
        for key in ordered_keys:
            for relevance, asset in sorted(
                groups[key],
                key=lambda pair: pair[0]
                * (float(pair[1].quality) if self.use_quality else 1.0),
                reverse=True,
            ):
                selected.append(
                    RetrievedItem(
                        text=asset.content,
                        score=float(relevance),
                        memory_id=asset.id,
                        metadata={
                            "facet": asset.facet,
                            "code": self._code(asset),
                            "independent_sources": len(
                                {(a.source_class, a.owner) for _, a in groups[key]}
                            ),
                            "chosen": key == best_key,
                        },
                    )
                )
        return selected[: max(1, int(top_k))]


# ----------------------------------------------------------------------
# GovMem reimplementation
# ----------------------------------------------------------------------


@dataclass
class GovMemDecision:
    memory_id: str
    decision: str
    support: float
    counterevidence: float
    quality: float
    reason: str


class GovMemReimpl(_BaseMemorySystem):
    """Write-path governance: promote / reject / needs-review.

    The paper estimates dependency-aware support, retrieves counterevidence,
    assigns scope, and gates promotion.  This implementation computes support
    from independent ``(source_class, owner)`` pairs, counterevidence from
    conflicting codes on the same facet, and applies conservative thresholds.
    Only promoted facts enter the underlying store.
    """

    def __init__(
        self,
        *,
        promote_support: float = 0.8,
        reject_quality: float = 0.30,
        embedder: HashEmbedder | None = None,
    ):
        super().__init__(embedder=embedder or HashEmbedder())
        self.promote_support = float(promote_support)
        self.reject_quality = float(reject_quality)
        self.decisions: list[GovMemDecision] = []
        self._retriever = CosineRetriever(self.embedder)

    def ingest(self, assets: Iterable[MemoryAsset]) -> None:
        assets = list(assets)
        self.assets = {asset.id: asset for asset in assets}
        self.decisions = []
        by_facet: dict[str, list[MemoryAsset]] = defaultdict(list)
        for asset in assets:
            by_facet[asset.facet].append(asset)

        promoted: list[MemoryAsset] = []
        for asset in assets:
            same = by_facet[asset.facet]
            code = str(asset.metadata.get("answer_code", ""))
            # Keep only one vote per independent provenance pair.
            independent = {
                (other.source_class, other.owner)
                for other in same
                if str(other.metadata.get("answer_code", "")) == code
            }
            support = float(
                sum(
                    max(
                        float(other.quality)
                        for other in same
                        if (other.source_class, other.owner) == pair
                        and str(other.metadata.get("answer_code", "")) == code
                    )
                    for pair in independent
                )
            )
            counter = sum(
                float(other.quality)
                for other in same
                if str(other.metadata.get("answer_code", "")) != code
            )
            quality = float(asset.quality)
            if quality < self.reject_quality or counter > support + 0.25:
                decision, reason = "reject", "low quality or dominant counterevidence"
            elif support >= self.promote_support and counter <= support:
                decision, reason = "promote", "independent support exceeds threshold"
            else:
                decision, reason = "needs_review", "insufficient independent support"
            self.decisions.append(
                GovMemDecision(
                    memory_id=asset.id,
                    decision=decision,
                    support=support,
                    counterevidence=counter,
                    quality=quality,
                    reason=reason,
                )
            )
            if decision == "promote":
                promoted.append(asset)
        self.promoted = promoted
        self._retriever = CosineRetriever(self.embedder)
        self._retriever.extend([(asset.id, asset.content) for asset in promoted])

    def search(self, agent_id: str, query: str, *, top_k: int = 3) -> list[RetrievedItem]:
        return self._retriever.search(query, top_k=max(1, int(top_k)))

    def summary(self) -> dict:
        counts: dict[str, int] = defaultdict(int)
        for decision in self.decisions:
            counts[decision.decision] += 1
        return {
            "n_candidates": len(self.decisions),
            "promoted": counts["promote"],
            "rejected": counts["reject"],
            "needs_review": counts["needs_review"],
            "promoted_ids": [asset.id for asset in self.promoted],
        }


# ----------------------------------------------------------------------
# RCR-Router reimplementation
# ----------------------------------------------------------------------


class RCRRouterReimpl(_BaseMemorySystem):
    """Role-aware context routing with a per-agent token budget.

    RCR-Router scores a shared memory store by role, task stage, and relevance,
    then selects a subset under a strict per-agent token budget.  This
    implementation uses a static role/source affinity as the role signal and a
    size-aware greedy selection.  It is a retrieval policy, not the paper's
    full iterative write-back router.
    """

    _ROLE_AFFINITY = {
        "researcher": {"doc": 1.0, "human": 0.9, "tool": 0.7, "derived": 0.6},
        "coder": {"tool": 1.0, "doc": 0.8, "derived": 0.9, "human": 0.5},
        "ops": {"tool": 1.0, "human": 0.9, "doc": 0.7, "derived": 0.6},
    }

    def __init__(
        self,
        *,
        token_budget: int = 40,
        use_quality: bool = True,
        embedder: HashEmbedder | None = None,
    ):
        super().__init__(embedder=embedder or HashEmbedder())
        self.token_budget = int(token_budget)
        self.use_quality = bool(use_quality)
        self._retriever = CosineRetriever(self.embedder)
        self.roles: dict[str, str] = {}

    def set_roles(self, roles: dict[str, str]) -> None:
        self.roles = dict(roles)

    def ingest(self, assets: Iterable[MemoryAsset]) -> None:
        assets = list(assets)
        super().ingest(assets)
        self._retriever = CosineRetriever(self.embedder)
        self._retriever.extend([(asset.id, asset.content) for asset in assets])

    def search(self, agent_id: str, query: str, *, top_k: int = 3) -> list[RetrievedItem]:
        candidates = self._retriever.search(query, top_k=max(top_k * 4, 12))
        role = self.roles.get(agent_id, "researcher")
        affinity = self._ROLE_AFFINITY.get(role, self._ROLE_AFFINITY["researcher"])
        selected: list[RetrievedItem] = []
        used_tokens = 0
        for item in candidates:
            asset = self.assets.get(item.memory_id or "")
            if asset is None:
                continue
            score = (
                item.score
                * (0.5 + 0.5 * affinity.get(asset.source_class, 0.5))
                * (0.5 + 0.5 * float(asset.quality) if self.use_quality else 1.0)
            )
            tokens = len(asset.content.split())
            if selected and used_tokens + tokens > self.token_budget:
                continue
            selected.append(
                RetrievedItem(
                    text=asset.content,
                    score=float(score),
                    memory_id=asset.id,
                    metadata={"role": role, "tokens": tokens},
                )
            )
            used_tokens += tokens
            if len(selected) >= max(1, int(top_k)):
                break
        return selected


# ----------------------------------------------------------------------
# MemPrice reimplementation
# ----------------------------------------------------------------------


class MemPriceReimpl(_BaseMemorySystem):
    """Online context allocation with an adaptive price on occupancy.

    MemPrice treats context as a shared resource and raises a price when
    occupancy is above target.  This implementation maintains a per-agent
    Lagrange-style price ``lambda`` updated from realised token occupancy, and
    admits a memory only if ``relevance * quality / size >= lambda``.
    """

    def __init__(
        self,
        *,
        target_tokens: int = 40,
        eta: float = 0.15,
        initial_price: float = 0.05,
        use_quality: bool = True,
        embedder: HashEmbedder | None = None,
    ):
        super().__init__(embedder=embedder or HashEmbedder())
        self.target_tokens = int(target_tokens)
        self.eta = float(eta)
        self.price = float(initial_price)
        self.use_quality = bool(use_quality)
        self._prices: dict[str, float] = {}
        self._retriever = CosineRetriever(self.embedder)

    def ingest(self, assets: Iterable[MemoryAsset]) -> None:
        assets = list(assets)
        super().ingest(assets)
        self._retriever = CosineRetriever(self.embedder)
        self._retriever.extend([(asset.id, asset.content) for asset in assets])

    def _price(self, agent_id: str) -> float:
        return self._prices.setdefault(agent_id, self.price)

    def search(self, agent_id: str, query: str, *, top_k: int = 3) -> list[RetrievedItem]:
        candidates = self._retriever.search(query, top_k=max(top_k * 4, 12))
        price = self._price(agent_id)
        scored: list[tuple[float, RetrievedItem, int]] = []
        for item in candidates:
            asset = self.assets.get(item.memory_id or "")
            if asset is None:
                continue
            tokens = max(1, len(asset.content.split()))
            density = (
                item.score
                * (0.5 + 0.5 * float(asset.quality) if self.use_quality else 1.0)
                / tokens
            )
            scored.append((density, item, tokens))
        scored.sort(key=lambda triple: triple[0], reverse=True)
        selected: list[RetrievedItem] = []
        used = 0
        for density, item, tokens in scored:
            if density < price and selected:
                continue
            if selected and used + tokens > self.target_tokens:
                continue
            selected.append(
                RetrievedItem(
                    text=item.text,
                    score=float(density),
                    memory_id=item.memory_id,
                    metadata={"price": price, "tokens": tokens},
                )
            )
            used += tokens
            if len(selected) >= max(1, int(top_k)):
                break
        # Dual update: occupancy above target raises the price.
        ratio = used / max(1, self.target_tokens)
        price *= float(np.exp(self.eta * (ratio - 1.0)))
        price = float(np.clip(price, 1e-4, 10.0))
        self._prices[agent_id] = price
        return selected

    def summary(self) -> dict:
        return {"prices": {agent: round(value, 6) for agent, value in sorted(self._prices.items())}}


__all__ = [
    "MemGPTReimpl",
    "RAPTORReimpl",
    "GraphRAGReimpl",
    "CAMAReimpl",
    "GovMemReimpl",
    "RCRRouterReimpl",
    "MemPriceReimpl",
]
