"""Public memory pool (team commons) for AgoraPlanner.

This module is deliberately separate from the bilateral market:

- private sellers price scoped memory offers;
- the public pool is a reviewed catalog of reusable memory with access
  licenses, lineage, review quorum, embargo, source diversity and reputation.

The pool does not receive a market price.  Its governance primitives are
review/admission, embargo, source diversity and reputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from agora_planner.demand_first_rsi import MemoryRecord


@dataclass
class PoolEntry:
    memory: MemoryRecord
    license: str = "team"
    review_status: str = "pending"  # pending | approved | rejected
    reviews: list[tuple[str, float]] = field(default_factory=list)
    publish_step: int = 0
    embargo_until: int = 0
    reputation: float = 0.5
    usage_count: int = 0
    success_count: int = 0
    source: str = ""

    def available(self, step: int) -> bool:
        return (
            self.review_status == "approved"
            and step >= self.embargo_until
            and self.license in ("team", "public")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory.id,
            "license": self.license,
            "review_status": self.review_status,
            "reviews": [{"reviewer": r, "score": s} for r, s in self.reviews],
            "publish_step": self.publish_step,
            "embargo_until": self.embargo_until,
            "reputation": round(self.reputation, 4),
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "source": self.source,
        }


@dataclass
class PoolPolicy:
    review_quorum: int = 2
    review_threshold: float = 0.5
    quality_threshold: float = 0.5
    max_correlation: float = 0.6
    embargo_steps: int = 0
    source_cap: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_quorum": self.review_quorum,
            "review_threshold": self.review_threshold,
            "quality_threshold": self.quality_threshold,
            "max_correlation": self.max_correlation,
            "embargo_steps": self.embargo_steps,
            "source_cap": self.source_cap,
        }


class MemoryPool:
    """A reviewed catalog, not a market.

    A memory enters only after quality/correlation checks and a review quorum.
    Access is an access license, not ownership transfer.  The pool has no
    ask/bid; usage updates reputation, not price.
    """

    def __init__(self, policy: PoolPolicy | None = None):
        self.policy = policy or PoolPolicy()
        self.entries: dict[str, PoolEntry] = {}

    def publish(
        self,
        memory: MemoryRecord,
        *,
        step: int = 0,
        license: str = "team",
        embargo_steps: int | None = None,
    ) -> PoolEntry:
        if memory.quality < self.policy.quality_threshold:
            raise ValueError(f"memory {memory.id} below pool quality threshold")
        if memory.correlation_risk > self.policy.max_correlation:
            raise ValueError(f"memory {memory.id} above pool correlation ceiling")
        if memory.id in self.entries:
            return self.entries[memory.id]
        embargo = self.policy.embargo_steps if embargo_steps is None else embargo_steps
        entry = PoolEntry(
            memory=memory,
            license=license,
            publish_step=step,
            embargo_until=step + max(0, int(embargo)),
            source=memory.source,
        )
        self.entries[memory.id] = entry
        return entry

    def review(
        self,
        memory_id: str,
        reviewer: str,
        score: float,
        *,
        conflict: bool = False,
    ) -> PoolEntry:
        entry = self.entries[memory_id]
        if conflict:
            return entry
        entry.reviews.append((reviewer, float(score)))
        if len(entry.reviews) >= self.policy.review_quorum:
            mean = sum(s for _, s in entry.reviews) / len(entry.reviews)
            entry.review_status = "approved" if mean >= self.policy.review_threshold else "rejected"
        return entry

    def force_approve(self, memory_id: str) -> PoolEntry:
        entry = self.entries[memory_id]
        entry.review_status = "approved"
        return entry

    def available(self, step: int, *, facet: str | None = None, limit: int = 8) -> list[PoolEntry]:
        entries = [
            e for e in self.entries.values()
            if e.available(step) and (facet is None or e.memory.facet == facet)
        ]
        entries.sort(key=lambda e: (e.reputation, e.memory.quality), reverse=True)
        if self.policy.source_cap > 0:
            selected: list[PoolEntry] = []
            counts: dict[str, int] = {}
            for entry in entries:
                if counts.get(entry.source, 0) >= self.policy.source_cap:
                    continue
                selected.append(entry)
                counts[entry.source] = counts.get(entry.source, 0) + 1
                if len(selected) >= limit:
                    break
            return selected
        return entries[:limit]

    def record_usage(self, memory_ids: Iterable[str], success: bool) -> None:
        for mid in memory_ids:
            entry = self.entries.get(mid)
            if entry is None:
                continue
            entry.usage_count += 1
            if success:
                entry.success_count += 1
                entry.reputation = min(1.0, entry.reputation + 0.05)
            else:
                entry.reputation = max(0.0, entry.reputation - 0.05)

    def stats(self) -> dict[str, Any]:
        approved = [e for e in self.entries.values() if e.review_status == "approved"]
        return {
            "n_entries": len(self.entries),
            "n_approved": len(approved),
            "n_pending": sum(1 for e in self.entries.values() if e.review_status == "pending"),
            "n_rejected": sum(1 for e in self.entries.values() if e.review_status == "rejected"),
            "mean_reputation": (
                sum(e.reputation for e in self.entries.values()) / max(1, len(self.entries))
            ),
            "policy": self.policy.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "stats": self.stats(),
            "entries": {mid: e.to_dict() for mid, e in self.entries.items()},
        }
