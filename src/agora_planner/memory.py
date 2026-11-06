"""Memory assets and the asset registry.

A memory is modelled as an asset with:

* ``content`` — the natural-language lesson;
* ``facet`` — the latent knowledge facet it covers (benchmark ground truth);
* ``source_class`` — where it came from (document, tool, human, derivation);
* ``quality`` — reliability in ``[0, 1]``;
* ``scope`` — the task family / regime in which it is valid;
* ``production_cost`` — the effort needed to produce and verify it;
* ``correlation_risk`` — how much sharing it across agents couples their errors.

The latent fields are intentionally part of the asset: the benchmark uses them to
compute ground-truth values, while an LLM producer only sees the natural-language
content and a task trace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable


@dataclass
class MemoryAsset:
    id: str
    content: str
    facet: str
    source_class: str
    quality: float
    scope: str
    version: int
    owner: str
    production_cost: float
    created_round: int = 0
    correlation_risk: float = 0.5
    origin: str = "producer"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quality"] = round(float(self.quality), 6)
        data["production_cost"] = round(float(self.production_cost), 6)
        data["correlation_risk"] = round(float(self.correlation_risk), 6)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryAsset":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def copy(self, **changes: Any) -> "MemoryAsset":
        return replace(self, **changes)


@dataclass
class Transformation:
    """A proposed edit to the memory registry."""

    op: str
    inputs: list[str]
    output: MemoryAsset | None = None
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "inputs": list(self.inputs),
            "output": self.output.to_dict() if self.output else None,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }


class MemoryStore:
    """A mutable registry of memory assets keyed by id."""

    def __init__(self, assets: Iterable[MemoryAsset] | None = None):
        self.assets: dict[str, MemoryAsset] = {}
        for asset in assets or []:
            self.add(asset)

    # ------------------------------------------------------------------
    # Basic registry operations
    # ------------------------------------------------------------------
    def add(self, asset: MemoryAsset) -> None:
        if asset.id in self.assets:
            raise ValueError(f"Duplicate memory id: {asset.id}")
        self.assets[asset.id] = asset

    def remove(self, asset_id: str) -> MemoryAsset:
        if asset_id not in self.assets:
            raise KeyError(f"Unknown memory id: {asset_id}")
        return self.assets.pop(asset_id)

    def get(self, asset_id: str) -> MemoryAsset:
        return self.assets[asset_id]

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self.assets

    def __len__(self) -> int:
        return len(self.assets)

    def __iter__(self):
        return iter(self.assets.values())

    def ids(self) -> list[str]:
        return sorted(self.assets)

    def all(self) -> list[MemoryAsset]:
        return list(self.assets.values())

    def by_facet(self, facet: str) -> list[MemoryAsset]:
        return [asset for asset in self.assets.values() if asset.facet == facet]

    def by_source(self, source_class: str) -> list[MemoryAsset]:
        return [
            asset
            for asset in self.assets.values()
            if asset.source_class == source_class
        ]

    def copy(self) -> "MemoryStore":
        return MemoryStore(asset.copy() for asset in self.assets.values())

    def to_dicts(self) -> list[dict[str, Any]]:
        return [self.assets[key].to_dict() for key in self.ids()]

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for asset in sorted(self.assets.values(), key=lambda a: a.id):
            rows.append(
                {
                    "id": asset.id,
                    "facet": asset.facet,
                    "source": asset.source_class,
                    "quality": round(asset.quality, 3),
                    "cost": round(asset.production_cost, 3),
                    "scope": asset.scope,
                    "owner": asset.owner,
                    "corr": round(asset.correlation_risk, 3),
                    "origin": asset.origin,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------
    def apply(self, transformation: Transformation) -> "MemoryStore":
        """Return a new store with the transformation applied (no mutation)."""

        new_store = self.copy()
        op = transformation.op
        inputs = list(transformation.inputs)

        if op == "retire":
            for asset_id in inputs:
                if asset_id in new_store:
                    new_store.remove(asset_id)
            return new_store

        if op in {"merge", "split", "refresh", "abstract", "add"}:
            if transformation.output is None:
                raise ValueError(f"Transformation {op} requires an output asset")
            for asset_id in inputs:
                if asset_id in new_store:
                    new_store.remove(asset_id)
            new_store.add(transformation.output)
            return new_store

        raise ValueError(f"Unknown transformation op: {op}")


def merge_assets(
    left: MemoryAsset,
    right: MemoryAsset,
    *,
    new_id: str,
    content: str | None = None,
    rationale: str = "",
) -> Transformation:
    """Merge two memories covering the same facet.

    Quality follows a noisy-OR: the merged memory is at least as reliable as
    either input.  If both inputs share a source class, the merged asset keeps
    that source (and therefore its correlation risk); if they come from
    different sources, the merge is treated as a cross-source synthesis and the
    correlation risk is reduced.
    """

    same_source = left.source_class == right.source_class
    poisoned = bool(
        left.metadata.get("poison") or right.metadata.get("poison")
    )
    quality = 1.0 - (1.0 - max(0.0, left.quality)) * (
        1.0 - max(0.0, right.quality)
    )
    quality = min(0.99, quality)
    if poisoned:
        # Merging a poisoned memory must not launder it into a high-quality
        # asset.  The merged record stays flagged and its effective quality is
        # discounted; a separate repair/verification step is required.
        quality = min(quality, 0.25)
    source = left.source_class if same_source else "derived"
    corr = (
        max(left.correlation_risk, right.correlation_risk)
        if same_source
        else min(left.correlation_risk, right.correlation_risk) * 0.8
    )
    metadata = {
        "merged_from": [left.id, right.id],
        "rationale": rationale,
    }
    if poisoned:
        metadata["poison"] = True
    merged = MemoryAsset(
        id=new_id,
        content=content
        or f"{left.content.strip()} {right.content.strip()}".strip(),
        facet=left.facet,
        source_class=source,
        quality=quality,
        scope=left.scope if left.scope == right.scope else "global",
        version=max(left.version, right.version) + 1,
        owner=left.owner,
        production_cost=max(left.production_cost, right.production_cost),
        created_round=max(left.created_round, right.created_round),
        correlation_risk=corr,
        origin="rsi_merge",
        metadata=metadata,
    )
    return Transformation(
        op="merge",
        inputs=[left.id, right.id],
        output=merged,
        rationale=rationale,
        metadata={"same_source": same_source},
    )


def refresh_asset(
    asset: MemoryAsset,
    *,
    new_id: str,
    quality_delta: float = 0.08,
    cost_delta: float = 0.01,
    content: str | None = None,
    rationale: str = "",
) -> Transformation:
    refreshed = asset.copy(
        id=new_id,
        content=content or asset.content,
        quality=min(0.99, max(0.0, asset.quality + quality_delta)),
        production_cost=max(0.0, asset.production_cost + cost_delta),
        version=asset.version + 1,
        origin="rsi_refresh",
        metadata={**asset.metadata, "refreshed_from": asset.id, "rationale": rationale},
    )
    return Transformation(
        op="refresh",
        inputs=[asset.id],
        output=refreshed,
        rationale=rationale,
    )


def split_asset(
    asset: MemoryAsset,
    *,
    new_ids: list[str],
    scopes: list[str],
    content_prefix: str = "",
    rationale: str = "",
) -> Transformation:
    if len(new_ids) != len(scopes) or not new_ids:
        raise ValueError("split_asset needs one id and one scope per split")
    # A split is represented as a single transformation whose output is the
    # first child; additional children are carried in metadata and materialised
    # by the caller.  This keeps the Transformation dataclass simple.
    first = asset.copy(
        id=new_ids[0],
        scope=scopes[0],
        content=f"{content_prefix}{asset.content}".strip(),
        version=asset.version + 1,
        origin="rsi_split",
        metadata={**asset.metadata, "split_from": asset.id, "split_group": new_ids},
    )
    return Transformation(
        op="split",
        inputs=[asset.id],
        output=first,
        rationale=rationale,
        metadata={
            "children": [
                asset.copy(
                    id=new_id,
                    scope=scope,
                    content=f"{content_prefix}{asset.content}".strip(),
                    version=asset.version + 1,
                    origin="rsi_split",
                    metadata={
                        **asset.metadata,
                        "split_from": asset.id,
                        "split_group": new_ids,
                    },
                ).to_dict()
                for new_id, scope in zip(new_ids, scopes)
            ]
        },
    )


def apply_transformation_with_children(
    store: MemoryStore, transformation: Transformation
) -> MemoryStore:
    """Apply a transformation, materialising split children when present."""

    if transformation.op == "split" and transformation.metadata.get("children"):
        new_store = store.copy()
        for asset_id in transformation.inputs:
            if asset_id in new_store:
                new_store.remove(asset_id)
        for child in transformation.metadata["children"]:
            new_store.add(MemoryAsset.from_dict(child))
        return new_store
    return store.apply(transformation)
