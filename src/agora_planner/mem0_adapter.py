"""Official Mem0 adapter (``mem0ai``) for the text-memory benchmark.

This module wraps the *official* package, not a policy proxy.  It is kept
optional: the package is installed in the repository-local ``.vendor``
directory because this workstation cannot use pip's default proxy path.

Design notes
------------
* The official package is configured with Qdrant in local mode and a
  deterministic hash embedder.  Mem0 does not ship a hash embedder, so we
  register one through ``EmbedderFactory``.  This is an embedding substitution,
  not a change to Mem0's memory logic.
* ``infer=False`` stores each benchmark fact as a raw memory.  This is the
  reproducible, LLM-free mode used for the main table.
* ``infer=True`` exercises Mem0's LLM extraction/update path.  DeepSeek is
  reached through an ``httpx`` client with ``trust_env=False`` because the
  system proxy on this machine cannot reach the API.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .embeddings import HashEmbedder, RetrievedItem
from .memory import MemoryAsset

_VENDOR = Path(__file__).resolve().parents[2] / ".vendor"
if _VENDOR.exists() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# Telemetry must be disabled before importing mem0.
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("MEM0_TELEMETRY_SAMPLE_RATE", "0")

_MEM0_READY = False
_MEM0_IMPORT_ERROR: Exception | None = None


def _prepare_mem0() -> None:
    """Import mem0 and install the hash-embedder / DeepSeek patches once."""

    global _MEM0_READY, _MEM0_IMPORT_ERROR
    if _MEM0_READY:
        return
    if _MEM0_IMPORT_ERROR is not None:
        raise RuntimeError(f"Mem0 is unavailable: {_MEM0_IMPORT_ERROR}") from _MEM0_IMPORT_ERROR

    try:
        import httpx
        from mem0.configs.embeddings.base import BaseEmbedderConfig
        from mem0.embeddings.base import EmbeddingBase
        from mem0.utils import factory as factory_mod

        class HashEmbedding(EmbeddingBase):
            """Mem0-compatible wrapper around :class:`HashEmbedder`."""

            def __init__(self, config=None, dim: int = 384):
                super().__init__(config)
                self.dim = int(dim)
                self._embedder = HashEmbedder(dim=self.dim)

            def embed(self, text, memory_action=None):
                return self._embedder.embed(str(text)).tolist()

        original_create = factory_mod.EmbedderFactory.create.__func__

        def create(cls, provider_name, config, vector_config=None):
            config = config or {}
            if provider_name == "openai" and config.get("model") == "hash-embedding":
                dim = int(config.get("embedding_dims", 384))
                return HashEmbedding(BaseEmbedderConfig(**config), dim=dim)
            return original_create(cls, provider_name, config, vector_config)

        factory_mod.EmbedderFactory.create = classmethod(create)

        # DeepSeekLLM builds ``OpenAI(...)`` without an httpx client.  On this
        # machine the system proxy breaks that connection, so inject a
        # trust_env=False client.
        import mem0.llms.deepseek as deepseek_mod

        original_openai = deepseek_mod.OpenAI

        def openai_client(*args, **kwargs):
            kwargs.setdefault("http_client", httpx.Client(trust_env=False))
            return original_openai(*args, **kwargs)

        deepseek_mod.OpenAI = openai_client
        _MEM0_READY = True
    except Exception as exc:  # pragma: no cover - environment-specific
        _MEM0_IMPORT_ERROR = exc
        raise RuntimeError(f"Mem0 is unavailable: {exc}") from exc


@dataclass
class Mem0Settings:
    infer: bool = False
    embedding_dims: int = 384
    collection_prefix: str = "agora"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None


class OfficialMem0Store:
    """A single Mem0 namespace backed by local Qdrant."""

    def __init__(
        self,
        namespace: str,
        *,
        settings: Mem0Settings | None = None,
        workdir: str | Path | None = None,
    ):
        _prepare_mem0()
        from mem0 import Memory

        self.namespace = namespace
        self.settings = settings or Mem0Settings()
        self._workdir = Path(workdir) if workdir else Path(
            tempfile.mkdtemp(prefix="agora_mem0_")
        )
        self._workdir.mkdir(parents=True, exist_ok=True)
        collection = (
            f"{self.settings.collection_prefix}_{namespace}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        collection = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in collection)

        llm_config: dict[str, Any]
        if self.settings.infer:
            from .llm import _load_dotenv

            _load_dotenv()
            api_key = (
                self.settings.llm_api_key
                or os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            base_url = (
                self.settings.llm_base_url
                or os.environ.get("DEEPSEEK_BASE_URL")
                or "https://api.deepseek.com"
            )
            llm_config = {
                "provider": "deepseek",
                "config": {
                    "model": self.settings.llm_model
                    or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
                    "api_key": api_key or "dummy",
                    "deepseek_base_url": base_url,
                    "temperature": 0.0,
                },
            }
        else:
            # A dummy LLM config is still required by Memory; infer=False never
            # calls it.
            llm_config = {
                "provider": "deepseek",
                "config": {"model": "deepseek-flash", "api_key": "dummy"},
            }

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection,
                    "path": str(self._workdir / collection),
                    "embedding_model_dims": int(self.settings.embedding_dims),
                    "on_disk": False,
                },
            },
            "llm": llm_config,
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "hash-embedding",
                    "embedding_dims": int(self.settings.embedding_dims),
                },
            },
        }
        self._memory = Memory.from_config(config)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add_texts(
        self,
        texts: Iterable[str],
        *,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> list[dict]:
        results: list[dict] = []
        uid = user_id or self.namespace
        for text in texts:
            result = self._memory.add(
                [{"role": "user", "content": str(text)}],
                user_id=uid,
                infer=self.settings.infer,
                metadata=metadata,
            )
            results.append(result if isinstance(result, dict) else {"result": result})
        return results

    def add_assets(
        self,
        assets: Iterable[MemoryAsset],
        *,
        user_id: str | None = None,
    ) -> list[dict]:
        results: list[dict] = []
        uid = user_id or self.namespace
        for asset in assets:
            metadata = {
                "memory_id": asset.id,
                "facet": asset.facet,
                "quality": float(asset.quality),
                "source_class": asset.source_class,
                **{k: v for k, v in asset.metadata.items() if isinstance(v, (str, int, float, bool))},
            }
            result = self._memory.add(
                [{"role": "user", "content": asset.content}],
                user_id=uid,
                infer=self.settings.infer,
                metadata=metadata,
            )
            results.append(result if isinstance(result, dict) else {"result": result})
        return results

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        user_id: str | None = None,
    ) -> list[RetrievedItem]:
        uid = user_id or self.namespace
        raw = self._memory.search(
            query,
            filters={"user_id": uid},
            top_k=max(1, int(top_k)),
        )
        rows = raw.get("results", []) if isinstance(raw, dict) else raw
        items: list[RetrievedItem] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata") or {}
            items.append(
                RetrievedItem(
                    text=str(row.get("memory", "")),
                    score=float(row.get("score", 0.0) or 0.0),
                    memory_id=metadata.get("memory_id"),
                    metadata=dict(metadata),
                )
            )
        return items

    def close(self) -> None:
        try:
            self._memory.close()
        except Exception:
            pass


class Mem0MemoryBank:
    """One Mem0 namespace per agent, or one shared namespace."""

    def __init__(
        self,
        agent_ids: list[str],
        *,
        shared: bool = False,
        settings: Mem0Settings | None = None,
        workdir: str | Path | None = None,
    ):
        self.agent_ids = list(agent_ids)
        self.shared = shared
        self.settings = settings or Mem0Settings()
        self._workdir = Path(workdir) if workdir else Path(
            tempfile.mkdtemp(prefix="agora_mem0_bank_")
        )
        self._stores: dict[str, OfficialMem0Store] = {}
        if shared:
            self._stores["__shared__"] = OfficialMem0Store(
                "team", settings=self.settings, workdir=self._workdir
            )
        else:
            for agent_id in self.agent_ids:
                self._stores[agent_id] = OfficialMem0Store(
                    agent_id, settings=self.settings, workdir=self._workdir
                )

    def ingest(self, assets_by_agent: dict[str, list[MemoryAsset]]) -> None:
        if self.shared:
            seen: set[str] = set()
            merged: list[MemoryAsset] = []
            for assets in assets_by_agent.values():
                for asset in assets:
                    if asset.id not in seen:
                        seen.add(asset.id)
                        merged.append(asset)
            self._stores["__shared__"].add_assets(merged, user_id="team")
            return
        for agent_id, assets in assets_by_agent.items():
            store = self._stores.get(agent_id)
            if store is not None:
                store.add_assets(assets, user_id=agent_id)

    def search(self, agent_id: str, query: str, *, top_k: int = 3) -> list[RetrievedItem]:
        if self.shared:
            return self._stores["__shared__"].search(query, top_k=top_k, user_id="team")
        store = self._stores.get(agent_id)
        if store is None:
            return []
        return store.search(query, top_k=top_k, user_id=agent_id)

    def close(self) -> None:
        for store in self._stores.values():
            store.close()


__all__ = ["Mem0Settings", "OfficialMem0Store", "Mem0MemoryBank"]
