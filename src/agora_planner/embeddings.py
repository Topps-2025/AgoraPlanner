"""Deterministic hash embeddings and a small cosine retriever.

The official memory systems normally use a neural embedding model.  In this
offline environment the HuggingFace hub is unreachable, so the adapters share a
deterministic hashed bag-of-words embedder.  This keeps the comparison between
systems about their *memory management* rather than about embedding quality.
Every adapter (Mem0, MemGPT reimplementation, RAPTOR reimplementation,
GraphRAG reimplementation) uses the same embedder.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")
_CODE_RE = re.compile(r"\b[A-Z0-9]{4}\b")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "this",
    "that",
    "these",
    "those",
    "and",
    "or",
    "not",
    "with",
    "from",
    "as",
    "its",
    "it",
    "be",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "may",
    "might",
    "access",
    "code",
}


class HashEmbedder:
    """Hashed bag-of-words embedding with optional character trigrams.

    Access-code-shaped tokens are removed before hashing: the retrieval score
    should depend on the *topic* and *facet*, not on memorising the answer
    token.  This makes the benchmark a memory-selection test rather than a
    lexical shortcut test.
    """

    def __init__(self, dim: int = 384, *, char_ngrams: bool = True):
        self.dim = int(dim)
        self.char_ngrams = bool(char_ngrams)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        # Drop access codes before lower-casing.
        text = _CODE_RE.sub(" ", str(text))
        words = [
            token
            for token in _WORD_RE.findall(text.lower())
            if token not in _STOPWORDS and len(token) > 1
        ]
        return words

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        words = self._tokens(text)
        for token in words:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:12], 16) % self.dim
            sign = 1.0 if int(digest[12:14], 16) % 2 == 0 else -1.0
            vec[idx] += sign
            if self.char_ngrams and len(token) >= 4:
                padded = f"#{token}#"
                for i in range(len(padded) - 2):
                    gram = padded[i : i + 3]
                    gdigest = hashlib.md5(gram.encode("utf-8")).hexdigest()
                    gidx = int(gdigest[:12], 16) % self.dim
                    gsign = 1.0 if int(gdigest[12:14], 16) % 2 == 0 else -1.0
                    vec[gidx] += 0.35 * gsign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self.embed(text) for text in texts])


@dataclass
class RetrievedItem:
    text: str
    score: float
    memory_id: str | None = None
    metadata: dict = field(default_factory=dict)


class CosineRetriever:
    """An in-memory cosine retriever over ``(memory_id, text)`` pairs."""

    def __init__(self, embedder: HashEmbedder | None = None):
        self.embedder = embedder or HashEmbedder()
        self.items: list[tuple[str, str]] = []
        self._matrix: np.ndarray | None = None
        self._dirty = True

    def add(self, memory_id: str, text: str) -> None:
        self.items.append((memory_id, text))
        self._dirty = True

    def extend(self, pairs: list[tuple[str, str]]) -> None:
        self.items.extend(pairs)
        self._dirty = True

    def _ensure_matrix(self) -> None:
        if self._dirty:
            self._matrix = self.embedder.embed_many([text for _, text in self.items])
            self._dirty = False

    def search(self, query: str, top_k: int = 3) -> list[RetrievedItem]:
        if not self.items:
            return []
        self._ensure_matrix()
        q = self.embedder.embed(query)
        scores = self._matrix @ q
        # True and decoy facts can have *identical* embeddings (same template,
        # different access code, and the code is stripped before hashing).  A
        # plain argsort would then depend on set/insertion order and therefore
        # on PYTHONHASHSEED.  Break ties with a stable hash of the memory id so
        # every process sees the same ranking.
        def tie_break(idx: int) -> str:
            memory_id = self.items[int(idx)][0]
            return hashlib.md5(memory_id.encode("utf-8")).hexdigest()

        order = sorted(
            range(len(self.items)),
            key=lambda idx: (-float(scores[idx]), tie_break(idx)),
        )[: max(1, int(top_k))]
        return [
            RetrievedItem(
                text=self.items[int(idx)][1],
                score=float(scores[int(idx)]),
                memory_id=self.items[int(idx)][0],
            )
            for idx in order
        ]

    def __len__(self) -> int:
        return len(self.items)


__all__ = ["HashEmbedder", "RetrievedItem", "CosineRetriever"]
