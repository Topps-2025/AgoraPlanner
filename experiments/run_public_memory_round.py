"""One-round public memory benchmark across named baselines.

Datasets
--------
- LoCoMo (snap-research/locomo): long conversations with QA and turn-level
  evidence ids.
- LongMemEval (xiaowu0162/longmemeval-cleaned): timestamped chat sessions with
  session-level evidence ids.

Baselines
---------
- Raw-RAG: BM25 over all memories.
- private / shared_all / selective: allocation-style retrieval scopes.
- cosine_shared: deterministic hash-embedding cosine retrieval.
- Mem0 official (``mem0ai``, infer=False, local Qdrant, hash embeddings).
- Faithful reimplementations: MemGPT, RAPTOR, GraphRAG, CAMA, GovMem,
  RCR-Router, MemPrice (explicitly labelled ``official=false``).
- oracle: evidence memories first (upper bound for retrieval).

Metrics
-------
Retrieval: recall@k, context precision, MRR, nDCG@k.
Answer: exact match, token F1, substring containment.
Cost: prompt/completion tokens, tokens/query, latency/query, success per 1k
tokens.  The reader can be Qwen3-8B (local vLLM), DeepSeek flash, or GLM flash.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.embeddings import CosineRetriever, HashEmbedder  # noqa: E402
from agora_planner.memory import MemoryAsset  # noqa: E402
from agora_planner.memory_systems import (  # noqa: E402
    CAMAReimpl,
    GovMemReimpl,
    GraphRAGReimpl,
    MemGPTReimpl,
    MemPriceReimpl,
    RAPTORReimpl,
    RCRRouterReimpl,
)

try:  # BM25 is optional; fall back to a simple token-overlap scorer.
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover
    BM25Okapi = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


@dataclass
class PublicQuery:
    id: str
    question: str
    answer: str
    evidence_ids: frozenset[str]
    agent_id: str
    category: str = ""
    dataset: str = ""


@dataclass
class PublicWorld:
    name: str
    memories: list[MemoryAsset]
    queries: list[PublicQuery]
    notes: dict[str, Any] = field(default_factory=dict)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def load_locomo(
    path: str | Path, *, max_conversations: int = 2, max_queries: int = 30
) -> PublicWorld:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    memories: list[MemoryAsset] = []
    queries: list[PublicQuery] = []
    for conv in data[:max_conversations]:
        sample_id = str(conv.get("sample_id", "conv"))
        conversation = conv.get("conversation", {})
        speakers = [
            conversation.get("speaker_a", "speaker_a"),
            conversation.get("speaker_b", "speaker_b"),
        ]
        session_keys = sorted(
            [key for key in conversation if re.fullmatch(r"session_\d+", key)],
            key=lambda key: int(key.split("_")[1]),
        )
        turn_index = 0
        for session_key in session_keys:
            date = conversation.get(f"{session_key}_date_time", "")
            for turn in conversation.get(session_key, []):
                dia_id = str(turn.get("dia_id") or f"{session_key}:{turn_index}")
                memories.append(
                    MemoryAsset(
                        id=f"{sample_id}:{dia_id}",
                        content=f"[{date}] {turn.get('speaker', speakers[0])}: {turn.get('text', '')}",
                        facet=session_key,
                        source_class="conversation",
                        quality=0.8,
                        scope="public",
                        version=0,
                        owner=str(turn.get("speaker", speakers[0])),
                        production_cost=0.1,
                        created_round=turn_index,
                        correlation_risk=0.1,
                        metadata={"session_id": session_key, "dia_id": dia_id, "date": date},
                    )
                )
                turn_index += 1
        for idx, qa in enumerate(conv.get("qa", [])[:max_queries]):
            evidence = frozenset(
                f"{sample_id}:{item}" for item in (qa.get("evidence") or [])
            )
            queries.append(
                PublicQuery(
                    id=f"{sample_id}_q{idx}",
                    question=str(qa.get("question", "")),
                    answer=str(qa.get("answer", "")),
                    evidence_ids=evidence,
                    agent_id=speakers[idx % 2],
                    category=str(qa.get("category", "")),
                    dataset="locomo",
                )
            )
    return PublicWorld("locomo", memories, queries, {"source": str(path)})


def load_longmemeval(
    path: str | Path, *, max_instances: int = 30
) -> PublicWorld:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    memories: list[MemoryAsset] = []
    queries: list[PublicQuery] = []
    for inst in data[:max_instances]:
        qid = str(inst.get("question_id", "q"))
        question = str(inst.get("question", ""))
        answer = str(inst.get("answer", ""))
        evidence_sessions = {str(x) for x in (inst.get("answer_session_ids") or [])}
        sessions = inst.get("haystack_sessions") or []
        dates = inst.get("haystack_dates") or []
        session_ids = inst.get("haystack_session_ids") or [f"{qid}_s{i}" for i in range(len(sessions))]
        for si, session in enumerate(sessions):
            session_id = str(session_ids[si]) if si < len(session_ids) else f"{qid}_s{si}"
            date = str(dates[si]) if si < len(dates) else ""
            for ti, turn in enumerate(session):
                content = f"[{date}] {turn.get('role', 'user')}: {turn.get('content', '')}"
                memories.append(
                    MemoryAsset(
                        id=f"{qid}:{session_id}:{ti}",
                        content=content,
                        facet=session_id,
                        source_class="conversation",
                        quality=0.8,
                        scope="public",
                        version=0,
                        owner=str(turn.get("role", "user")),
                        production_cost=0.1,
                        created_round=si * 100 + ti,
                        correlation_risk=0.1,
                        metadata={
                            "session_id": session_id,
                            "date": date,
                            "has_answer": bool(turn.get("has_answer", False)),
                        },
                    )
                )
        queries.append(
            PublicQuery(
                id=qid,
                question=question,
                answer=answer,
                evidence_ids=frozenset(evidence_sessions),
                agent_id="user",
                category=str(inst.get("question_type", "")),
                dataset="longmemeval",
            )
        )
    return PublicWorld("longmemeval", memories, queries, {"source": str(path)})


def load_memoryagentbench(
    path: str | Path, *, max_rows: int = 2, max_queries: int = 5
) -> PublicWorld:
    """MemoryAgentBench parquet rows: a fact list + questions/answers.

    The released parquet does not include per-question evidence ids, so
    retrieval recall is reported as N/A and only QA/token metrics are used.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    memories: list[MemoryAsset] = []
    queries: list[PublicQuery] = []
    for row_index, row in frame.head(max_rows).iterrows():
        lines = [line.strip() for line in str(row["context"]).splitlines() if line.strip()]
        for line_index, line in enumerate(lines):
            content = re.sub(r"^\d+\.\s*", "", line)
            memories.append(
                MemoryAsset(
                    id=f"mab_{row_index}_{line_index}",
                    content=content,
                    facet=f"row{row_index}",
                    source_class="fact",
                    quality=0.8,
                    scope="public",
                    version=0,
                    owner="agent",
                    production_cost=0.1,
                    created_round=line_index,
                    correlation_risk=0.1,
                    metadata={"row": int(row_index)},
                )
            )
        questions = list(row["questions"])[:max_queries]
        answers = list(row["answers"])[:max_queries]
        for query_index, (question, answer) in enumerate(zip(questions, answers)):
            if hasattr(answer, "tolist"):
                answer = answer.tolist()
            if isinstance(answer, (list, tuple)):
                answer = answer[0] if answer else ""
            queries.append(
                PublicQuery(
                    id=f"mab_{row_index}_{query_index}",
                    question=str(question),
                    answer=str(answer),
                    evidence_ids=frozenset(),
                    agent_id="agent",
                    category="conflict_resolution",
                    dataset="memoryagentbench",
                )
            )
    return PublicWorld("memoryagentbench", memories, queries, {"source": str(path), "evidence": "not released"})


# ----------------------------------------------------------------------
# Retrieval baselines
# ----------------------------------------------------------------------


def _bm25_scores(memories: Sequence[MemoryAsset], query: str) -> list[float]:
    if not memories:
        return []
    if BM25Okapi is not None:
        corpus = [_tokenize(memory.content) for memory in memories]
        bm25 = BM25Okapi(corpus)
        return [float(x) for x in bm25.get_scores(_tokenize(query))]
    query_tokens = set(_tokenize(query))
    return [
        float(len(query_tokens & set(_tokenize(memory.content))))
        for memory in memories
    ]


def _rank(
    memories: Sequence[MemoryAsset],
    query: str,
    *,
    top_k: int,
    per_session_cap: int | None = None,
) -> list[MemoryAsset]:
    scores = _bm25_scores(memories, query)
    order = sorted(range(len(memories)), key=lambda i: -scores[i])
    out: list[MemoryAsset] = []
    seen_sessions: dict[str, int] = {}
    for idx in order:
        memory = memories[int(idx)]
        if per_session_cap is not None:
            session = str(memory.metadata.get("session_id", memory.facet))
            if seen_sessions.get(session, 0) >= per_session_cap:
                continue
            seen_sessions[session] = seen_sessions.get(session, 0) + 1
        out.append(memory)
        if len(out) >= top_k:
            break
    return out


_BM25_INDEX_CACHE: dict[tuple[int, str, str], Any] = {}


def _retrieve_bm25(
    world: PublicWorld,
    query: PublicQuery,
    *,
    top_k: int,
    scope: str,
) -> list[MemoryAsset]:
    if scope == "private":
        pool = [m for m in world.memories if m.owner == query.agent_id]
    else:
        pool = list(world.memories)
    if BM25Okapi is not None:
        # Reuse the BM25 index across queries and baselines; rebuilding it per
        # call is prohibitive on 50k--100k-line MemoryAgentBench rows.
        key = (id(world), scope, query.agent_id if scope == "private" else "")
        index = _BM25_INDEX_CACHE.get(key)
        if index is None:
            index = BM25Okapi([_tokenize(memory.content) for memory in pool])
            _BM25_INDEX_CACHE[key] = index
        scores = [float(x) for x in index.get_scores(_tokenize(query.question))]
        order = sorted(range(len(pool)), key=lambda i: -scores[i])
    else:
        fallback_scores = _bm25_scores(pool, query.question)
        order = sorted(range(len(pool)), key=lambda i: -fallback_scores[i])
    out: list[MemoryAsset] = []
    seen_sessions: dict[str, int] = {}
    for idx in order:
        memory = pool[int(idx)]
        if scope == "selective":
            session = str(memory.metadata.get("session_id", memory.facet))
            if seen_sessions.get(session, 0) >= 1:
                continue
            seen_sessions[session] = seen_sessions.get(session, 0) + 1
        out.append(memory)
        if len(out) >= top_k:
            break
    return out


def _retrieve_cosine(world: PublicWorld, query: PublicQuery, *, top_k: int) -> list[MemoryAsset]:
    retriever = CosineRetriever(HashEmbedder())
    retriever.extend([(m.id, m.content) for m in world.memories])
    by_id = {m.id: m for m in world.memories}
    return [
        by_id[item.memory_id]
        for item in retriever.search(query.question, top_k=top_k)
        if item.memory_id in by_id
    ]


def _retrieve_oracle(world: PublicWorld, query: PublicQuery, *, top_k: int) -> list[MemoryAsset]:
    by_session: dict[str, list[MemoryAsset]] = {}
    for memory in world.memories:
        by_session.setdefault(str(memory.metadata.get("session_id", memory.facet)), []).append(memory)
    out: list[MemoryAsset] = []
    for evidence in query.evidence_ids:
        # LoCoMo evidence is a memory id; LongMemEval evidence is a session id.
        direct = next((m for m in world.memories if m.id == evidence), None)
        if direct is not None:
            out.append(direct)
        else:
            session_memories = by_session.get(evidence, [])
            # Prefer a turn explicitly marked as containing the answer.
            session_memories = sorted(
                session_memories,
                key=lambda memory: (not bool(memory.metadata.get("has_answer", False))),
            )
            out.extend(session_memories[:1])
    seen = {m.id for m in out}
    if len(out) < top_k:
        for memory in _rank(world.memories, query.question, top_k=top_k * 2):
            if memory.id not in seen:
                out.append(memory)
                seen.add(memory.id)
            if len(out) >= top_k:
                break
    return out[:top_k]


class _ReimplBank:
    def __init__(self, system: Any):
        self.system = system

    def ingest(self, assets: Sequence[MemoryAsset]) -> None:
        self.system.ingest(assets)

    def search(self, agent_id: str, query: str, top_k: int) -> list[MemoryAsset]:
        items = self.system.search(agent_id, query, top_k=top_k)
        return [self.system.assets[item.memory_id] for item in items if item.memory_id in self.system.assets]

    def close(self) -> None:
        self.system.close()


_DENSE_MODEL = None
_DENSE_TOKENIZER = None
_DENSE_DEVICE = None


def _load_dense_model():
    global _DENSE_MODEL, _DENSE_TOKENIZER, _DENSE_DEVICE
    if _DENSE_MODEL is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = os.environ.get(
            "AGORA_EMBED_MODEL", "/root/autodl-tmp/models/bge-small-en-v1.5"
        )
        _DENSE_TOKENIZER = AutoTokenizer.from_pretrained(path)
        _DENSE_MODEL = AutoModel.from_pretrained(path)
        _DENSE_MODEL.eval()
        _DENSE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _DENSE_MODEL.to(_DENSE_DEVICE)
    return _DENSE_MODEL, _DENSE_TOKENIZER, _DENSE_DEVICE


def _encode_dense(texts: Sequence[str], batch_size: int = 64):
    import torch

    model, tokenizer, device = _load_dense_model()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    return torch.cat(chunks) if chunks else torch.zeros((0, 1))


class _DenseBank:
    """Dense bi-encoder retrieval over the same memory pool."""

    def __init__(self, assets: Sequence[MemoryAsset]):
        import torch

        self.assets = list(assets)
        self.matrix = _encode_dense([asset.content[:2400] for asset in self.assets])
        self.device = torch.device("cpu")

    def search(self, agent_id: str, query: str, top_k: int) -> list[MemoryAsset]:
        import torch

        query_vec = _encode_dense([query])[0]
        scores = self.matrix @ query_vec
        count = min(top_k, len(self.assets))
        indices = torch.topk(scores, count).indices.tolist()
        return [self.assets[int(index)] for index in indices]

    def close(self) -> None:
        return None


class _Mem0Bank:
    def __init__(self, agent_ids: Sequence[str], *, shared: bool, workdir: str):
        # The Qdrant store tries to download the fastembed Qdrant/bm25 model
        # during collection creation.  On an offline box each failed attempt
        # costs ~36s, so disable the sparse encoder before constructing Memory;
        # dense hash retrieval is the intended backend.
        try:
            from mem0.vector_stores.qdrant import Qdrant

            Qdrant._get_bm25_encoder = lambda self: None
        except Exception:
            pass

        from agora_planner.mem0_adapter import Mem0MemoryBank, Mem0Settings

        self.bank = Mem0MemoryBank(
            list(agent_ids),
            shared=shared,
            settings=Mem0Settings(infer=False),
            workdir=workdir,
        )

    def ingest(self, assets_by_agent: dict[str, list[MemoryAsset]]) -> None:
        self.bank.ingest(assets_by_agent)

    def search(self, agent_id: str, query: str, top_k: int) -> list[MemoryAsset]:
        items = self.bank.search(agent_id, query, top_k=top_k)
        # Mem0 returns text, not necessarily memory ids; map by exact content.
        by_text = {asset.content: asset for asset in self._assets}
        return [by_text[item.text] for item in items if item.text in by_text]

    def close(self) -> None:
        self.bank.close()


# ----------------------------------------------------------------------
# Reader
# ----------------------------------------------------------------------


class Reader:
    def __init__(
        self,
        backend: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 160,
    ):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        self.backend = backend
        self.model = model or {
            "qwen": "qwen3-8b",
            "deepseek": "deepseek-chat",
            "glm": "glm-4-flash",
        }[backend]
        self.base_url = base_url or {
            "qwen": "http://127.0.0.1:8000/v1",
            "deepseek": "https://api.deepseek.com",
            "glm": "https://open.bigmodel.cn/api/paas/v4",
        }[backend]
        self.api_key = (
            api_key
            or (
                os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPU_API_KEY")
                if self.backend == "glm"
                else os.environ.get("DEEPSEEK_API_KEY")
            )
            or "EMPTY"
        )
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=120)

    def answer(self, question: str, passages: Sequence[MemoryAsset]) -> tuple[str, dict[str, int]]:
        context = "\n".join(
            f"[{idx + 1}] {memory.content[:2400]}" for idx, memory in enumerate(passages)
        )
        prompt = (
            "Answer the question using only the retrieved memory passages. "
            "Be concise. If the answer is not present, answer \"unknown\".\n\n"
            f"Question: {question}\n\nPassages:\n{context}\n\n"
            'Return JSON: {"answer": "..."}'
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if self.backend == "qwen":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        # Fresh client/connection per call: long reader matrices can otherwise
        # block forever on a half-closed keep-alive socket.
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=120)
        try:
            response = client.chat.completions.create(**kwargs)
        finally:
            client.close()
        content = (response.choices[0].message.content or "").strip()
        answer = _extract_answer(content)
        usage = getattr(response, "usage", None)
        tokens = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }
        return answer, tokens


def _extract_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            answer = payload.get("answer")
            if answer is not None:
                return str(answer).strip()
        except Exception:
            pass
    return text


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def _normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, gold: str) -> float:
    gold_norm = _normalize(gold)
    return 1.0 if gold_norm and gold_norm in _normalize(prediction) else 0.0


def _evidence_key(memory: MemoryAsset, query: PublicQuery) -> str | None:
    if memory.id in query.evidence_ids:
        return memory.id
    session = str(memory.metadata.get("session_id", memory.facet))
    if session in query.evidence_ids:
        return session
    return None


def _evidence_hits(retrieved: Sequence[MemoryAsset], query: PublicQuery) -> list[float]:
    hits: list[float] = []
    seen: set[str] = set()
    for memory in retrieved:
        key = _evidence_key(memory, query)
        if key is not None and key not in seen:
            seen.add(key)
            hits.append(1.0)
        else:
            hits.append(0.0)
    return hits


def retrieval_metrics(retrieved: Sequence[MemoryAsset], query: PublicQuery, k: int) -> dict[str, float]:
    if not query.evidence_ids:
        return {f"recall@{k}": float("nan"), f"context_precision@{k}": float("nan"), "mrr": float("nan"), f"ndcg@{k}": float("nan")}
    hits = _evidence_hits(retrieved, query)[:k]
    n_evidence = max(1, len(query.evidence_ids))
    recall = min(1.0, sum(hits) / n_evidence)
    precision = sum(hits) / max(1, len(hits))
    mrr = 0.0
    for idx, hit in enumerate(hits):
        if hit > 0:
            mrr = 1.0 / (idx + 1)
            break
    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(len(hits), int(n_evidence))))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {
        f"recall@{k}": recall,
        f"context_precision@{k}": precision,
        "mrr": mrr,
        f"ndcg@{k}": ndcg,
    }


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


def _asset_dict(world: PublicWorld) -> dict[str, MemoryAsset]:
    return {memory.id: memory for memory in world.memories}


def _baseline_names() -> list[str]:
    return [
        "raw_rag",
        "private",
        "selective",
        "cosine_shared",
        "dense_shared",
        "oracle",
        "memgpt_reimpl",
        "raptor_reimpl",
        "graphrag_reimpl",
        "cama_reimpl",
        "govmem_reimpl",
        "rcr_reimpl",
        "memprice_reimpl",
        "mem0_shared",
    ]


def _make_reimpl(name: str):
    if name == "memgpt_reimpl":
        return MemGPTReimpl(core_slots=2, recall_slots=4)
    if name == "raptor_reimpl":
        return RAPTORReimpl(branch_facts=4, use_quality=True)
    if name == "graphrag_reimpl":
        return GraphRAGReimpl(expand_neighbors=3, use_quality=True)
    if name == "cama_reimpl":
        return CAMAReimpl(candidate_k=12, use_quality=True)
    if name == "govmem_reimpl":
        return GovMemReimpl(promote_support=0.8, reject_quality=0.3)
    if name == "rcr_reimpl":
        return RCRRouterReimpl(token_budget=96, use_quality=True)
    if name == "memprice_reimpl":
        return MemPriceReimpl(target_tokens=96, use_quality=True)
    raise KeyError(name)


def run_world(
    world: PublicWorld,
    *,
    baselines: Sequence[str],
    reader: Reader,
    top_k: int,
    output_dir: Path,
) -> dict[str, Any]:
    by_id = _asset_dict(world)
    agent_ids = sorted({m.owner for m in world.memories}) or ["agent"]
    results: dict[str, Any] = {}
    per_query: list[dict[str, Any]] = []

    banks: dict[str, Any] = {}
    for name in baselines:
        if name.endswith("_reimpl"):
            try:
                bank = _ReimplBank(_make_reimpl(name))
                bank.ingest(world.memories)
                banks[name] = bank
            except Exception as exc:
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}
        elif name == "mem0_shared":
            try:
                import tempfile

                bank = _Mem0Bank(agent_ids, shared=True, workdir=tempfile.mkdtemp(prefix="mem0_round_"))
                bank._assets = list(world.memories)
                bank.ingest({agent: list(world.memories) for agent in agent_ids})
                banks[name] = bank
            except Exception as exc:
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}
        elif name == "dense_shared":
            try:
                banks[name] = _DenseBank(world.memories)
            except Exception as exc:
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}

    for name in baselines:
        if name in results and "error" in results[name]:
            continue
        rows: list[dict[str, Any]] = []
        start_all = time.perf_counter()
        for query in world.queries:
            query_start = time.perf_counter()
            try:
                if name == "raw_rag":
                    retrieved = _retrieve_bm25(world, query, top_k=top_k, scope="shared")
                elif name == "private":
                    retrieved = _retrieve_bm25(world, query, top_k=top_k, scope="private")
                elif name == "selective":
                    retrieved = _retrieve_bm25(world, query, top_k=top_k, scope="selective")
                elif name == "cosine_shared":
                    retrieved = _retrieve_cosine(world, query, top_k=top_k)
                elif name == "oracle":
                    retrieved = _retrieve_oracle(world, query, top_k=top_k)
                elif name in banks:
                    retrieved = banks[name].search(query.agent_id, query.question, top_k)
                else:
                    raise KeyError(name)
                answer, tokens = reader.answer(query.question, retrieved)
                metrics = retrieval_metrics(retrieved, query, top_k)
                metrics.update(
                    {
                        "answer_exact": exact_match(answer, query.answer),
                        "answer_f1": token_f1(answer, query.answer),
                        "answer_contains": contains_answer(answer, query.answer),
                        "prompt_tokens": tokens["prompt_tokens"],
                        "completion_tokens": tokens["completion_tokens"],
                        "total_tokens": tokens["prompt_tokens"] + tokens["completion_tokens"],
                    }
                )
                rows.append(
                    {
                        "query_id": query.id,
                        "dataset": query.dataset,
                        "baseline": name,
                        "question": query.question,
                        "gold": query.answer,
                        "prediction": answer,
                        "category": query.category,
                        "latency_ms": (time.perf_counter() - query_start) * 1000.0,
                        "retrieved_ids": [m.id for m in retrieved],
                        **metrics,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "query_id": query.id,
                        "dataset": query.dataset,
                        "baseline": name,
                        "question": query.question,
                        "gold": query.answer,
                        "category": query.category,
                        "latency_ms": (time.perf_counter() - query_start) * 1000.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        elapsed = time.perf_counter() - start_all
        valid = [row for row in rows if "error" not in row]
        n = max(1, len(valid))
        total_tokens = sum(row.get("total_tokens", 0) for row in valid)
        results[name] = {
            "official": name == "mem0_shared",
            "n_queries": len(valid),
            "recall@k": sum(row.get(f"recall@{top_k}", 0.0) for row in valid) / n,
            "context_precision@k": sum(row.get(f"context_precision@{top_k}", 0.0) for row in valid) / n,
            "mrr": sum(row.get("mrr", 0.0) for row in valid) / n,
            "ndcg@k": sum(row.get(f"ndcg@{top_k}", 0.0) for row in valid) / n,
            "answer_exact": sum(row.get("answer_exact", 0.0) for row in valid) / n,
            "answer_f1": sum(row.get("answer_f1", 0.0) for row in valid) / n,
            "answer_contains": sum(row.get("answer_contains", 0.0) for row in valid) / n,
            "tokens_per_query": total_tokens / n,
            "latency_ms_per_query": elapsed * 1000.0 / n,
            "success_per_1k_tokens": (
                (sum(row.get("answer_exact", 0.0) for row in valid) / n)
                / max(1e-9, total_tokens / 1000.0 / n)
            ),
            "errors": len(rows) - len(valid),
        }
        per_query.extend(rows)
    for bank in banks.values():
        try:
            bank.close()
        except Exception:
            pass
    (output_dir / f"per_query_{world.name}.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in per_query),
        encoding="utf-8",
    )
    return {"n_memories": len(world.memories), "n_queries": len(world.queries), "baselines": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="One-round public memory benchmark")
    parser.add_argument("--locomo", default="/root/autodl-tmp/agora_planner_cli/data/LoCoMo/repo/data/locomo10.json")
    parser.add_argument("--longmemeval", default="/root/autodl-tmp/agora_planner_cli/data/LongMemEval/data/longmemeval_oracle.json")
    parser.add_argument("--memoryagentbench", default="/root/autodl-tmp/agora_planner_cli/data/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet")
    parser.add_argument("--mab-rows", type=int, default=2)
    parser.add_argument("--mab-queries", type=int, default=5)
    parser.add_argument("--datasets", default="locomo,longmemeval")
    parser.add_argument("--max-queries", type=int, default=30)
    parser.add_argument("--max-conversations", type=int, default=2)
    parser.add_argument("--max-instances", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reader", default="qwen", choices=("qwen", "deepseek", "glm"))
    parser.add_argument("--reader-model", default=None)
    parser.add_argument("--reader-max-tokens", type=int, default=160)
    parser.add_argument("--baselines", default="all")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "public_memory_round"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = Reader(args.reader, model=args.reader_model, max_tokens=args.reader_max_tokens)
    baselines = _baseline_names() if args.baselines == "all" else [x for x in args.baselines.split(",") if x]
    worlds: list[PublicWorld] = []
    if "locomo" in args.datasets and Path(args.locomo).exists():
        worlds.append(
            load_locomo(
                args.locomo,
                max_conversations=args.max_conversations,
                max_queries=args.max_queries,
            )
        )
    if "longmemeval" in args.datasets and Path(args.longmemeval).exists():
        try:
            worlds.append(load_longmemeval(args.longmemeval, max_instances=args.max_instances))
        except Exception as exc:
            print(f"[warn] LongMemEval not loaded: {exc}", file=sys.stderr)
    if "memoryagentbench" in args.datasets and Path(args.memoryagentbench).exists():
        try:
            worlds.append(
                load_memoryagentbench(
                    args.memoryagentbench,
                    max_rows=args.mab_rows,
                    max_queries=args.mab_queries,
                )
            )
        except Exception as exc:
            print(f"[warn] MemoryAgentBench not loaded: {exc}", file=sys.stderr)
    summary: dict[str, Any] = {
        "config": vars(args),
        "reader": {"backend": args.reader, "model": reader.model, "base_url": reader.base_url},
        "datasets": {},
    }
    for world in worlds:
        print(f"[run] {world.name}: {len(world.memories)} memories, {len(world.queries)} queries", flush=True)
        summary["datasets"][world.name] = run_world(
            world,
            baselines=baselines,
            reader=reader,
            top_k=args.top_k,
            output_dir=output_dir,
        )
        print(f"[done] {world.name}", flush=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# One-round public memory benchmark", ""]
    for dataset, payload in summary["datasets"].items():
        lines += [f"## {dataset}", "", "| baseline | official | n | recall@k | nDCG@k | exact | F1 | tokens/q | latency ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, row in payload["baselines"].items():
            if "error" in row:
                lines.append(f"| {name} |  |  |  |  |  |  |  | error: {row['error'][:60]} |")
                continue
            lines.append(
                f"| {name} | {row['official']} | {row['n_queries']} | {row['recall@k']:.3f} | "
                f"{row['ndcg@k']:.3f} | {row['answer_exact']:.3f} | {row['answer_f1']:.3f} | "
                f"{row['tokens_per_query']:.0f} | {row['latency_ms_per_query']:.0f} |"
            )
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
