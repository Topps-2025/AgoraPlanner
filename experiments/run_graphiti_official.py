"""Run official Graphiti (temporal knowledge graph) on the text benchmark.

Uses Graphiti's public ``Graphiti`` / ``KuzuDriver`` / ``OpenAIGenericClient``
APIs.  The only adaptations are a deterministic hash embedder and DeepSeek as
the OpenAI-compatible chat backend (Graphiti explicitly supports providers
without native json_schema via ``structured_output_mode='json_object'``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPHITI_REPO = Path(os.environ.get("GRAPHITI_REPO", str(ROOT / "baselines" / "graphiti")))
os.environ.setdefault("EMBEDDING_DIM", "384")
for path in [str(GRAPHITI_REPO), str(ROOT / "src")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from openai import OpenAI  # noqa: E402

from agora_planner.embeddings import HashEmbedder  # noqa: E402
from agora_planner.text_benchmark import (  # noqa: E402
    TextBenchmarkConfig,
    TextMemoryBenchmark,
    extract_code,
)

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.cross_encoder.client import CrossEncoderClient  # noqa: E402
from graphiti_core.driver.kuzu_driver import KuzuDriver  # noqa: E402
from graphiti_core.embedder.client import EmbedderClient  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402


class BGEEmbedder:
    def __init__(self, path: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.path = path or os.environ.get(
            "AGORA_EMBED_MODEL", "/root/autodl-tmp/models/bge-small-en-v1.5"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModel.from_pretrained(self.path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def embed(self, text):
        import torch

        encoded = self.tokenizer(
            [str(text)], padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)[0].cpu()


class HashEmbedderClient(EmbedderClient):
    def __init__(self, dim: int = 384):
        if os.environ.get("GRAPHITI_EMBEDDER", "hash").lower() == "bge":
            self.embedder = BGEEmbedder()
        else:
            self.embedder = HashEmbedder(dim=dim)

    async def create(self, input_data):
        if isinstance(input_data, str):
            return self.embedder.embed(input_data).tolist()
        if isinstance(input_data, list):
            if input_data and isinstance(input_data[0], str):
                return self.embedder.embed(input_data[0]).tolist()
            return self.embedder.embed(str(input_data)).tolist()
        return self.embedder.embed(str(input_data)).tolist()

    async def create_batch(self, input_data_list):
        return [self.embedder.embed(text).tolist() for text in input_data_list]


class NoopReranker(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]):
        return [(text, 0.0) for text in passages]


class Reader:
    def __init__(self, model: str = "deepseek-flash"):
        self.client = OpenAI(
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            timeout=120,
        )
        self.model = model

    def answer(self, question: str, facts: list[str]) -> tuple[str, int, int, int]:
        context = "\n".join(f"[{idx + 1}] {fact}" for idx, fact in enumerate(facts))
        prompt = (
            "Answer the question using only the retrieved memory facts. Be concise. "
            "If the answer is not present, answer \"unknown\".\n\n"
            f"Question: {question}\n\nFacts:\n{context}\n\n"
            'Return JSON: {"answer": "..."}'
        )
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return (
            text,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            int((time.time() - start) * 1000),
        )


def make_bench(seed: int) -> TextMemoryBenchmark:
    return TextMemoryBenchmark(
        TextBenchmarkConfig(
            seed=seed,
            n_agents=3,
            n_facets=16,
            facts_per_agent=4,
            n_public_facts=4,
            decoy_fraction=0.40,
            decoy_style="exact",
            n_train_tasks=24,
            n_eval_tasks=24,
            max_memories_per_agent=3,
            solver="beam",
        )
    )


async def run(seed: int, max_facts: int, max_queries: int, output_dir: Path) -> dict:
    bench = make_bench(seed)
    assets = list(bench.memories)[:max_facts]
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / f"graphiti_seed{seed}.kuzu"
    if db_path.exists():
        db_path.unlink()
    driver = KuzuDriver(db=str(db_path))
    # Graphiti's request-scope helper expects a Neo4j-style ``_database``
    # attribute; Kuzu has a single embedded database, so an empty value is
    # correct here.
    driver._database = None  # type: ignore[attr-defined]
    # KuzuDriver.build_indices_and_constraints() is a no-op in this release,
    # so create the FTS indices that its search path expects.
    import kuzu

    conn = kuzu.Connection(driver.db)
    for query in (
        "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
        "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
        "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
        "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
    ):
        try:
            conn.execute(query)
        except Exception as exc:  # already-created index
            print("FTS index warning:", exc, flush=True)
    conn.close()
    llm = OpenAIGenericClient(
        config=LLMConfig(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
            small_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            max_tokens=2048,
        ),
        max_tokens=2048,
        structured_output_mode="json_object",
    )
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm,
        embedder=HashEmbedderClient(),
        cross_encoder=NoopReranker(),
    )
    await graphiti.build_indices_and_constraints()
    ingest_start = time.time()
    for asset in assets:
        await graphiti.add_episode(
            name=asset.id,
            episode_body=asset.content,
            source_description="controlled memory fact",
            reference_time=datetime.now(timezone.utc),
            source=EpisodeType.text,
            group_id="text_benchmark",
        )
    ingest_s = time.time() - ingest_start
    reader = Reader()
    rows = []
    for task in bench.eval_tasks[:max_queries]:
        query = bench.queries[task.id]
        edges = await graphiti.search(query.question, num_results=5, group_ids=["text_benchmark"])
        facts = [str(getattr(edge, "fact", "") or "") for edge in edges]
        answer, prompt_tokens, completion_tokens, latency_ms = reader.answer(query.question, facts)
        parsed = extract_code(answer)
        rows.append(
            {
                "query_id": query.id,
                "gold": query.gold_answer,
                "parsed": parsed,
                "correct": parsed == query.gold_answer,
                "gold_in_context": any(query.gold_answer in fact for fact in facts),
                "facts": facts,
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
            }
        )
        print(query.id, query.gold_answer, parsed, rows[-1]["correct"], flush=True)
    payload = {
        "seed": seed,
        "engine": "official graphiti-core + KuzuDriver",
        "n_facts": len(assets),
        "n_queries": len(rows),
        "ingest_s": ingest_s,
        "rows": rows,
        "accuracy": sum(row["correct"] for row in rows) / max(1, len(rows)),
        "context_recall": sum(row["gold_in_context"] for row in rows) / max(1, len(rows)),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
    }
    (output_dir / f"results_seed{seed}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print("ACC", payload["accuracy"], "CTX", payload["context_recall"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-facts", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "graphiti_official"))
    args = parser.parse_args()
    asyncio.run(run(args.seed, args.max_facts, args.max_queries, Path(args.output_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
