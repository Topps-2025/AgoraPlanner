"""Run the official MemLens memory pipeline on the text benchmark.

MemLens is a dialogue-memory application with a Milvus vector store.  This
runner keeps its official atomic-memory / hierarchy / Shapley / organizer
pipeline, points its OpenAI-compatible model client at DeepSeek, uses the same
hash embedding endpoint as the other baselines, and replaces the Milvus server
with embedded Milvus Lite (``milvus-lite``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMLENS_REPO = Path(os.environ.get("MEMLENS_REPO", str(ROOT / "baselines" / "MemLens")))
for path in [
    str(MEMLENS_REPO),
    str(ROOT / "src"),
    "/root/autodl-tmp/vendor_memlens",
]:
    if path not in sys.path:
        sys.path.insert(0, path)

from agora_planner.text_benchmark import (  # noqa: E402
    TextBenchmarkConfig,
    TextMemoryBenchmark,
    extract_code,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "memlens_official"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "atomic_memory": {"min_length": 5},
        "shapley": {"sampling_count": 2, "score_threshold": -1, "similarity_metric": "cosine"},
        "embedding": {
            "type": "openai",
            "openai_api_key": "dummy",
            "openai_model": "hash-embedding",
        },
        "storage": {"type": "json", "path": str(output_dir / "memory_storage.json")},
        "vector_db": {
            "host": "embedded",
            "port": 0,
            "collection_name": "memlens_memory",
            "metric_type": "COSINE",
        },
        "llm": {
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "model_name": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
            "timeout": 300,
            "max_retries": 2,
            "small_model_type": "openai",
            "large_model_type": "openai",
            "splitter_model_type": "openai",
        },
    }
    config_path = output_dir / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    from llm_memory_system.core.pipeline import MemoryPipeline

    db_path = output_dir / f"memlens_seed{args.seed}.db"
    if db_path.exists():
        db_path.unlink()

    def _init_vector_store(self):  # noqa: ANN001
        from pymilvus import connections

        connections.connect(alias="default", uri=str(db_path))
        self.vector_store_ready = True

    MemoryPipeline._initialize_vector_store = _init_vector_store

    bench = TextMemoryBenchmark(
        TextBenchmarkConfig(
            seed=args.seed,
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
    pipeline = MemoryPipeline(str(config_path))
    history = [
        {
            "user": asset.content,
            "agent": "",
            "user_id": "text_benchmark",
            "round_id": asset.id,
        }
        for asset in bench.memories
    ]
    first_query = bench.queries[bench.eval_tasks[0].id]
    pipeline.run(history, "Controlled memory facts for a team.", first_query.question)
    pipeline.store_mid_term_memories("text_benchmark", 0.0)

    rows = []
    for task in bench.eval_tasks:
        query = bench.queries[task.id]
        retrieved = pipeline._retrieve_top_memories(query.question, top_k=args.top_k)
        response = pipeline.chat_with_memory(query.question)
        parsed = extract_code(response)
        rows.append(
            {
                "query_id": query.id,
                "gold": query.gold_answer,
                "parsed": parsed,
                "correct": parsed == query.gold_answer,
                "gold_in_context": any(query.gold_answer in item.get("content", "") for item in retrieved),
                "retrieved": [item.get("content", "")[:300] for item in retrieved],
                "response": response[:2000],
            }
        )
        print(query.id, query.gold_answer, parsed, rows[-1]["correct"], flush=True)
    payload = {
        "seed": args.seed,
        "engine": "official MemLens pipeline + embedded Milvus Lite",
        "n_facts": len(list(bench.memories)),
        "rows": rows,
        "accuracy": sum(row["correct"] for row in rows) / max(1, len(rows)),
        "context_recall": sum(row["gold_in_context"] for row in rows) / max(1, len(rows)),
    }
    (output_dir / f"results_seed{args.seed}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("ACC", payload["accuracy"], "CTX", payload["context_recall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
