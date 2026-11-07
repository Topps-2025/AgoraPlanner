"""Run official Microsoft GraphRAG 2.7.2 on the text-memory benchmark.

This uses GraphRAG's public ``build_index`` / ``basic_search`` / ``local_search``
APIs.  The only adaptation is the embedding backend: a local OpenAI-compatible
hash-embedding endpoint (the same deterministic embedder as the other systems),
because neural embedding endpoints are not reachable from the cloud box.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from agora_planner.embeddings import HashEmbedder  # noqa: E402
from agora_planner.text_benchmark import TextBenchmarkConfig, TextMemoryBenchmark  # noqa: E402

CODE_RE = re.compile(r"\b([A-Z0-9]{4})\b")


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class HashEmbeddingServer:
    def __init__(self, port: int = 8765, dim: int = 384):
        self.embedder = HashEmbedder(dim=dim)
        self.port = port
        self.requests = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                inputs = data.get("input", [])
                if isinstance(inputs, str):
                    inputs = [inputs]
                outer.requests += len(inputs)
                payload = {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": outer.embedder.embed(text).tolist(),
                        }
                        for index, text in enumerate(inputs)
                    ],
                    "model": data.get("model", "hash-embedding"),
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                }
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):  # noqa: ANN002
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):  # noqa: ANN002
        self.server.shutdown()


def write_benchmark(workdir: Path, seed: int) -> list[dict]:
    bench = TextMemoryBenchmark(
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
    input_dir = workdir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    for asset in bench.memories:
        (input_dir / f"{asset.id}.txt").write_text(
            f"{asset.content}\n", encoding="utf-8"
        )
    queries = []
    for task in bench.eval_tasks:
        query = bench.queries[task.id]
        queries.append(
            {
                "id": query.id,
                "agent": query.agent_id,
                "question": query.question,
                "gold": query.gold_answer,
                "facet": query.facet,
                "required": list(query.required_fact_ids),
                "decoys": list(query.decoy_fact_ids),
            }
        )
    (workdir / "queries.json").write_text(
        json.dumps(queries, indent=2), encoding="utf-8"
    )
    return queries


def write_settings(workdir: Path, port: int, model: str) -> None:
    prompts_src = ROOT / "experiments" / "graphrag_prompts"
    prompts_dst = workdir / "prompts"
    if prompts_src.exists():
        shutil.copytree(prompts_src, prompts_dst, dirs_exist_ok=True)
    settings = {
        "models": {
            "default_chat_model": {
                "type": "openai_chat",
                "model_provider": "openai",
                "auth_type": "api_key",
                "api_key": "${GRAPHRAG_API_KEY}",
                "model": model,
                "encoding_model": "cl100k_base",
                "api_base": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "model_supports_json": True,
                "concurrent_requests": 2,
                "async_mode": "threaded",
                "retry_strategy": "native",
                "max_retries": 2,
                "tokens_per_minute": None,
                "requests_per_minute": None,
            },
            "default_embedding_model": {
                "type": "openai_embedding",
                "model_provider": "openai",
                "auth_type": "api_key",
                "api_key": "dummy",
                "model": "hash-embedding",
                "encoding_model": "cl100k_base",
                "api_base": f"http://127.0.0.1:{port}/v1",
                "concurrent_requests": 2,
                "async_mode": "threaded",
                "retry_strategy": "native",
                "max_retries": 2,
                "tokens_per_minute": None,
                "requests_per_minute": None,
            },
        },
        "input": {
            "storage": {"type": "file", "base_dir": "input"},
            "file_type": "text",
            "text_column": "text",
            "title_column": "title",
        },
        "chunks": {"size": 300, "overlap": 0, "group_by_columns": ["id"]},
        "output": {"type": "file", "base_dir": "output"},
        "cache": {"type": "file", "base_dir": "cache"},
        "reporting": {"type": "file", "base_dir": "reports"},
        "vector_store": {
            "default_vector_store": {
                "type": "lancedb",
                "db_uri": "output/lancedb",
                "container_name": "default",
            }
        },
        "embed_text": {
            "model_id": "default_embedding_model",
            "vector_store_id": "default_vector_store",
        },
        "extract_graph": {
            "model_id": "default_chat_model",
            "prompt": "prompts/extract_graph.txt",
            "entity_types": ["concept", "fact", "organization", "person", "location"],
            "max_gleanings": 0,
        },
        "summarize_descriptions": {
            "model_id": "default_chat_model",
            "prompt": "prompts/summarize_descriptions.txt",
            "max_length": 200,
        },
        "cluster_graph": {"max_cluster_size": 5},
        "community_reports": {
            "model_id": "default_chat_model",
            "graph_prompt": "prompts/community_report.txt",
            "text_prompt": "prompts/community_report.txt",
            "max_length": 500,
            "max_input_length": 2000,
        },
        "basic_search": {
            "chat_model_id": "default_chat_model",
            "embedding_model_id": "default_embedding_model",
            "prompt": "prompts/basic_search.txt",
        },
        "local_search": {
            "chat_model_id": "default_chat_model",
            "embedding_model_id": "default_embedding_model",
            "prompt": "prompts/local_search.txt",
        },
        "workflows": [
            "load_input_documents",
            "create_base_text_units",
            "create_final_documents",
            "extract_graph",
            "finalize_graph",
            "create_communities",
            "create_final_text_units",
            "create_community_reports",
            "generate_text_embeddings",
        ],
    }
    (workdir / "settings.yaml").write_text(
        yaml.safe_dump(settings, sort_keys=False), encoding="utf-8"
    )


def content_text(context) -> str:  # noqa: ANN001
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    return json.dumps(context, default=str)


def run(seed: int, port: int, model: str, output_root: Path, local: bool) -> dict:
    workdir = output_root / f"seed{seed}"
    workdir.mkdir(parents=True, exist_ok=True)
    queries = write_benchmark(workdir, seed)
    write_settings(workdir, port, model)
    os.environ["GRAPHRAG_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")

    import graphrag.index.workflows  # noqa: F401
    from graphrag.api import basic_search, build_index, local_search
    from graphrag.config.load_config import load_config

    config = load_config(workdir, workdir / "settings.yaml")
    results = asyncio.run(build_index(config, verbose=False))
    workflow_errors = [str(item.errors) for item in results if item.errors]
    out = workdir / "output"
    text_units = pd.read_parquet(out / "text_units.parquet")
    entities = pd.read_parquet(out / "entities.parquet")
    communities = pd.read_parquet(out / "communities.parquet")
    reports = pd.read_parquet(out / "community_reports.parquet")
    relationships = pd.read_parquet(out / "relationships.parquet")
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8")) if (out / "stats.json").exists() else {}

    rows = []
    for query in queries:
        try:
            response, context = asyncio.run(
                basic_search(config, text_units, query["question"], verbose=False)
            )
            ctx = content_text(context)
            match = CODE_RE.search(str(response or ""))
            parsed = match.group(1) if match else None
            rows.append(
                {
                    **query,
                    "mode": "basic",
                    "response": response,
                    "parsed": parsed,
                    "correct": parsed == query["gold"],
                    "gold_in_context": query["gold"] in ctx,
                    "context_chars": len(ctx),
                }
            )
        except Exception as exc:  # pragma: no cover - network/API failures
            rows.append({**query, "mode": "basic", "error": str(exc), "correct": False})
        print("B", query["id"], rows[-1].get("parsed"), query["gold"], rows[-1].get("correct"), flush=True)

    local_rows = []
    if local:
        for query in queries:
            try:
                response, context = asyncio.run(
                    local_search(
                        config,
                        entities,
                        communities,
                        reports,
                        text_units,
                        relationships,
                        None,
                        1,
                        "Multiple Paragraphs",
                        query["question"],
                        verbose=False,
                    )
                )
                ctx = content_text(context)
                match = CODE_RE.search(str(response or ""))
                parsed = match.group(1) if match else None
                local_rows.append(
                    {
                        **query,
                        "mode": "local",
                        "response": response,
                        "parsed": parsed,
                        "correct": parsed == query["gold"],
                        "gold_in_context": query["gold"] in ctx,
                        "context_chars": len(ctx),
                    }
                )
            except Exception as exc:  # pragma: no cover
                local_rows.append({**query, "mode": "local", "error": str(exc), "correct": False})
            print("L", query["id"], local_rows[-1].get("parsed"), query["gold"], local_rows[-1].get("correct"), flush=True)

    def summarize(items: list[dict]) -> dict:
        valid = [row for row in items if "error" not in row]
        return {
            "n": len(items),
            "n_valid": len(valid),
            "accuracy": sum(row.get("correct", False) for row in items) / max(1, len(items)),
            "context_recall": sum(row.get("gold_in_context", False) for row in items) / max(1, len(items)),
            "mean_context_chars": (
                sum(row.get("context_chars", 0) for row in valid) / max(1, len(valid))
            ),
        }

    payload = {
        "seed": seed,
        "engine": "graphrag==2.7.2",
        "model": model,
        "workflows": [item.workflow for item in results],
        "workflow_errors": workflow_errors,
        "stats": stats,
        "basic": summarize(rows),
        "local": summarize(local_rows) if local else None,
        "rows": rows,
        "local_rows": local_rows,
    }
    (workdir / "results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(
        "ACC",
        payload["basic"]["accuracy"],
        "CTX",
        payload["basic"]["context_recall"],
        "WORKFLOWS",
        len(payload["workflows"]),
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=int(os.environ.get("GRAPHRAG_SEED", "0")))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=os.environ.get("GRAPHRAG_MODEL", "deepseek-flash"))
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "graphrag_official"),
    )
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        with HashEmbeddingServer(port=args.port):
            run(args.seed, args.port, args.model, output_root, args.local)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
