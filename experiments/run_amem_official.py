"""Run the official A-MEM memory system on the 26-fact text benchmark.

The official ``agentic_memory`` package is used unchanged for note analysis,
linking, and retrieval.  Two environment adaptations are made:

* ChromaDB's neural embedding function is replaced by the same deterministic
  hash embedder used by the other controlled baselines;
* A-MEM's OpenAI controller is pointed at DeepSeek's OpenAI-compatible endpoint.

The answer reader is the same DeepSeek flash prompt used by the other systems.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMEM_REPO = Path(os.environ.get("AMEM_REPO", str(ROOT / "baselines" / "a-mem")))
for path in [str(AMEM_REPO), str(ROOT / "src")]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("OPENAI_BASE_URL", os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))

# A-MEM imports sentence-transformers at module import time even though this
# runner replaces its Chroma embedding function with a hash embedder.  Stub the
# optional package so the incompatible transformers/huggingface-hub pairing in
# the shared venv cannot break the import.
import importlib.util  # noqa: E402
import types  # noqa: E402

if importlib.util.find_spec("sentence_transformers") is not None:
    _st = types.ModuleType("sentence_transformers")

    class _SentenceTransformer:  # pragma: no cover - unused placeholder
        def __init__(self, *args, **kwargs):
            raise RuntimeError("sentence-transformers is stubbed; use the hash embedder")

    _st.SentenceTransformer = _SentenceTransformer
    _st.__version__ = "0.0.0-stub"
    sys.modules.setdefault("sentence_transformers", _st)

if importlib.util.find_spec("transformers") is not None:
    try:
        import huggingface_hub  # noqa: F401
        from packaging.version import Version

        _hub_ok = Version(huggingface_hub.__version__) < Version("1.0")
    except Exception:
        _hub_ok = False
    if not _hub_ok:
        _tf = types.ModuleType("transformers")
        _tf.AutoModel = object
        _tf.AutoTokenizer = object
        _tf.AutoConfig = object
        sys.modules.setdefault("transformers", _tf)

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from openai import OpenAI  # noqa: E402

from agora_planner.embeddings import HashEmbedder  # noqa: E402
from agora_planner.text_benchmark import (  # noqa: E402
    TextBenchmarkConfig,
    TextMemoryBenchmark,
    extract_code,
)

CODE_RE = re.compile(r"\b([A-Z0-9]{4})\b")


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


class HashChromaRetriever:
    """Minimal Chroma retriever with hash or BGE embeddings (official API)."""

    def __init__(self, collection_name: str = "memories", model_name: str = "hash", dim: int = 384):
        if os.environ.get("AMEM_EMBEDDER", "hash").lower() == "bge":
            self.embedder = BGEEmbedder()
        else:
            self.embedder = HashEmbedder(dim=dim)
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=None
        )

    def add_document(self, document: str, metadata: dict, doc_id: str):
        processed = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                processed[key] = json.dumps(value)
            else:
                processed[key] = str(value)
        self.collection.add(
            documents=[document],
            embeddings=[self.embedder.embed(document).tolist()],
            metadatas=[processed],
            ids=[doc_id],
        )

    def delete_document(self, doc_id: str):
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5):
        return self.collection.query(
            query_embeddings=[self.embedder.embed(query).tolist()], n_results=k
        )


class Reader:
    def __init__(self, model: str = "deepseek-flash", max_tokens: int = 512):
        self.client = OpenAI(
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            timeout=120,
        )
        self.model = model
        self.max_tokens = max_tokens

    def answer(self, question: str, memories: list[dict]) -> tuple[str, int, int]:
        context = "\n".join(
            f"[{idx + 1}] {item.get('content', '')}" for idx, item in enumerate(memories)
        )
        prompt = (
            "Answer the question using only the retrieved memory passages. "
            "Be concise. If the answer is not present, answer \"unknown\".\n\n"
            f"Question: {question}\n\nPassages:\n{context}\n\n"
            'Return JSON: {"answer": "..."}'
        )
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return text, int(getattr(usage, "prompt_tokens", 0) or 0), int(
            getattr(usage, "completion_tokens", 0) or 0
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "amem_official"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Patch the official MemorySystem's retriever class; the A-MEM algorithm
    # (note analysis, linking, retrieval) is otherwise unchanged.
    import agentic_memory.memory_system as memory_system
    from agentic_memory.llm_controller import OpenAIController

    memory_system.ChromaRetriever = HashChromaRetriever

    # DeepSeek supports JSON object mode but not OpenAI's json_schema response
    # format.  Keep A-MEM's prompts and JSON parsing, but use the compatible
    # response_format value.
    _orig_completion = OpenAIController.get_completion

    def _compatible_completion(self, prompt, response_format, temperature=0.7):
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            response_format = {"type": "json_object"}
        return _orig_completion(self, prompt, response_format, temperature)

    OpenAIController.get_completion = _compatible_completion
    from agentic_memory.memory_system import AgenticMemorySystem as MemorySystem

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
    system = MemorySystem(
        model_name="hash",
        llm_backend="openai",
        llm_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        evo_threshold=10000,
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
    )
    ingest_start = time.time()
    for asset in bench.memories:
        system.add_note(asset.content)
    ingest_s = time.time() - ingest_start
    reader = Reader()
    rows = []
    for task in bench.eval_tasks:
        query = bench.queries[task.id]
        retrieved = system.search(query.question, k=args.top_k)
        answer, prompt_tokens, completion_tokens = reader.answer(query.question, retrieved)
        parsed = extract_code(answer)
        rows.append(
            {
                "query_id": query.id,
                "gold": query.gold_answer,
                "parsed": parsed,
                "correct": parsed == query.gold_answer,
                "gold_in_context": any(query.gold_answer in item.get("content", "") for item in retrieved),
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "contexts": [item.get("content", "")[:300] for item in retrieved],
            }
        )
        print(query.id, query.gold_answer, parsed, rows[-1]["correct"], flush=True)
    payload = {
        "seed": args.seed,
        "engine": "official agentic_memory (A-MEM)",
        "n_facts": len(list(bench.memories)),
        "ingest_s": ingest_s,
        "rows": rows,
        "accuracy": sum(row["correct"] for row in rows) / max(1, len(rows)),
        "context_recall": sum(row["gold_in_context"] for row in rows) / max(1, len(rows)),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
    }
    (output_dir / f"results_seed{args.seed}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("ACC", payload["accuracy"], "CTX", payload["context_recall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
