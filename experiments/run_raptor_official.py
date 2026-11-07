"""Run the official RAPTOR implementation on the text-memory benchmark.

The official repo is checked out under ``_vendor_test/raptor_extract``.  We use
its extensibility points (custom embedding / summarization / QA models) rather
than modifying the RAPTOR algorithm itself.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_RAPTOR_ENV = os.environ.get("RAPTOR_REPO")
_RAPTOR_CANDIDATES = [
    Path(_RAPTOR_ENV) if _RAPTOR_ENV else None,
    ROOT / "baselines" / "raptor",
    ROOT / "_vendor_test" / "raptor_extract" / "raptor-master",
]
RAPTOR_REPO = next(
    (path for path in _RAPTOR_CANDIDATES if path and (path / "raptor" / "__init__.py").exists()),
    ROOT / "baselines" / "raptor",
)
for path in [
    str(RAPTOR_REPO),
    str(ROOT / ".vendor_raptor"),
    str(ROOT / ".vendor_marble"),
    str(ROOT / ".vendor_graphrag"),
    str(ROOT / "src"),
]:
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "_vendor_test" / "tiktoken_cache"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import importlib.util  # noqa: E402
import types  # noqa: E402

# RAPTOR imports sentence-transformers at module import time, but this runner
# uses its custom HashEmbeddingModel.  Stub the optional dependency so an
# incompatible transformers/huggingface-hub pairing cannot break the import.
if importlib.util.find_spec("sentence_transformers") is not None:
    _st = types.ModuleType("sentence_transformers")

    class _SentenceTransformer:  # pragma: no cover - unused placeholder
        def __init__(self, *args, **kwargs):
            raise RuntimeError("sentence-transformers is stubbed; use HashEmbeddingModel")

    _st.SentenceTransformer = _SentenceTransformer
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
        _tf.T5ForConditionalGeneration = object
        _tf.T5Tokenizer = object
        sys.modules.setdefault("transformers", _tf)

import tiktoken  # noqa: E402

from agora_planner.config import LLMConfig  # noqa: E402
from agora_planner.embeddings import HashEmbedder  # noqa: E402
from agora_planner.llm import DeepSeekClient, _load_dotenv  # noqa: E402
from agora_planner.text_benchmark import (  # noqa: E402
    TextBenchmarkConfig,
    TextMemoryBenchmark,
    extract_code,
)

from raptor import (  # noqa: E402
    BaseEmbeddingModel,
    BaseQAModel,
    BaseSummarizationModel,
    ClusterTreeConfig,
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    TreeRetrieverConfig,
)


class HashEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, dim: int = 384):
        self.embedder = HashEmbedder(dim=dim)

    def create_embedding(self, text):
        return self.embedder.embed(text).tolist()


class DeepSeekSummarizationModel(BaseSummarizationModel):
    def __init__(self, client: DeepSeekClient):
        self.client = client

    def summarize(self, context, max_tokens=150):
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a memory-tree summarizer. Preserve every access "
                        "code and facet name exactly. Do not invent codes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize the following memory notes, keeping every "
                        f"access code and its facet:\n{context}"
                    ),
                },
            ],
            max_tokens=max(256, int(max_tokens)),
            temperature=0.0,
        )
        return response


class DeepSeekQAModel(BaseQAModel):
    def __init__(self, client: DeepSeekClient):
        self.client = client

    def answer_question(self, context, question):
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer from the provided memory context. If multiple "
                        "conflicting codes appear, choose the one with the most "
                        "reliable/consistent evidence. Reply with exactly the "
                        "4-character code or UNKNOWN."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return response


def main() -> None:
    _load_dotenv()
    seed = int(os.environ.get("RAPTOR_SEED", "0"))
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
    text = "\n".join(asset.content for asset in bench.memories)
    client = DeepSeekClient(LLMConfig(max_calls=200, max_tokens=1024))
    embedding_model = HashEmbeddingModel()
    summary_model = DeepSeekSummarizationModel(client)
    qa_model = DeepSeekQAModel(client)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    tree_builder_config = ClusterTreeConfig(
        tokenizer=tokenizer,
        max_tokens=512,
        num_layers=3,
        threshold=0.5,
        top_k=5,
        selection_mode="top_k",
        summarization_length=200,
        summarization_model=summary_model,
        embedding_models={"EMB": embedding_model},
        cluster_embedding_model="EMB",
    )
    tree_retriever_config = TreeRetrieverConfig(
        tokenizer=tokenizer,
        top_k=3,
        threshold=0.5,
        selection_mode="top_k",
        context_embedding_model="EMB",
        embedding_model=embedding_model,
        num_layers=None,
        start_layer=None,
    )
    config = RetrievalAugmentationConfig(
        tree_builder_config=tree_builder_config,
        tree_retriever_config=tree_retriever_config,
        qa_model=qa_model,
        embedding_model=embedding_model,
        summarization_model=summary_model,
    )
    ra = RetrievalAugmentation(config)
    ra.add_documents(text)
    rows = []
    for task in bench.eval_tasks:
        query = bench.queries[task.id]
        context, _ = ra.retrieve(
            query.question,
            top_k=3,
            max_tokens=1500,
            collapse_tree=True,
            return_layer_information=True,
        )
        answer = qa_model.answer_question(context, query.question)
        parsed = extract_code(answer)
        rows.append(
            {
                "query_id": query.id,
                "gold": query.gold_answer,
                "parsed": parsed,
                "correct": parsed == query.gold_answer,
                "gold_in_context": query.gold_answer in context,
                "answer": answer,
                "context_chars": len(context),
            }
        )
        print(query.id, query.gold_answer, parsed, rows[-1]["correct"], rows[-1]["gold_in_context"], flush=True)
    payload = {
        "seed": seed,
        "rows": rows,
        "accuracy": sum(row["correct"] for row in rows) / max(1, len(rows)),
        "context_recall": sum(row["gold_in_context"] for row in rows) / max(1, len(rows)),
        "llm_usage": client.usage.to_dict(),
    }
    output = Path(os.environ.get("RAPTOR_OUTPUT_DIR", str(ROOT / "results" / "raptor_official")))
    output.mkdir(parents=True, exist_ok=True)
    (output / f"results_seed{seed}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("ACC", payload["accuracy"], "CTX", payload["context_recall"], flush=True)


if __name__ == "__main__":
    main()
