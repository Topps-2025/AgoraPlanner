"""Run official RAPTOR on the public LoCoMo QA set.

The RAPTOR tree builder/retriever is the official ``parthsarthi03/raptor``
implementation.  The QA model is a generic DeepSeek reader (the official
RAPTOR QA extensibility point), and the embedding backend is the same hash
embedder used by the controlled baselines.
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
RAPTOR_REPO = Path(os.environ.get("RAPTOR_REPO", str(ROOT / "baselines" / "raptor")))
for path in [str(RAPTOR_REPO), str(ROOT / "src"), str(ROOT / "experiments")]:
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

if importlib.util.find_spec("sentence_transformers") is not None:
    _st = types.ModuleType("sentence_transformers")

    class _SentenceTransformer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("stubbed")

    _st.SentenceTransformer = _SentenceTransformer
    sys.modules.setdefault("sentence_transformers", _st)
if importlib.util.find_spec("transformers") is not None:
    try:
        import huggingface_hub
        from packaging.version import Version

        _ok = Version(huggingface_hub.__version__) < Version("1.0")
    except Exception:
        _ok = False
    if not _ok:
        _tf = types.ModuleType("transformers")
        _tf.T5ForConditionalGeneration = object
        _tf.T5Tokenizer = object
        sys.modules.setdefault("transformers", _tf)

import tiktoken  # noqa: E402

from raptor import (  # noqa: E402
    BaseEmbeddingModel,
    BaseQAModel,
    BaseSummarizationModel,
    ClusterTreeConfig,
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
    TreeRetrieverConfig,
)

from agora_planner.config import LLMConfig  # noqa: E402
from agora_planner.embeddings import HashEmbedder  # noqa: E402
from agora_planner.llm import DeepSeekClient  # noqa: E402
from run_public_memory_round import exact_match, load_locomo, token_f1  # noqa: E402


class HashEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, dim: int = 384):
        self.embedder = HashEmbedder(dim=dim)

    def create_embedding(self, text):
        return self.embedder.embed(text).tolist()


class BGEEmbeddingModel(BaseEmbeddingModel):
    """Dense BGE embeddings (same local model as the dense_shared baseline)."""

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

    def create_embedding(self, text):
        import torch

        encoded = self.tokenizer(
            [str(text)],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled[0].cpu().tolist()


class DeepSeekSummary(BaseSummarizationModel):
    def __init__(self, client: DeepSeekClient):
        self.client = client

    def summarize(self, context, max_tokens=150):
        return self.client.chat(
            [
                {"role": "system", "content": "Summarize the memory notes, preserving dates and names."},
                {"role": "user", "content": context},
            ],
            max_tokens=max(256, int(max_tokens)),
            temperature=0.0,
        )


class DeepSeekQA(BaseQAModel):
    def __init__(self, client: DeepSeekClient):
        self.client = client

    def answer_question(self, context, question):
        return self.client.chat(
            [
                {
                    "role": "system",
                    "content": "Answer the question from the memory context. Be concise; if absent say unknown.",
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            max_tokens=512,
            temperature=0.0,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=30)
    parser.add_argument("--embedder", default=os.environ.get("RAPTOR_EMBEDDER", "hash"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "raptor_official_locomo"))
    args = parser.parse_args()
    world = load_locomo(
        ROOT / "data" / "LoCoMo" / "repo" / "data" / "locomo10.json",
        max_conversations=2,
        max_queries=15,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    text = "\n".join(memory.content for memory in world.memories)
    client = DeepSeekClient(LLMConfig(max_calls=1200, max_tokens=1024))
    emb = BGEEmbeddingModel() if args.embedder == "bge" else HashEmbeddingModel()
    summary = DeepSeekSummary(client)
    qa = DeepSeekQA(client)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tree_config = ClusterTreeConfig(
        tokenizer=tokenizer,
        max_tokens=512,
        num_layers=3,
        threshold=0.5,
        top_k=5,
        selection_mode="top_k",
        summarization_length=200,
        summarization_model=summary,
        embedding_models={"EMB": emb},
        cluster_embedding_model="EMB",
    )
    retriever_config = TreeRetrieverConfig(
        tokenizer=tokenizer,
        top_k=3,
        threshold=0.5,
        selection_mode="top_k",
        context_embedding_model="EMB",
        embedding_model=emb,
        num_layers=None,
        start_layer=None,
    )
    config = RetrievalAugmentationConfig(
        tree_builder_config=tree_config,
        tree_retriever_config=retriever_config,
        qa_model=qa,
        embedding_model=emb,
        summarization_model=summary,
    )
    ra = RetrievalAugmentation(config)
    ra.add_documents(text)
    rows = []
    for query in world.queries[: args.max_queries]:
        context, _ = ra.retrieve(
            query.question,
            top_k=3,
            max_tokens=1500,
            collapse_tree=True,
            return_layer_information=True,
        )
        answer = qa.answer_question(context, query.question)
        rows.append(
            {
                "query_id": query.id,
                "gold": query.answer,
                "prediction": answer,
                "exact": exact_match(answer, query.answer),
                "f1": token_f1(answer, query.answer),
                "contains": query.answer.lower() in answer.lower(),
                "context_chars": len(context),
            }
        )
        print(query.id, rows[-1]["exact"], rows[-1]["f1"], flush=True)
    payload = {
        "engine": "official parthsarthi03/raptor",
        "dataset": "LoCoMo",
        "n_queries": len(rows),
        "rows": rows,
        "exact": sum(r["exact"] for r in rows) / max(1, len(rows)),
        "f1": sum(r["f1"] for r in rows) / max(1, len(rows)),
        "contains": sum(r["contains"] for r in rows) / max(1, len(rows)),
        "llm_usage": client.usage.to_dict(),
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("ACC", payload["exact"], payload["f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
