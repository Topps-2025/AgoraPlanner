"""Run the official Letta/MemGPT V1 server on the text-memory benchmark.

The server is started separately with the project-local Python 3.12 vendor
environment.  This script uses the REST API with a local hash-embedding
endpoint and DeepSeek as the LLM.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("LETTA_BASE_URL", "http://127.0.0.1:8283")
EMBED_BASE = "http://127.0.0.1:8766"
CODE_RE = re.compile(r"\b([A-Z0-9]{4})\b")


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return str(value)



def _answer_text(data: dict) -> str:
    """Return only the assistant messages, excluding reasoning/tool output."""
    if not isinstance(data, dict):
        return str(data)
    parts = []
    for message in data.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        if message.get("message_type") == "assistant_message" and message.get("content"):
            parts.append(str(message["content"]))
    if parts:
        return " ".join(parts)
    return _flatten(data)


def _codes(text: str) -> list[str]:
    return [m.group(1) for m in CODE_RE.finditer(text or "")]


def _wait_health(session: requests.Session, timeout: float = 120.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = session.get(f"{BASE}/v1/health", timeout=2)
            if response.status_code < 500:
                print("health", response.status_code, response.text[:200], flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _create_agent(session: requests.Session, name: str) -> tuple[str | None, dict]:
    payloads = [
        {
            "name": name,
            "memory_blocks": [
                {"label": "persona", "value": "I am a careful memory-grounded agent."},
                {"label": "human", "value": "The team stores facts in archival memory."},
            ],
            "llm_config": {
                "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
                "model_endpoint_type": "deepseek",
                "model_endpoint": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "context_window": 64000,
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            "embedding_config": {
                "embedding_endpoint_type": "openai",
                "embedding_endpoint": EMBED_BASE + "/v1",
                "embedding_model": "hash-embedding",
                "embedding_dim": 384,
                "embedding_chunk_size": 300,
            },
            "tools": ["archival_memory_insert", "archival_memory_search", "conversation_search"],
            "include_base_tools": True,
            "include_base_tool_rules": False,
        },
        {
            "name": name,
            "model": "deepseek/deepseek-flash",
            "embedding": "openai/hash-embedding",
            "memory_blocks": [
                {"label": "persona", "value": "I am a careful memory-grounded agent."},
            ],
            "include_base_tools": True,
        },
    ]
    last: dict = {}
    for payload in payloads:
        response = session.post(f"{BASE}/v1/agents/", json=payload, timeout=60)
        print("create", response.status_code, response.text[:500], flush=True)
        try:
            data = response.json()
        except Exception:
            data = {"text": response.text}
        last = data
        if response.status_code < 300 and isinstance(data, dict):
            agent_id = data.get("id") or data.get("agent_id")
            if agent_id:
                return str(agent_id), data
    return None, last


STOP_CODES = {
    "CODE", "TEXT", "DATA", "FACT", "WITH", "FROM", "THIS", "THAT", "WHAT",
    "WHEN", "WHERE", "WHICH", "VAULT", "FACET", "TRUE", "FALSE", "NONE",
    "NULL", "MORE", "ALSO", "EACH", "THEY", "HAVE", "BEEN", "INTO", "OVER",
    "SUCH", "ONLY", "SOME", "THAN", "THEN", "THEM", "THEIR", "THERE",
    "THESE", "THOSE", "WILL", "WOULD", "COULD", "SHOULD", "ABOUT", "AFTER",
    "BEFORE", "BETWEEN", "DURING", "UNDER", "WHILE",
}


def main() -> None:
    _load_dotenv()
    seed = int(os.environ.get("LETTA_SEED", "0"))
    sys.path.insert(0, str(ROOT / "src"))
    from agora_planner.text_benchmark import TextBenchmarkConfig, TextMemoryBenchmark

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
    session = _session()
    if not _wait_health(session):
        raise SystemExit("Letta server is not healthy")
    agent_id, created = _create_agent(session, f"memgpt-pilot-{seed}")
    if not agent_id:
        raise SystemExit(f"Could not create agent: {created}")
    print("AGENT", agent_id, flush=True)
    for asset in bench.memories:
        response = session.post(
            f"{BASE}/v1/agents/{agent_id}/archival-memory",
            json={"text": asset.content},
            timeout=60,
        )
        if response.status_code >= 300:
            print("insert failed", response.status_code, response.text[:300], flush=True)
    print("inserted", len(bench.memories), flush=True)
    q = bench.queries[bench.eval_tasks[0].id]
    sr = session.get(
        f"{BASE}/v1/agents/{agent_id}/archival-memory/search",
        params={"query": q.question, "top_k": 3},
        timeout=60,
    )
    print("direct_search", sr.status_code, sr.text[:500], flush=True)
    rows = []
    for task in bench.eval_tasks:
        query = bench.queries[task.id]
        response = session.post(
            f"{BASE}/v1/agents/{agent_id}/messages",
            json={"input": "Answer with exactly the access code, no other text. " + query.question, "max_steps": 8},
            timeout=180,
        )
        try:
            data = response.json()
        except Exception:
            data = {"text": response.text}
        text = _answer_text(data)
        found = _codes(text)
        candidates = [c for c in found if c not in STOP_CODES]
        # Prefer tokens that mix letters and digits (the access-code format)
        mixed = [c for c in candidates if any(ch.isalpha() for ch in c) and any(ch.isdigit() for ch in c)]
        parsed = mixed[0] if mixed else (candidates[-1] if candidates else None)
        rows.append({
            "query_id": query.id,
            "gold": query.gold_answer,
            "parsed": parsed,
            "correct": parsed == query.gold_answer,
            "gold_in_response": query.gold_answer in text,
            "status": response.status_code,
            "response_text": text[:2000],
        })
        print(query.id, query.gold_answer, parsed, rows[-1]["correct"], rows[-1]["gold_in_response"], flush=True)
    payload = {
        "seed": seed,
        "agent_id": agent_id,
        "rows": rows,
        "accuracy": sum(r["correct"] for r in rows) / max(1, len(rows)),
        "gold_in_response": sum(r["gold_in_response"] for r in rows) / max(1, len(rows)),
    }
    output = Path(os.environ.get("LETTA_OUTPUT_DIR", str(ROOT / "results" / "letta_official")))
    output.mkdir(parents=True, exist_ok=True)
    (output / f"results_seed{seed}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("ACC", payload["accuracy"], "GOLD_IN_RESP", payload["gold_in_response"], flush=True)


if __name__ == "__main__":
    main()
