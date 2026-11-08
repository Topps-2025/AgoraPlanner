"""Estimate USD API cost for the one-round cloud experiment.

Public QA and RSI runs record prompt/completion tokens from the API response.
GraphRAG/Letta/MARBLE/GateMem rows are marked as estimates unless the run
recorded token usage.  Local Qwen3-8B is served by vLLM and has zero marginal
API cost (GPU amortisation is not included).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(os.environ.get("AGORA_CLOUD_ROOT", "/root/autodl-tmp/agora_planner_cli"))
RESULTS = ROOT / "results"
OUT = RESULTS / "cost_report.json"

# USD per 1M tokens.  DeepSeek flash: off-peak list price (the run was on a
# Sunday UTC); GLM-5.3-Flash: official CNY list price / 7.2 CNY per USD.
RATES = {
    "deepseek-flash": {"input": 0.15, "output": 0.60, "source": "DeepSeek list price, off-peak 2026-10"},
    "glm-5.3-flash": {"input": 0.8 / 7.2, "output": 2.8 / 7.2, "source": "Zhipu list price CNY 0.8/2.8 per 1M, FX 7.2"},
    "glm-4-flash": {"input": 0.8 / 7.2, "output": 2.8 / 7.2, "source": "GLM list-price assumption"},
    "qwen3-8b-local": {"input": 0.0, "output": 0.0, "source": "local vLLM; GPU amortisation excluded"},
}


def cost(prompt: int, completion: int, model: str) -> float:
    rate = RATES.get(model, RATES["deepseek-flash"])
    return prompt / 1_000_000 * rate["input"] + completion / 1_000_000 * rate["output"]


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def public_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("results/public_round_main", "deepseek-flash", "DeepSeek flash"),
        ("results/public_longmemeval_s", "deepseek-flash", "DeepSeek flash"),
        ("results/public_longmemeval_m", "deepseek-flash", "DeepSeek flash"),
        ("results/public_glm_smoke", "glm-4-flash", "GLM-4-flash"),
        ("results/public_glm53_smoke", "glm-5.3-flash", "GLM-5.3-flash"),
        ("results/public_glm53_main", "glm-5.3-flash", "GLM-5.3-flash"),
        ("results/public_qwen_smoke", "qwen3-8b-local", "Qwen3-8B local"),
    ]
    for rel, model, label in specs:
        base = ROOT / rel
        if not base.exists():
            continue
        for path in sorted(base.glob("per_query_*.jsonl")):
            groups: dict[tuple[str, str], dict[str, float]] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("dataset")), str(row.get("baseline")))
                item = groups.setdefault(key, {"prompt": 0.0, "completion": 0.0, "n": 0.0})
                item["prompt"] += float(row.get("prompt_tokens", 0) or 0)
                item["completion"] += float(row.get("completion_tokens", 0) or 0)
                item["n"] += 1
            for (dataset, baseline), item in groups.items():
                rows.append(
                    {
                        "component": "public_qa",
                        "reader": label,
                        "model": model,
                        "dataset": dataset,
                        "baseline": baseline,
                        "n": int(item["n"]),
                        "prompt_tokens": int(item["prompt"]),
                        "completion_tokens": int(item["completion"]),
                        "usd": cost(item["prompt"], item["completion"], model),
                        "estimated": False,
                    }
                )
    return rows


def marble_coding_rows() -> list[dict[str, Any]]:
    base = RESULTS / "marble_coding"
    if not base.exists():
        return []
    total = 0
    n = 0
    for path in sorted(base.glob("task_*_result.jsonl")):
        try:
            data = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        except Exception:
            continue
        total += int(data.get("token_usage", 0) or 0)
        n += 1
    if not n:
        return []
    prompt = int(total * 0.7)
    completion = total - prompt
    return [
        {
            "component": "official_marble",
            "reader": "DeepSeek flash",
            "model": "deepseek-flash",
            "dataset": "MARBLE Coding 20-task pilot",
            "baseline": "MARBLE coding environment",
            "n": n,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": cost(prompt, completion, "deepseek-flash"),
            "estimated": True,
        }
    ]


def marble_comm_rows() -> list[dict[str, Any]]:
    base = RESULTS / "marble_bench_comm"
    if not base.exists():
        return []
    total = 0
    n = 0
    for path in sorted(base.glob("bargaining_task*_result.jsonl")):
        try:
            data = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        except Exception:
            continue
        total += int(data.get("token_usage", 0) or 0)
        n += 1
    if not n:
        return []
    prompt = int(total * 0.7)
    completion = total - prompt
    return [
        {
            "component": "official_marble",
            "reader": "DeepSeek flash",
            "model": "deepseek-flash",
            "dataset": "MARBLE Bargaining communication subsample (10 tasks)",
            "baseline": "MARBLE bargaining + communication evaluator",
            "n": n,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": cost(prompt, completion, "deepseek-flash"),
            "estimated": True,
        }
    ]


def public_stratified_rows() -> list[dict[str, Any]]:
    """Measured usage for the per-instance stratified public runs."""
    dirs = [
        "results/public_stratified_locomo",
        "results/public_stratified_lme",
        "results/public_stratified_lme_mem0_b",
        "results/public_stratified_lme_mem0_a3",
        "results/public_stratified_lme_mem0_c3",
        "results/public_stratified_mab",
        "results/public_stratified_mab_ttl",
        "results/public_stratified_mab_mem0",
    ]
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for rel in dirs:
        for path in sorted((ROOT / rel).glob("per_query_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("query_id")), str(row.get("baseline")))
                current = seen.get(key)
                if current is None or ("error" in current and "error" not in row):
                    seen[key] = row
    prompt = sum(int(row.get("prompt_tokens", 0) or 0) for row in seen.values())
    completion = sum(int(row.get("completion_tokens", 0) or 0) for row in seen.values())
    if not (prompt or completion):
        return []
    return [
        {
            "component": "public_stratified",
            "reader": "DeepSeek flash",
            "model": "deepseek-flash",
            "dataset": "LoCoMo/LongMemEval/MemoryAgentBench stratified",
            "baseline": "6 baselines",
            "n": len(seen),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": cost(prompt, completion, "deepseek-flash"),
            "estimated": False,
        }
    ]


def public_full_rows() -> list[dict[str, Any]]:
    """Measured usage for the full-set LoCoMo / LongMemEval shards."""
    dirs = [f"results/public_full_locomo_{i}" for i in range(5)]
    dirs += [f"results/public_full_lme_{i}" for i in range(3)]
    dirs += [f"results/public_full_locomo_mem0_{i}" for i in (0, 1, 3, 4, 5, 8, 9)]
    dirs += [
        "results/public_full_locomo_dense",
        "results/public_full_locomo_mem0_bge",
        "results/public_full_lme_dense",
    ]
    dirs += [
        f"results/public_full_mab_{suffix}"
        for suffix in (
            "1", "2", "3", "4", "5", "6", "7",
            "ar0", "ar1", "ar2", "ar3", "cr0",
            "memgov_0", "memgov_1", "memgov_2",
        )
    ]
    dirs += [
        str(path.relative_to(ROOT))
        for pattern in ("public_full_mab_mem0_single_*", "public_full_mab_mem0_seq_*")
        for path in sorted(RESULTS.glob(pattern))
    ]
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for rel in dirs:
        for path in sorted((ROOT / rel).glob("per_query_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("query_id")), str(row.get("baseline")))
                current = seen.get(key)
                if current is None or ("error" in current and "error" not in row):
                    seen[key] = row
    prompt = sum(int(row.get("prompt_tokens", 0) or 0) for row in seen.values())
    completion = sum(int(row.get("completion_tokens", 0) or 0) for row in seen.values())
    if not (prompt or completion):
        return []
    return [
        {
            "component": "public_full",
            "reader": "DeepSeek flash",
            "model": "deepseek-flash",
            "dataset": "LoCoMo 1986 / LongMemEval 500 / MemoryAgentBench 3671",
            "baseline": "2-4 baselines",
            "n": len(seen),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": cost(prompt, completion, "deepseek-flash"),
            "estimated": False,
        }
    ]


def raptor_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((RESULTS / "raptor_official").glob("results_seed*.json")):
        data = load(path) or {}
        usage = data.get("llm_usage", {})
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        rows.append(
            {
                "component": "official_raptor",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": "text_memory_26fact",
                "baseline": "RAPTOR official",
                "seed": data.get("seed"),
                "n": len(data.get("rows", [])),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": False,
            }
        )
    bge_locomo = load(RESULTS / "raptor_official_locomo_bge" / "results.json")
    if bge_locomo:
        usage = bge_locomo.get("llm_usage", {})
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        rows.append(
            {
                "component": "official_raptor",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": "LoCoMo (BGE-small)",
                "baseline": "RAPTOR official + BGE-small embeddings",
                "n": int(bge_locomo.get("n_queries", 0)),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": False,
            }
        )
    locomo = load(RESULTS / "raptor_official_locomo" / "results.json")
    if locomo:
        usage = locomo.get("llm_usage", {})
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        rows.append(
            {
                "component": "official_raptor",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": "LoCoMo",
                "baseline": "RAPTOR official + DeepSeek reader",
                "n": int(locomo.get("n_queries", 0)),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": False,
            }
        )
    return rows


def rsi_rows() -> list[dict[str, Any]]:
    rows = []
    for rel in ("results/rsi_qfocus_multiseed", "results/rsi_qfocus"):
        summary = load(ROOT / rel / "summary.json")
        if not summary:
            continue
        for idx, seed_result in enumerate(summary.get("results", [])):
            seed = seed_result.get("seed", idx)
            for mode, body in seed_result.get("results", {}).items():
                usage = body.get("llm_usage") or {}
                prompt = int(usage.get("prompt_tokens", 0))
                completion = int(usage.get("completion_tokens", 0))
                if not prompt and not completion:
                    continue
                rows.append(
                    {
                        "component": "rsi",
                        "reader": "DeepSeek flash",
                        "model": "deepseek-flash",
                        "dataset": "task_collaboration_long_horizon",
                        "baseline": f"meta_rsi/{mode}",
                        "seed": seed,
                        "n": int(usage.get("calls", 0)),
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "usd": cost(prompt, completion, "deepseek-flash"),
                        "estimated": False,
                    }
                )
    return rows


def gatemem_rows() -> list[dict[str, Any]]:
    rows = []
    base = RESULTS / "gatemem_official"
    if not base.exists():
        return rows
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        summary = (
            load(run_dir / "summary.json")
            or load(run_dir / "metrics.json")
            or load(run_dir / "eval_partial" / "summary.json")
            or {}
        )
        # The official runner may nest metrics under "summary" or "metrics".
        payload = summary.get("summary", summary) if isinstance(summary, dict) else {}
        usage = summary.get("llm_usage") or payload.get("llm_usage") or {}
        tokens = (summary.get("llm") or {}).get("tokens", {}) if isinstance(summary, dict) else {}
        prompt = int(usage.get("prompt_tokens", tokens.get("input_tokens", 0)))
        completion = int(usage.get("completion_tokens", tokens.get("output_tokens", 0)))
        n_pred = sum(1 for _ in (run_dir / "predictions.jsonl").open(encoding="utf-8")) if (run_dir / "predictions.jsonl").exists() else 0
        estimated = not (prompt or completion)
        if estimated:
            # Conservative approximation: 1.2k prompt + 120 completion per checkpoint.
            prompt = n_pred * 1200
            completion = n_pred * 120
        rows.append(
            {
                "component": "gatemem_official",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": run_dir.name,
                "baseline": run_dir.name,
                "n": n_pred,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": estimated,
            }
        )
    return rows


def graphrag_rows() -> list[dict[str, Any]]:
    rows = []
    for seed_dir in sorted((RESULTS / "graphrag_official_bge").glob("seed*")):
        result = load(seed_dir / "results.json") or {}
        calls = 45 + (24 if result.get("local") else 0)
        prompt = calls * 1200
        completion = calls * 150
        rows.append(
            {
                "component": "official_graphrag",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": "text_memory_26fact (BGE-small)",
                "baseline": "GraphRAG official basic (BGE)",
                "seed": result.get("seed"),
                "n": len(result.get("rows", [])) + len(result.get("local_rows", [])),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": True,
            }
        )
    for seed_dir in sorted((RESULTS / "graphrag_official").glob("seed*")):
        result = load(seed_dir / "results.json") or {}
        stats = result.get("stats") or {}
        n = len(result.get("rows", [])) + len(result.get("local_rows", []))
        # GraphRAG 2.7.2 stats expose workflow timings, not token usage. Assume
        # 45 indexing calls (26 chunks + community reports) and 48 search calls.
        calls = 45 + (24 if result.get("local") else 0)
        prompt = calls * 1200
        completion = calls * 150
        rows.append(
            {
                "component": "official_graphrag",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": "text_memory_26fact",
                "baseline": "GraphRAG official basic+local",
                "seed": result.get("seed"),
                "n": n,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": True,
            }
        )
    return rows


def letta_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((RESULTS / "letta_official").glob("results_seed*.json")):
        data = load(path) or {}
        n = len(data.get("rows", []))
        # Letta agent turns include tool calls; assume 3 calls per query with a
        # ~7k-token context and ~120 output tokens, based on server logs.
        calls = n * 3
        prompt = calls * 7000
        completion = calls * 120
        rows.append(
            {
                "component": "official_letta",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": "text_memory_26fact",
                "baseline": "Letta 0.16.8 server",
                "seed": data.get("seed"),
                "n": n,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": True,
            }
        )
    return rows


def marble_rows() -> list[dict[str, Any]]:
    rows = []
    for name in ("base_result.jsonl", "coding_result.jsonl"):
        path = RESULTS / "marble_pilot" / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            total = int(data.get("token_usage", 0) or 0)
            prompt = int(total * 0.7)
            completion = total - prompt
            rows.append(
                {
                    "component": "official_marble",
                    "reader": "DeepSeek flash",
                    "model": "deepseek-flash",
                    "dataset": "MARBLE smoke",
                    "baseline": name.replace("_result.jsonl", ""),
                    "n": 1,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "usd": cost(prompt, completion, "deepseek-flash"),
                    "estimated": True,
                }
            )
    return rows


def marble_bench_rows() -> list[dict[str, Any]]:
    """Full official MultiAgentBench Bargaining sweep (100 tasks).

    ``token_usage`` is recorded per task but the prompt/completion split is not,
    so the split is an estimate.
    """
    rows = []
    base = RESULTS / "marble_bench"
    if not base.exists():
        return rows
    files: dict[int, Path] = {}
    for path in base.glob("bargaining_task*_result.jsonl"):
        digits = "".join(ch for ch in path.name.split("_result")[0] if ch.isdigit())
        if not digits:
            continue
        task_id = int(digits)
        current = files.get(task_id)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            files[task_id] = path
    total = 0
    n = 0
    for task_id, path in sorted(files.items()):
        try:
            data = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        except Exception:
            continue
        tokens = int(data.get("token_usage", 0) or 0)
        if not tokens:
            continue
        total += tokens
        n += 1
    if not n:
        return rows
    prompt = int(total * 0.7)
    completion = total - prompt
    rows.append(
        {
            "component": "official_marble",
            "reader": "DeepSeek flash",
            "model": "deepseek-flash",
            "dataset": "MultiAgentBench Bargaining (100 tasks)",
            "baseline": "MARBLE bargaining",
            "n": n,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd": cost(prompt, completion, "deepseek-flash"),
            "estimated": True,
        }
    )
    return rows


def glm53_chunk_rows() -> list[dict[str, Any]]:
    rows = []
    base = RESULTS / "public_glm53_full"
    if not base.exists():
        base = RESULTS / "public_glm53_chunks"
    if not base.exists():
        return rows
    for chunk in sorted(p for p in base.iterdir() if p.is_dir()):
        for path in sorted(chunk.glob("per_query_*.jsonl")):
            prompt = completion = 0
            n = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt += int(row.get("prompt_tokens", 0) or 0)
                completion += int(row.get("completion_tokens", 0) or 0)
                n += 1
            if n:
                rows.append(
                    {
                        "component": "public_glm53_main",
                        "reader": "GLM-5.3-flash",
                        "model": "glm-5.3-flash",
                        "dataset": chunk.name,
                        "baseline": chunk.name,
                        "n": n,
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "usd": cost(prompt, completion, "glm-5.3-flash"),
                        "estimated": False,
                    }
                )
    return rows


def amem_rows() -> list[dict[str, Any]]:
    rows = []
    for rel, label, dataset in (
        ("amem_official", "A-MEM official", "text_memory_26fact"),
        ("amem_official_bge", "A-MEM official (BGE-small)", "text_memory_26fact (BGE)"),
    ):
        data = load(RESULTS / rel / "results_seed0.json")
        if not data:
            continue
        prompt = int(data.get("prompt_tokens", 0))
        completion = int(data.get("completion_tokens", 0))
        rows.append(
            {
                "component": "official_amem",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": dataset,
                "baseline": label,
                "n": len(data.get("rows", [])),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": True,
            }
        )
    return rows


def graphiti_rows() -> list[dict[str, Any]]:
    rows = []
    for rel, label, dataset in (
        ("graphiti_official", "Graphiti official", "text_memory_26fact"),
        ("graphiti_official_bge", "Graphiti official (BGE-small)", "text_memory_26fact (BGE)"),
    ):
        data = load(RESULTS / rel / "results_seed0.json")
        if not data:
            continue
        prompt = int(data.get("prompt_tokens", 0))
        completion = int(data.get("completion_tokens", 0))
        rows.append(
            {
                "component": "official_graphiti",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": dataset,
                "baseline": label,
                "n": int(data.get("n_queries", 0)),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": True,
            }
        )
    return rows


def memlens_rows() -> list[dict[str, Any]]:
    rows = []
    for rel, label, dataset in (
        ("memlens_official", "MemLens official pipeline", "text_memory_26fact"),
        ("memlens_official_bge", "MemLens official (BGE-small)", "text_memory_26fact (BGE)"),
    ):
        data = load(RESULTS / rel / "results_seed0.json")
        if not data:
            continue
        # MemLens does not expose token usage; estimate its hierarchy/Shapley
        # calls plus one reader call per query.
        calls = 92 + 24
        prompt = calls * 500
        completion = calls * 150
        rows.append(
            {
                "component": "official_memlens",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": dataset,
                "baseline": label,
                "n": int(data.get("n_queries", 0)) or len(data.get("rows", [])),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": True,
            }
        )
    return rows


def autogen_rows() -> list[dict[str, Any]]:
    data = load(RESULTS / "autogen_official" / "autogen_pilot.json")
    if not data:
        return []
    rows = []
    for mode, items in data.get("modes", {}).items():
        prompt = sum(int(row.get("prompt_tokens", 0)) for row in items)
        completion = sum(int(row.get("completion_tokens", 0)) for row in items)
        rows.append(
            {
                "component": "official_autogen",
                "reader": "DeepSeek flash",
                "model": "deepseek-flash",
                "dataset": "text_memory_26fact",
                "baseline": f"AutoGen {mode}",
                "n": len(items),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "usd": cost(prompt, completion, "deepseek-flash"),
                "estimated": False,
            }
        )
    return rows


def main() -> int:
    rows: list[dict[str, Any]] = []
    for fn in (
        public_rows,
        public_stratified_rows,
        public_full_rows,
        raptor_rows,
        rsi_rows,
        gatemem_rows,
        graphrag_rows,
        letta_rows,
        marble_rows,
        marble_bench_rows,
        marble_comm_rows,
        marble_coding_rows,
        amem_rows,
        graphiti_rows,
        memlens_rows,
        autogen_rows,
        glm53_chunk_rows,
    ):
        try:
            rows.extend(fn())
        except Exception as exc:  # pragma: no cover - diagnostics
            rows.append({"component": fn.__name__, "error": str(exc)})

    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        if "error" in row:
            continue
        key = row["component"]
        item = totals.setdefault(key, {"usd": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "n": 0.0})
        item["usd"] += float(row.get("usd", 0.0))
        item["prompt_tokens"] += float(row.get("prompt_tokens", 0.0))
        item["completion_tokens"] += float(row.get("completion_tokens", 0.0))
        item["n"] += float(row.get("n", 0.0))
    payload = {
        "rates_usd_per_million": RATES,
        "rows": rows,
        "totals": totals,
        "grand_total_usd": sum(item["usd"] for item in totals.values()),
        "notes": [
            "Public QA and RAPTOR/RSI token counts are measured from API usage.",
            "GraphRAG/Letta/GateMem/MARBLE rows are estimates where usage was not logged.",
            "Local Qwen3-8B has zero marginal API cost; GPU amortisation is excluded.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["totals"], indent=2))
    print("grand_total_usd", payload["grand_total_usd"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
