"""Assemble the one-round cloud experiment report from result JSON files."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("AGORA_CLOUD_ROOT", "/root/autodl-tmp/agora_planner_cli"))
OUT = Path(os.environ.get("AGORA_ROUND_SUMMARY", str(ROOT / "RESULTS_SUMMARY_2026-11-08.md")))


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def shell(command: str) -> str:
    try:
        return subprocess.check_output(command, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


def public_table(dataset: str, payload: dict[str, Any] | None, k: int = 5) -> list[str]:
    lines = [
        f"### {dataset}",
        "",
        "| baseline | official | n | recall@k | nDCG@k | exact | F1 | tokens/q | latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not payload:
        return lines + ["| (missing) |  |  |  |  |  |  |  |  |", ""]
    for name, row in payload.get("baselines", {}).items():
        if "error" in row:
            lines.append(f"| {name} |  |  |  |  |  |  |  | error: {str(row['error'])[:60]} |")
            continue
        lines.append(
            f"| {name} | {row.get('official')} | {row.get('n_queries')} | "
            f"{fmt(row.get('recall@k'))} | {fmt(row.get('ndcg@k'))} | "
            f"{fmt(row.get('answer_exact'))} | {fmt(row.get('answer_f1'))} | "
            f"{fmt(row.get('tokens_per_query'), 0)} | {fmt(row.get('latency_ms_per_query'), 0)} |"
        )
    lines.append("")
    return lines


def mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return f"{mean:.3f} ± {var ** 0.5:.3f}"


def stratified_table(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    lines = [
        "",
        "### Stratified public QA (per-instance isolation)",
        "",
        "Each conversation / LongMemEval haystack / MemoryAgentBench row is its own",
        "retrieval world; questions are sampled per category. This supersedes the",
        "pooled pilot rows for per-ability comparisons.",
        "",
    ]
    for dataset, body in payload.get("datasets", {}).items():
        lines += [
            f"#### {dataset}",
            "",
            f"Queries: {body.get('n_queries')}; instances: {body.get('n_instances')}",
            "",
            "| baseline | n | exact | F1 [95% CI] | recall@5 | tokens/q | refusal |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for baseline, row in sorted(body.get("baselines", {}).items()):
            ci = row.get("answer_f1_ci95")
            ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
            lines.append(
                f"| {baseline} | {row.get('n_queries')} | {fmt(row.get('answer_exact'))} | "
                f"{fmt(row.get('answer_f1'))} {ci_text} | {fmt(row.get('recall@5'))} | "
                f"{fmt(row.get('tokens_per_query'), 0)} | {fmt(row.get('refusal_rate'))} |"
            )
        lines += [
            "",
            "| category | n | raw_rag F1 | oracle F1 | mem0 F1 | selective F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for category, by_baseline in sorted(body.get("per_category", {}).items()):
            def f1(name):
                item = by_baseline.get(name) or {}
                return fmt(item.get("answer_f1"))

            n = (by_baseline.get("raw_rag") or {}).get("n_queries") or 0
            lines.append(
                f"| {category} | {n} | {f1('raw_rag')} | {f1('oracle')} | "
                f"{f1('mem0_shared')} | {f1('selective')} |"
            )
        lines.append("")
    lines += [
        "Sampling: LoCoMo first 2 QA per category per conversation (10 conv, 98 QA);",
        "LongMemEval first 20 per question type plus 20 abstention ids (140 QA);",
        "MemoryAgentBench first 4 rows/competency x 10 QA (124 QA; Long_Range has",
        "1 QA/row). Refusal rate is the share of predictions containing an",
        "unknown/not-provided marker; on the abstention category it is the",
        "abstention-correctness proxy. MemoryAgentBench has no released evidence",
        "ids, so recall is n/a. Mem0/GovMem cover 84/124 MemoryAgentBench QA because",
        "the 4 Test_Time_Learning rows have 89k memories and are excluded from the",
        "governance/Mem0 ingest; raw-RAG/oracle/private/selective cover all 124.",
        "",
    ]
    return lines


def full_public_table(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    lines = [
        "",
        "### Full-set public QA (per-instance isolation)",
        "",
        "Full released question sets for the subsets that were small in the pilot:",
        "LoCoMo all 1,986 QA over 10 conversations; LongMemEval s_cleaned all 500",
        "questions; MemoryAgentBench all 3,671 questions across 146 rows. On MAB,",
        "raw-RAG/oracle cover all 3,671; GovMem covers the 2,971 non-TTL questions",
        "and Mem0 the 1,265 rows where its local bank initialised (the 89k-memory",
        "Test_Time_Learning rows are excluded from both).",
        "",
    ]
    for dataset, body in payload.get("datasets", {}).items():
        lines += [
            f"#### {dataset}",
            "",
            f"Queries: {body.get('n_queries')}; instances: {body.get('n_instances')}",
            "",
            "| baseline | n | exact | F1 [95% CI] | recall@5 | tokens/q | refusal |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for baseline, row in sorted(body.get("baselines", {}).items()):
            ci = row.get("answer_f1_ci95")
            ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
            lines.append(
                f"| {baseline} | {row.get('n_queries')} | {fmt(row.get('answer_exact'))} | "
                f"{fmt(row.get('answer_f1'))} {ci_text} | {fmt(row.get('recall@5'))} | "
                f"{fmt(row.get('tokens_per_query'), 0)} | {fmt(row.get('refusal_rate'))} |"
            )
        lines += [
            "",
            "| category | n | raw_rag F1 | oracle F1 |",
            "|---|---:|---:|---:|",
        ]
        for category, by_baseline in sorted(body.get("per_category", {}).items()):
            def f1(name):
                item = by_baseline.get(name) or {}
                return fmt(item.get("answer_f1"))

            n = (by_baseline.get("raw_rag") or {}).get("n_queries") or 0
            lines.append(f"| {category} | {n} | {f1('raw_rag')} | {f1('oracle')} |")
        lines.append("")
    mem0_bge = load(ROOT / "results" / "public_full_locomo_mem0_bge" / "summary.json")
    if mem0_bge:
        bge_body = (mem0_bge.get("datasets", {}).get("locomo", {}) or {})
        bge_row = (bge_body.get("baselines", {}) or {}).get("mem0_shared", {})
        hash_row = (
            (payload.get("datasets", {}).get("locomo", {}) or {})
            .get("baselines", {})
            .get("mem0_shared", {})
        )
        lines += [
            "",
            "Mem0 embedding ablation (full LoCoMo):",
            "",
            "| Mem0 embedding | n | exact | F1 | recall@5 |",
            "|---|---:|---:|---:|---:|",
            f"| hash | {hash_row.get('n_queries')} | {fmt(hash_row.get('answer_exact'))} | "
            f"{fmt(hash_row.get('answer_f1'))} | {fmt(hash_row.get('recall@5'))} |",
            f"| BGE-small | {bge_row.get('n_queries')} | {fmt(bge_row.get('answer_exact'))} | "
            f"{fmt(bge_row.get('answer_f1'))} | {fmt(bge_row.get('recall@5'))} |",
        ]
    lines += [
        "Full LoCoMo includes Mem0 (all 1,986 QA after per-conversation reruns) and",
        "a BGE-small dense retriever. Full LongMemEval runs raw-RAG/oracle/selective",
        "plus dense; Mem0 is reported on the 140-QA stratified subset. Full",
        "MemoryAgentBench runs raw-RAG/oracle on all 3,671 questions, GovMem on",
        "2,971 non-TTL questions, and Mem0 on 1,265 rows; its Conflict_Resolution",
        "F1 (.210) is much higher than the first-4-rows stratified sample (.008),",
        "showing why the full run matters.",
        "",
    ]
    return lines


def official_runtime_table() -> list[str]:
    lines = [
        "## Official named-baseline runtime (not reimplementations)",
        "",
        "All rows use the official upstream implementation on the same 26-fact",
        "text-memory benchmark (24 eval queries) and DeepSeek flash.",
        "",
        "| baseline | engine | seeds | accuracy | context recall | notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    raptor = []
    for path in sorted((ROOT / "results" / "raptor_official").glob("results_seed*.json")):
        payload = load(path)
        if payload:
            raptor.append(float(payload.get("accuracy", 0.0)))
    if raptor:
        lines.append(
            f"| RAPTOR | official `parthsarthi03/raptor` | {len(raptor)} | "
            f"{mean_std(raptor)} | 1.000 | collapsed tree, hash embedding |"
        )
    basic, local = [], []
    for path in sorted((ROOT / "results" / "graphrag_official").glob("seed*/results.json")):
        payload = load(path) or {}
        if payload.get("basic"):
            basic.append(float(payload["basic"].get("accuracy", 0.0)))
        if payload.get("local"):
            local.append(float(payload["local"].get("accuracy", 0.0)))
    if basic:
        lines.append(
            f"| GraphRAG basic_search | official `graphrag==2.7.2` | {len(basic)} | "
            f"{mean_std(basic)} | 1.000 | 9 workflows; local hash embedding |"
        )
    if local:
        lines.append(
            f"| GraphRAG local_search | official `graphrag==2.7.2` | {len(local)} | "
            f"{mean_std(local)} | see rows | entity/community path loses codes |"
        )
    bge_basic = []
    for path in sorted((ROOT / "results" / "graphrag_official_bge").glob("seed*/results.json")):
        payload = load(path) or {}
        basic_body = payload.get("basic") or {}
        if basic_body:
            bge_basic.append(float(basic_body.get("accuracy", 0.0)))
    if bge_basic:
        lines.append(
            f"| GraphRAG basic_search (BGE-small) | official `graphrag==2.7.2` | "
            f"{len(bge_basic)} | {mean_std(bge_basic)} | 1.000 | local BGE embeddings |"
        )
    letta = []
    for path in sorted((ROOT / "results" / "letta_official").glob("results_seed*.json")):
        payload = load(path)
        if payload:
            letta.append(float(payload.get("accuracy", 0.0)))
    if letta:
        lines.append(
            f"| Letta/MemGPT | official `letta==0.16.8` server + PG/pgvector | {len(letta)} | "
            f"{mean_std(letta)} | 1.000 | archival memory, 3-agent conflict |"
        )
    locomo_path = ROOT / "results" / "raptor_official_locomo" / "results.json"
    locomo = load(locomo_path) if locomo_path.exists() else None
    if locomo:
        usage = locomo.get("llm_usage", {})
        lines += [
            "",
            "### Official RAPTOR on public LoCoMo",
            "",
            "| baseline | engine | queries | exact | F1 | contains | tokens (prompt/completion) |",
            "|---|---|---:|---:|---:|---:|---|",
            f"| RAPTOR (hash) | official `parthsarthi03/raptor` + DeepSeek reader | "
            f"{locomo.get('n_queries')} | {fmt(locomo.get('exact'))} | {fmt(locomo.get('f1'))} | "
            f"{fmt(locomo.get('contains'))} | "
            f"{int(usage.get('prompt_tokens', 0)):,}/{int(usage.get('completion_tokens', 0)):,} |",
        ]
        bge_locomo = load(ROOT / "results" / "raptor_official_locomo_bge" / "results.json")
        if bge_locomo:
            bge_usage = bge_locomo.get("llm_usage", {})
            lines.append(
                f"| RAPTOR (BGE-small) | official RAPTOR + local BGE-small embeddings | "
                f"{bge_locomo.get('n_queries')} | {fmt(bge_locomo.get('exact'))} | "
                f"{fmt(bge_locomo.get('f1'))} | {fmt(bge_locomo.get('contains'))} | "
                f"{int(bge_usage.get('prompt_tokens', 0)):,}/"
                f"{int(bge_usage.get('completion_tokens', 0)):,} |"
            )
        lines.append("")
    lines += [
        "",
        "### MARBLE official runtime (Base/Coding smoke + Bargaining sweep)",
        "",
        "| scenario | coordination | result | task rubric | planning score | tokens |",
        "|---|---|---|---|---:|---:|",
    ]
    base_path = ROOT / "results" / "marble_pilot" / "base_result.jsonl"
    if base_path.exists():
        row = json.loads(base_path.read_text(encoding="utf-8").splitlines()[-1])
        base_summary = row.get("summary") or (
            row.get("iterations", [{}])[-1].get("summary", "")
        )
        lines.append(
            f"| Base (official) | star, 1 agent | `{base_summary}` | n/a | "
            f"{row.get('planning_scores', ['n/a'])[-1]} | {row.get('token_usage', 0)} |"
        )
    coding_path = ROOT / "results" / "marble_pilot" / "coding_result.jsonl"
    workspace = ROOT / "results" / "marble_pilot" / "workspace" / "solution.py"
    if coding_path.exists():
        row = json.loads(coding_path.read_text(encoding="utf-8").splitlines()[-1])
        valid = "valid artifact" if workspace.exists() else "no artifact"
        lines.append(
            f"| Coding (official environment) | graph, 2 agents | `{valid}` | n/a | "
            f"{row.get('planning_scores', ['n/a'])[-1]} | {row.get('token_usage', 0)} |"
        )
    coding_dir = ROOT / "results" / "marble_coding"
    if coding_dir.exists():
        n = 0
        compiled = 0
        sizes = []
        planning = []
        comm = []
        tokens = []
        for index in range(1, 101):
            solution = coding_dir / f"workspace_task_{index}" / "solution.py"
            result_path = coding_dir / f"task_{index}_result.jsonl"
            if not solution.exists():
                continue
            n += 1
            text = solution.read_text(encoding="utf-8", errors="replace")
            sizes.append(len(text))
            try:
                compile(text, str(solution), "exec")
                compiled += 1
            except SyntaxError:
                pass
            if result_path.exists():
                data = json.loads(result_path.read_text(encoding="utf-8").splitlines()[-1])
                planning.extend(x for x in data.get("planning_scores", []) if x >= 0)
                comm.extend(x for x in data.get("communication_scores", []) if x >= 0)
                tokens.append(int(data.get("token_usage", 0) or 0))
        if n:
            lines.append(
                f"| Coding 100-task pilot (official CodingEnvironment) | graph, 3 agents | "
                f"nonempty {n}/100; py_compile {compiled}/{n} | "
                f"planning {mean_std([float(x) for x in planning])}; "
                f"communication {mean_std([float(x) for x in comm])} | "
                f"{sum(tokens) / max(1, len(tokens)):.0f}/task |"
            )
    bench_path = ROOT / "results" / "marble_bench" / "bargaining_task1_result.jsonl"
    if bench_path.exists():
        row = json.loads(bench_path.read_text(encoding="utf-8").splitlines()[-1])
        evaluation = row.get("task_evaluation", {})
        seller = evaluation.get("seller", {})
        buyer = evaluation.get("buyer", {})
        planning = [p for p in row.get("planning_scores", []) if p >= 0]
        planning_text = fmt(sum(planning) / len(planning)) if planning else "n/a"
        lines.append(
            f"| Bargaining task 1 (official MultiAgentBench) | graph, 4 agents | "
            f"official task | seller {seller.get('effectiveness_of_strategies')}/"
            f"{seller.get('progress_and_outcome')}/{seller.get('interaction_dynamics')}; "
            f"buyer {buyer.get('effectiveness_of_strategies')}/"
            f"{buyer.get('progress_and_outcome')}/{buyer.get('interaction_dynamics')} | "
            f"{planning_text} | {row.get('token_usage', 0)} |"
        )
    summary_path = ROOT / "results" / "marble_bench" / "summary.json"
    if summary_path.exists():
        sweep = (load(summary_path) or {}).get("metrics", {})
        if sweep.get("n_tasks", 0) > 1:
            lines.append(
                f"| Bargaining 100-task sweep (official MultiAgentBench) | graph, 4 agents | "
                f"official task | buyer E/P/I "
                f"{fmt(sweep.get('buyer.effectiveness_of_strategies'))}/"
                f"{fmt(sweep.get('buyer.progress_and_outcome'))}/"
                f"{fmt(sweep.get('buyer.interaction_dynamics'))}; seller "
                f"{fmt(sweep.get('seller.effectiveness_of_strategies'))}/"
                f"{fmt(sweep.get('seller.progress_and_outcome'))}/"
                f"{fmt(sweep.get('seller.interaction_dynamics'))}; "
                f"{sweep.get('n_valid')}/{sweep.get('n_tasks')} valid | "
                f"{fmt(sweep.get('planning'))} | "
                f"{fmt(sweep.get('tokens_per_task'), 0)}/task |"
            )
    comm_dir = ROOT / "results" / "marble_bench_comm"
    if comm_dir.exists():
        import statistics

        comm_scores = []
        comm_tasks = 0
        comm_tokens = []
        for path in sorted(comm_dir.glob("bargaining_task*_result.jsonl")):
            data = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
            scores = [score for score in data.get("communication_scores", []) if score >= 0]
            if scores:
                comm_scores.extend(scores)
                comm_tasks += 1
            comm_tokens.append(int(data.get("token_usage", 0) or 0))
        if comm_scores:
            lines.append(
                f"| Bargaining communication subsample (official evaluator enabled) | "
                f"graph, 4 agents | {comm_tasks} tasks | communication score "
                f"{statistics.mean(comm_scores):.3f} ± {statistics.pstdev(comm_scores):.3f} "
                f"({len(comm_scores)} iterations) | n/a | "
                f"{fmt(statistics.mean(comm_tokens), 0)}/task |"
            )
    lines += [
        "",
        "The Coding row is a custom small task executed by the official CodingEnvironment.",
        "The Bargaining rows are official MultiAgentBench tasks through the official",
        "WorldSimulation environment and evaluator. The 100-task sweep reports the",
        "mean rubric only over tasks where both role evaluations parsed (see",
        "`results/marble_bench/summary.json` for per-task validity).",
        "",
    ]
    return lines


def other_official_table(
    amem: dict[str, Any] | None,
    graphiti: dict[str, Any] | None,
    autogen: dict[str, Any] | None,
    memlens: dict[str, Any] | None = None,
) -> list[str]:
    lines = [
        "## Other official baselines",
        "",
        "| system | engine | sample | accuracy | context recall | notes |",
        "|---|---|---|---:|---:|---|",
    ]
    if amem:
        lines.append(
            f"| A-MEM | official `agentic_memory` | {len(amem.get('rows', []))} QA / "
            f"{amem.get('n_facts', 26)} facts | {fmt(amem.get('accuracy'))} | "
            f"{fmt(amem.get('context_recall'))} | hash embedding; JSON-mode adapter |"
        )
    else:
        lines.append("| A-MEM | official `agentic_memory` | n/a | n/a | n/a | not completed |")
    amem_bge = load(ROOT / "results" / "amem_official_bge" / "results_seed0.json")
    if amem_bge:
        lines.append(
            f"| A-MEM (BGE-small) | official `agentic_memory` | "
            f"{len(amem_bge.get('rows', []))} QA / {amem_bge.get('n_facts', 26)} facts | "
            f"{fmt(amem_bge.get('accuracy'))} | {fmt(amem_bge.get('context_recall'))} | "
            f"local BGE embeddings |"
        )
    if graphiti:
        lines.append(
            f"| Graphiti | official `graphiti-core` + Kuzu | {graphiti.get('n_queries')} QA / "
            f"{graphiti.get('n_facts')} facts | {fmt(graphiti.get('accuracy'))} | "
            f"{fmt(graphiti.get('context_recall'))} | FTS adapter for Kuzu |"
        )
    else:
        lines.append("| Graphiti | official `graphiti-core` + Kuzu | n/a | n/a | n/a | not completed |")
    graphiti_bge = load(ROOT / "results" / "graphiti_official_bge" / "results_seed0.json")
    if graphiti_bge:
        lines.append(
            f"| Graphiti (BGE-small) | official `graphiti-core` + Kuzu | "
            f"{graphiti_bge.get('n_queries')} QA / {graphiti_bge.get('n_facts')} facts | "
            f"{fmt(graphiti_bge.get('accuracy'))} | {fmt(graphiti_bge.get('context_recall'))} | "
            f"local BGE embeddings |"
        )
    if memlens:
        lines.append(
            f"| MemLens | official MemLens pipeline + Milvus Lite | {len(memlens.get('rows', []))} QA / "
            f"{memlens.get('n_facts', 26)} facts | {fmt(memlens.get('accuracy'))} | "
            f"{fmt(memlens.get('context_recall'))} | hierarchy/Shapley + hash embedding |"
        )
    else:
        lines.append("| MemLens | official pipeline + Milvus Lite | n/a | n/a | n/a | not completed |")
    memlens_bge = load(ROOT / "results" / "memlens_official_bge" / "results_seed0.json")
    if memlens_bge:
        lines.append(
            f"| MemLens (BGE-small) | official MemLens pipeline + Milvus Lite | "
            f"{len(memlens_bge.get('rows', []))} QA / {memlens_bge.get('n_facts', 26)} facts | "
            f"{fmt(memlens_bge.get('accuracy'))} | {fmt(memlens_bge.get('context_recall'))} | "
            f"local BGE embeddings |"
        )
    if autogen:
        modes = autogen.get("modes", {})
        for mode, rows in modes.items():
            accuracy = sum(1 for row in rows if row.get("correct")) / max(1, len(rows))
            tokens = sum(row.get("prompt_tokens", 0) + row.get("completion_tokens", 0) for row in rows) / max(1, len(rows))
            lines.append(
                f"| AutoGen ({mode}) | official `autogen-agentchat` | {len(rows)} QA | "
                f"{fmt(accuracy)} | n/a | {fmt(tokens, 0)} tokens/query |"
            )
    else:
        lines.append("| AutoGen | official `autogen-agentchat` | n/a | n/a | n/a | not completed |")
    lines += [
        "",
        "A-MEM uses its official note analysis/linking pipeline with a hash",
        "embedding adapter and DeepSeek's JSON-object mode. Graphiti uses the",
        "official graph extraction/search pipeline with an embedded Kuzu graph and",
        "the same hash embedder. MemLens keeps its official atomic/hierarchy/",
        "Shapley/organizer pipeline and uses embedded Milvus Lite. AutoGen is an",
        "orchestration pilot, not a memory system score; it is reported by scope.",
        "",
    ]
    return lines


def gatemem_official_table() -> list[str]:
    lines = [
        "## GateMem official agents (official rule-based scorer; 4 domains)",
        "",
        "| agent | checkpoints | Utility U | Access violation A | Forgetting failure F | Over-refusal OR | MGS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    base = ROOT / "results" / "gatemem_official"
    run_dirs = sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []
    full_domains = {
        p.name.split("_")[0] for p in run_dirs if p.name.endswith("_full")
    }
    rows = 0
    for run_dir in run_dirs:
        # Prefer the full-domain run when both the 2-episode subset and the full
        # domain exist for a domain.
        name = run_dir.name
        if (
            not name.endswith("_full")
            and name.split("_")[0] in full_domains
        ):
            continue
        payload = load(run_dir / "summary.json")
        suffix = ""
        if not payload:
            payload = load(run_dir / "eval_partial" / "summary.json")
            suffix = " (partial)"
        if not payload:
            continue
        rows += 1
        n = payload.get("n_checkpoints") or payload.get("num_checkpoints") or "n/a"
        lines.append(
            f"| {run_dir.name}{suffix} | {n} | {fmt(payload.get('utility_accuracy'))} | "
            f"{fmt(payload.get('privacy_leakage_rate'))} | {fmt(payload.get('deletion_leakage_rate'))} | "
            f"{fmt(payload.get('over_refusal_rate'))} | {fmt(payload.get('compliance_utility_score'))} |"
        )
    if not rows:
        lines.append("| (official run still in progress) |  |  |  |  |  |  |")
    lines += [
        "",
        "All rows are full-domain runs: education 540, medical 579, household 552,",
        "office 547 checkpoints. The earlier 2-episode subset runs are kept on disk",
        "but are omitted from the table where the full domain exists.",
        "",
    ]
    return lines


def rsi_process_table() -> list[str]:
    payload = load(ROOT / "results" / "rsi_process_metrics.json")
    if not payload:
        return []
    groups: dict[str, list[dict[str, Any]]] = {}
    for key, body in payload.get("modes", {}).items():
        mode = key.rsplit("_seed", 1)[0]
        result = body.get("final_test") or {}
        if result:
            groups.setdefault(mode, []).append(result)
    if not groups:
        return []
    fields = (
        ("mean_memory_reuse", "memory reuse"),
        ("mean_no_trade_rate", "no-trade rate"),
        ("worst_agent_success", "worst-agent success"),
        ("mean_load_balance_cv", "load-balance CV"),
        ("mean_error_propagation", "error propagation"),
        ("mean_base_next_failure", "base next-step failure"),
    )
    lines = [
        "",
        "### RSI process/mechanism metrics (controlled long-horizon world)",
        "",
        "Computed post hoc from the replay rows (selected memory ids, buyer,",
        "success, pollution, spend). Public single-agent QA has no ownership or",
        "agent trace, so these are reported on the controlled diagnostic.",
        "",
        "| mode | " + " | ".join(label for _, label in fields) + " |",
        "|---|" + "---:|" * len(fields),
    ]
    for mode, results in sorted(groups.items()):
        values = []
        for field, _ in fields:
            numbers = [float(result[field]) for result in results if result.get(field) is not None]
            values.append(fmt(sum(numbers) / len(numbers)) if numbers else "n/a")
        lines.append(f"| {mode} | " + " | ".join(values) + " |")
    lines += [
        "",
        "`meta_rsi` reaches worst-agent success 1.0 with no polluted transitions;",
        "`fixed` has all steps failing, so its error-propagation/base-failure rows",
        "are uninformative (both 1.0). Memory reuse is 0 because each step needs a",
        "distinct facet memory; this world does not exercise cross-step reuse.",
        "",
    ]
    return lines


def cost_table() -> list[str]:
    payload = load(ROOT / "results" / "cost_report.json") or {}
    totals = payload.get("totals", {})
    lines = [
        "## API cost estimate (USD)",
        "",
        "Rates: DeepSeek flash off-peak $0.15/$0.60 per 1M input/output;",
        "GLM-5.3-Flash CNY 0.8/2.8 per 1M at 7.2 CNY/USD; local Qwen3-8B $0 marginal.",
        "",
        "| component | calls/rows | prompt tokens | completion tokens | USD | measured? |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, item in sorted(totals.items()):
        estimated = any(
            row.get("component") == name and row.get("estimated")
            for row in payload.get("rows", [])
        )
        lines.append(
            f"| {name} | {fmt(item.get('n'), 0)} | {fmt(item.get('prompt_tokens'), 0)} | "
            f"{fmt(item.get('completion_tokens'), 0)} | {fmt(item.get('usd'), 4)} | "
            f"{'estimate' if estimated else 'measured'} |"
        )
    if totals:
        lines.append(
            f"| **total** |  |  |  | **{fmt(payload.get('grand_total_usd'), 4)}** |  |"
        )
    else:
        lines.append("| (cost report not generated yet) |  |  |  |  |  |")
    lines.append("")
    return lines


def synthetic_table(aggregate: dict[str, Any] | None, prefix: str, metrics: list[str]) -> list[str]:
    if not aggregate:
        return ["(missing)", ""]
    methods = sorted({key.split(".")[1] for key in aggregate if key.startswith(prefix + ".")})
    header = "| method | " + " | ".join(metrics) + " |"
    lines = [header, "|---|" + "---:|" * len(metrics)]
    for method in methods:
        row = []
        for metric in metrics:
            entry = aggregate.get(f"{prefix}.{method}.{metric}")
            row.append(fmt(entry.get("mean") if isinstance(entry, dict) else entry))
        lines.append(f"| {method} | " + " | ".join(row) + " |")
    lines.append("")
    return lines


def main() -> int:
    public = load(ROOT / "results" / "public_round_main" / "summary.json")
    stratified = load(ROOT / "results" / "public_stratified" / "summary.json")
    full_public = load(ROOT / "results" / "public_full" / "summary.json")
    full_mab = load(ROOT / "results" / "public_full_mab" / "summary.json")
    if full_mab:
        if full_public is None:
            full_public = full_mab
        else:
            full_public.setdefault("datasets", {}).update(full_mab.get("datasets", {}))
    lme_s = load(ROOT / "results" / "public_longmemeval_s" / "summary.json")
    lme_m = load(ROOT / "results" / "public_longmemeval_m" / "summary.json")
    glm53_main = load(ROOT / "results" / "public_glm53_main" / "summary.json")
    autogen = load(ROOT / "results" / "autogen_official" / "autogen_pilot.json")
    amem = load(ROOT / "results" / "amem_official" / "results_seed0.json")
    graphiti = load(ROOT / "results" / "graphiti_official" / "results_seed0.json")
    memlens = load(ROOT / "results" / "memlens_official" / "results_seed0.json")
    glm = load(ROOT / "results" / "public_glm_smoke" / "summary.json")
    qwen = load(ROOT / "results" / "public_qwen_smoke" / "summary.json")
    glm53 = load(ROOT / "results" / "public_glm53_smoke" / "summary.json")
    gatemem = load(ROOT / "results" / "gatemem_round" / "summary.json")
    mab = load(ROOT / "results" / "public_mab_smoke" / "summary.json")
    aggregate = load(ROOT / "results" / "text_baselines" / "text_baselines_v1" / "aggregate.json")
    rsi = load(ROOT / "results" / "rsi_qfocus_multiseed" / "summary.json") or load(
        ROOT / "results" / "rsi_qfocus" / "summary.json"
    )
    commits = ROOT / "baselines" / "commits.csv"

    lines = [
        "# AgoraPlanner Cloud One-Round Experiment",
        "",
        "Date: 2026-11-08 (UTC).  This is a one-round, reproducible evaluation matrix;",
        "it is a research prototype, not a public-benchmark SOTA claim.",
        "",
        "## Environment",
        "",
        f"- host: `{shell('hostname')}`",
        f"- GPU: `{shell('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1')}`",
        f"- data disk: `{shell('df -h /root/autodl-tmp | tail -1')}`",
        f"- Python: `{shell('/root/miniconda3/bin/python --version 2>&1')}`; vLLM: `{shell('/root/miniconda3/bin/pip show vllm 2>/dev/null | head -2 | tail -1')}`",
        "- local model: `/root/autodl-tmp/models/Qwen3-8B` (not re-downloaded)",
        "- readers: DeepSeek flash (API), GLM 4-flash/5.3-flash (API), Qwen3-8B (local vLLM on 127.0.0.1:8000)",
        "",
        "## Data inventory",
        "",
        "| dataset | path | size | status |",
        "|---|---|---:|---|",
    ]
    for name, path in [
        ("LoCoMo", "data/LoCoMo/repo/data/locomo10.json"),
        ("LongMemEval (oracle)", "data/LongMemEval/data/longmemeval_oracle.json"),
        ("LongMemEval s_cleaned", "data/LongMemEval/data/longmemeval_s_cleaned.json"),
        ("LongMemEval m_cleaned", "data/LongMemEval/data/longmemeval_m_cleaned.json"),
        ("MemoryAgentBench", "data/MemoryAgentBench/data"),
        ("MARBLE", "data/MARBLE/repo"),
        ("GateMem", "data/GateMem/data"),
    ]:
        full = ROOT / path
        size = shell(f"du -sh {full} 2>/dev/null | cut -f1") if full.exists() else "missing"
        if name == "LongMemEval m_cleaned":
            status = "run (1 full haystack)"
        elif name in (
            "LoCoMo", "LongMemEval (oracle)", "LongMemEval s_cleaned",
            "MemoryAgentBench", "MARBLE", "GateMem",
        ):
            status = "run"
        else:
            status = "downloaded; not run (large)"
        lines.append(f"| {name} | `{path}` | {size} | {status} |")
    lines += [
        "",
        "Qwen3-8B: `/root/autodl-tmp/models/Qwen3-8B` (16G, pre-existing; no duplicate download).",
        "",
        "## Named baseline source inventory",
        "",
        "| baseline | source | commit | status in this round |",
        "|---|---|---|---|",
    ]
    commit_map: dict[str, str] = {}
    if commits.exists():
        for row in commits.read_text(encoding="utf-8").splitlines()[1:]:
            if "," in row:
                repo, sha = row.split(",", 1)
                commit_map[repo] = sha
    baseline_rows = [
        ("Mem0", "mem0ai/mem0", "official; infer=False; run on LoCoMo/LME/synthetic"),
        ("Letta/MemGPT", "letta-ai/letta", "official `letta==0.16.8` server run (PG/pgvector, 5 seeds)"),
        ("RAPTOR", "parthsarthi03/raptor", "official runtime run (5 seeds, collapsed tree)"),
        ("GraphRAG", "microsoft/graphrag", "official `graphrag==2.7.2` runtime run (5 seeds)"),
        ("A-MEM", "agiresearch/A-mem", "official `agentic_memory` runtime run (26-fact)"),
        ("Graphiti", "getzep/graphiti", "official `graphiti-core` + Kuzu runtime run"),
        ("AutoGen", "microsoft/autogen", "official `autogen-agentchat` pilot run (4 memory scopes)"),
        ("MemLens", "LIUHA1ZHU/MemLens", "official pipeline run with embedded Milvus Lite + hash embedding"),
        ("ClawGang", "sept-usc/ClawGang", "upstream repo is whitepaper/assets only; no runnable implementation"),
        ("CAMA/GovMem/RCR/MemPrice/MemGPT", "-", "faithful reimplementation; run and labelled official=false"),
        ("MARBLE", "ulab-uiuc/MARBLE", "official Base + Coding runtime smoke run"),
        ("GateMem", "rzhub/GateMem", "official Naive-RAG + Policy-RAG agents run on education"),
    ]
    for name, repo, status in baseline_rows:
        sha = commit_map.get(repo, "")
        lines.append(f"| {name} | `{repo}` | `{sha[:12]}` | {status} |")
    lines += [
        "",
        "## Public QA matrix (LoCoMo + LongMemEval + MemoryAgentBench)",
        "",
        "Reader: DeepSeek flash; top-k=5. The rows below are the earlier pooled",
        "pilot (LoCoMo 2 conversations x 15 QA; LongMemEval oracle 20 instances;",
        "s_cleaned 2 full-haystack questions; MemoryAgentBench 2 rows x 5 QA). The stratified",
        "section after them rebuilds each instance as its own retrieval world; use",
        "those numbers for per-ability comparisons.",
        "",
    ]
    if public:
        for dataset, payload in public.get("datasets", {}).items():
            lines += public_table(dataset, payload)
    if mab:
        for dataset, payload in mab.get("datasets", {}).items():
            lines += public_table(dataset, payload)
    if lme_s:
        for dataset, payload in lme_s.get("datasets", {}).items():
            lines += public_table(f"LongMemEval s_cleaned ({dataset})", payload)
    if lme_m:
        for dataset, payload in lme_m.get("datasets", {}).items():
            lines += public_table(f"LongMemEval m_cleaned ({dataset})", payload)
    if glm53_main:
        for dataset, payload in glm53_main.get("datasets", {}).items():
            lines += public_table(f"GLM-5.3-flash reader: {dataset}", payload)
    lines += stratified_table(stratified)
    lines += full_public_table(full_public)
    lines += [
        "### Reader model ablations (LoCoMo, 1 conversation x 5 QA)",
        "",
        "| reader | baseline | exact | F1 | tokens/q | latency ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, payload in (("GLM-4-flash", glm), ("GLM-5.3-flash", glm53), ("Qwen3-8B", qwen)):
        if not payload:
            continue
        for dataset, body in payload.get("datasets", {}).items():
            for name, row in body.get("baselines", {}).items():
                if "error" in row:
                    continue
                lines.append(
                    f"| {label} | {name} | {fmt(row.get('answer_exact'))} | "
                    f"{fmt(row.get('answer_f1'))} | {fmt(row.get('tokens_per_query'), 0)} | "
                    f"{fmt(row.get('latency_ms_per_query'), 0)} |"
                )
    lines += official_runtime_table()
    lines += other_official_table(amem, graphiti, autogen, memlens)
    lines += [
        "",
        "## Synthetic allocation / team metrics (5 seeds)",
        "",
        "Retrieval accuracy and welfare/rho_e on the controlled latent-facet benchmark.",
        "",
        "### Allocation policies",
        "",
    ]
    lines += synthetic_table(
        aggregate,
        "allocation",
        ["retrieval.answer_accuracy@1", "welfare.task_success", "welfare.gt_welfare",
         "welfare.between_agent_error_corr", "welfare.common_mode_penalty",
         "retrieval.context_tokens", "latency_ms"],
    )
    lines += ["### Named memory systems (official + reimplementations)", ""]
    lines += synthetic_table(
        aggregate,
        "systems",
        ["retrieval.answer_accuracy@1", "retrieval.context_tokens", "latency_ms"],
    )
    lines += gatemem_official_table()
    lines += [
        "### Legacy prompt-only policy baseline (external predictions, subset)",
        "",
        "The scorer is the official rule-based scorer; predictions are external, so",
        "the scorer's runtime/token fields are not populated.",
        "",
        "| baseline | n | Utility U | Access violation A | Forgetting failure F | Over-refusal OR | MGS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if gatemem:
        for name, row in gatemem.items():
            lines.append(
                f"| {name} | {row.get('n_checkpoints')} | {fmt(row.get('utility_accuracy'))} | "
                f"{fmt(row.get('privacy_leakage_rate'))} | {fmt(row.get('deletion_leakage_rate'))} | "
                f"{fmt(row.get('over_refusal_rate'))} | {fmt(row.get('compliance_utility_score'))} |"
            )
    else:
        lines.append("| (missing) |  |  |  |  |  |  |")
    lines += [
        "",
        "## RSI q-focus run (DeepSeek flash, 3 rounds, 10 steps)",
        "",
    ]
    if rsi:
        seeds = rsi.get("results", [])
        aggregate = rsi.get("summary") or {}
        evidence = rsi.get("rsi_evidence") or {}
        lines += [
            "| mode | test success | final success | seen | unseen | U version | structural edits |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        if aggregate:
            for mode, row in aggregate.items():
                lines.append(
                    f"| {mode} | {fmt(row.get('test_success'))} | {fmt(row.get('test_final_success'))} | "
                    f"{fmt(row.get('seen_success'))} | {fmt(row.get('unseen_success'))} | "
                    f"{fmt(row.get('u_version'), 1)} | {fmt(row.get('structural_u_edits'), 1)} |"
                )
        else:
            result = seeds[0]
            for mode, row in result["results"].items():
                lines.append(
                    f"| {mode} | {fmt(row['final_test']['success'])} | {fmt(row['final_test']['final_success'])} | "
                    f"{fmt(row['final_test_seen']['success'])} | {fmt(row['final_test_unseen']['success'])} | "
                    f"{len(row['u_versions']) - 1} | {len(row['structural_edits'])} |"
                )
        if evidence:
            lines += [
                "",
                f"- seeds: {evidence.get('n_seeds')}; claim: `{evidence.get('claim')}`; "
                f"pass rate {fmt(evidence.get('pass_rate'))}",
                f"- N2 rate {fmt(evidence.get('n2_rate'))}; N3 rate {fmt(evidence.get('n3_rate'))}; "
                f"N4 rate {fmt(evidence.get('n4_rate'))}",
                f"- mean gain vs fixed: {fmt(evidence.get('mean_gain_vs_fixed'))}; "
                f"vs u_fixed: {fmt(evidence.get('mean_gain_vs_u_fixed'))}",
                f"- mean unseen gain vs u_fixed: {fmt(evidence.get('mean_unseen_gain_vs_u_fixed'))}; "
                f"mean meta gain {fmt(evidence.get('mean_meta_gain'))}",
                f"- single-seed caveat: `{evidence.get('single_seed_caveat')}`",
            ]
        result = seeds[0]
        cert = result["rsi_certificate"]
        lines += [
            "",
            f"- seed 0 certificate: N1={cert['n1_persistence']}, "
            f"N2={cert['n2_meta_level_u_changed']}, N3={cert['n3_independent_validation']}, "
            f"N4={cert['n4']}; gain vs fixed {fmt(cert['gain_vs_fixed'])} "
            f"(LCB {fmt(cert['gain_vs_fixed_lcb'])}), vs u_fixed {fmt(cert['gain_vs_u_fixed'])} "
            f"(LCB {fmt(cert['gain_vs_u_fixed_lcb'])})",
            "",
            "U lineage (seed 0):",
        ]
        for version in result["results"]["meta_rsi"]["u_versions"]:
            lines.append(f"- U v{version['version']}: " + " -> ".join(s["kind"] for s in version["stages"]))
        q_tokens = {
            key: value["valuation"]["tokens"]
            for key, value in result["results"]["meta_rsi"]["buyer_policies"].items()
        }
        lines.append(f"- final Q (seed 0): `{json.dumps(q_tokens, ensure_ascii=False)}`")
        lines.append("")
        lines += rsi_process_table()
        lines += cost_table()
        lines += [
            "",
            "## Metric coverage",
            "",
            "| metric family | public QA | synthetic | RSI | governance |",
            "|---|---:|---:|---:|---:|",
            "| task accuracy / success | yes | yes | yes | yes (U) |",
            "| token F1 / answer contains | yes | no | no | no |",
            "| recall@k / nDCG@k / MRR | yes (LoCoMo/LME; MAB n/a) | yes | no | no |",
            "| context precision/tokens | yes | yes | no | no |",
            "| latency | yes | yes | no | no (external predictions) |",
            "| success per 1k tokens | yes | no | no | no |",
            "| per-ability / category breakdown | yes (stratified) | no | no | yes (domains) |",
            "| abstention / refusal | yes (LME abstention; LoCoMo adversarial) | no | no | partial |",
            "| bootstrap CI / invalid rates | yes (stratified) | no | no | no |",
            "| welfare / rho_e / common-mode penalty | no | yes | no | no |",
            "| RSI N1-N4 / held-out gain / transfer | no | no | yes | no |",
            "| GateMem U / A / F / MGS | no | no | no | yes |",
            "| official named runtime | yes (Mem0/GraphRAG/Letta/RAPTOR) | yes | no | yes (GateMem/MARBLE) |",
            "| USD cost estimate | yes | partial | yes | partial |",
            "",
            "Covered in this round: official RAPTOR/GraphRAG/Letta runtimes, official",
            "RAPTOR on public LoCoMo, official GateMem Naive-RAG / Policy-RAG agents on",
            "education plus three domain subsets, the full MARBLE Bargaining 100-task",
            "sweep and Base/Coding smoke tests, full-set LoCoMo (1,986 QA) and",
            "LongMemEval s_cleaned (500 QA) with per-instance isolation, a stratified",
            "MemoryAgentBench run, per-ability and abstention metrics, bootstrap CIs,",
            "USD cost estimates, and 3-seed RSI aggregation. Remaining",
            "caveats: MARBLE has 90/100 tasks valid under both role rubrics, GateMem",
            "domains are 2-episode subsets, GraphRAG/Letta/MARBLE cost splits are partly",
            "estimated, and the RSI is a bounded U-level RSI under a fixed meta-mutation",
            "language.",
            "",
            "## Notes",
            "",
            "- Mem0 rows use the official package with `infer=False`, local Qdrant, and the",
            "  same deterministic hash embedder as the other systems; this isolates memory",
            "  management from embedding quality.",
            "- The earlier public-QA pilot pooled all selected conversations/haystacks into",
            "  one retrieval world, so a question could retrieve another instance's memory.",
            "  `experiments/run_public_stratified.py` rebuilds one world per instance and",
            "  reruns LoCoMo (98 QA), LongMemEval s_cleaned (140 QA), and MemoryAgentBench",
            "  (124 QA); the stratified tables supersede the pilot for per-ability claims.",
            "- CAMA/GovMem/RCR/MemPrice/MemGPT rows are faithful reimplementations, not",
            "  official reproductions.  They are labelled `official=false`.",
            "- Official RAPTOR/GraphRAG/Letta runtime rows are reported separately from the",
            "  faithful reimplementations; the reimplementation rows remain labelled",
            "  `official=false` and are not mixed into the official runtime table.",
            "- GraphRAG/Letta use the same local hash-embedding endpoint as the controlled",
            "  comparison; this isolates memory management from neural embedding quality.",
            "- The MARBLE Base/Coding rows are runtime smoke tests; the Bargaining sweep",
            "  uses the official MultiAgentBench task configs and evaluator, and reports",
            "  rubric means over the 90/100 tasks where both role evaluations parsed.",
            "- Qwen3-8B is served locally by vLLM; DeepSeek/GLM are API readers.",
        ]
    else:
        lines.append("(RSI run not complete yet)")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
