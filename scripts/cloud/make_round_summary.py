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
    lines += [
        "",
        "### MARBLE official runtime smoke",
        "",
        "| scenario | coordination | result | planning score | tokens |",
        "|---|---|---|---:|---:|",
    ]
    base_path = ROOT / "results" / "marble_pilot" / "base_result.jsonl"
    if base_path.exists():
        row = json.loads(base_path.read_text(encoding="utf-8").splitlines()[-1])
        base_summary = row.get("summary") or (
            row.get("iterations", [{}])[-1].get("summary", "")
        )
        lines.append(
            f"| Base (official) | star, 1 agent | `{base_summary}` | "
            f"{row.get('planning_scores', ['n/a'])[-1]} | {row.get('token_usage', 0)} |"
        )
    coding_path = ROOT / "results" / "marble_pilot" / "coding_result.jsonl"
    workspace = ROOT / "results" / "marble_pilot" / "workspace" / "solution.py"
    if coding_path.exists():
        row = json.loads(coding_path.read_text(encoding="utf-8").splitlines()[-1])
        valid = "valid artifact" if workspace.exists() else "no artifact"
        lines.append(
            f"| Coding (official environment) | graph, 2 agents | `{valid}` | "
            f"{row.get('planning_scores', ['n/a'])[-1]} | {row.get('token_usage', 0)} |"
        )
    lines += [
        "",
        "The Coding row is a custom small task executed by the official CodingEnvironment,",
        "not a MultiAgentBench official score; the planner JSON parse failed after the",
        "solution was written (planning score -1).",
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
    if graphiti:
        lines.append(
            f"| Graphiti | official `graphiti-core` + Kuzu | {graphiti.get('n_queries')} QA / "
            f"{graphiti.get('n_facts')} facts | {fmt(graphiti.get('accuracy'))} | "
            f"{fmt(graphiti.get('context_recall'))} | FTS adapter for Kuzu |"
        )
    else:
        lines.append("| Graphiti | official `graphiti-core` + Kuzu | n/a | n/a | n/a | not completed |")
    if memlens:
        lines.append(
            f"| MemLens | official MemLens pipeline + Milvus Lite | {len(memlens.get('rows', []))} QA / "
            f"{memlens.get('n_facts', 26)} facts | {fmt(memlens.get('accuracy'))} | "
            f"{fmt(memlens.get('context_recall'))} | hierarchy/Shapley + hash embedding |"
        )
    else:
        lines.append("| MemLens | official pipeline + Milvus Lite | n/a | n/a | n/a | not completed |")
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
        "## GateMem official agents (education domain, official rule-based scorer)",
        "",
        "| agent | checkpoints | Utility U | Access violation A | Forgetting failure F | Over-refusal OR | MGS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    base = ROOT / "results" / "gatemem_official"
    rows = 0
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
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
    lines.append("")
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
        status = "run" if name in (
            "LoCoMo", "LongMemEval (oracle)", "LongMemEval s_cleaned",
            "MemoryAgentBench", "MARBLE", "GateMem",
        ) else "downloaded; not run (large)"
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
        "Reader: DeepSeek flash; top-k=5; LoCoMo 2 conversations x 15 QA;",
        "LongMemEval oracle 20 instances; LongMemEval s_cleaned 2 full haystacks;",
        "MemoryAgentBench 2 rows x 5 QA (no evidence labels).",
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
        lines += cost_table()
        lines += [
            "",
            "## Metric coverage",
            "",
            "| metric family | public QA | synthetic | RSI | governance |",
            "|---|---:|---:|---:|---:|",
            "| task accuracy / success | yes | yes | yes | yes (U) |",
            "| token F1 / answer contains | yes | no | no | no |",
            "| recall@k / nDCG@k / MRR | yes (LoCoMo/LME) | yes | no | no |",
            "| context precision/tokens | yes | yes | no | no |",
            "| latency | yes | yes | no | no (external predictions) |",
            "| success per 1k tokens | yes | no | no | no |",
            "| welfare / rho_e / common-mode penalty | no | yes | no | no |",
            "| RSI N1-N4 / held-out gain / transfer | no | no | yes | no |",
            "| GateMem U / A / F / MGS | no | no | no | yes |",
            "| official named runtime | yes (Mem0/GraphRAG/Letta/RAPTOR) | yes | no | yes (GateMem/MARBLE) |",
            "| USD cost estimate | yes | partial | yes | partial |",
            "",
            "Covered in this round: official RAPTOR/GraphRAG/Letta runtimes, official",
            "GateMem Naive-RAG / Policy-RAG agents, MARBLE Base + Coding runtime smoke,",
            "LongMemEval s_cleaned, USD cost estimates, and 3-seed RSI aggregation.",
            "Remaining caveats: MARBLE is a smoke test (custom small Coding task),",
            "GraphRAG/Letta cost rows are token estimates, and the RSI is a bounded",
            "U-level RSI under a fixed meta-mutation language.",
            "",
            "## Notes",
            "",
            "- Mem0 rows use the official package with `infer=False`, local Qdrant, and the",
            "  same deterministic hash embedder as the other systems; this isolates memory",
            "  management from embedding quality.",
            "- CAMA/GovMem/RCR/MemPrice/MemGPT rows are faithful reimplementations, not",
            "  official reproductions.  They are labelled `official=false`.",
            "- Official RAPTOR/GraphRAG/Letta runtime rows are reported separately from the",
            "  faithful reimplementations; the reimplementation rows remain labelled",
            "  `official=false` and are not mixed into the official runtime table.",
            "- GraphRAG/Letta use the same local hash-embedding endpoint as the controlled",
            "  comparison; this isolates memory management from neural embedding quality.",
            "- MARBLE results are runtime smoke tests, not official MultiAgentBench scores.",
            "- Qwen3-8B is served locally by vLLM; DeepSeek/GLM are API readers.",
        ]
    else:
        lines.append("(RSI run not complete yet)")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
