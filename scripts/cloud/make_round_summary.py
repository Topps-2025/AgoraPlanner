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
    glm = load(ROOT / "results" / "public_glm_smoke" / "summary.json")
    qwen = load(ROOT / "results" / "public_qwen_smoke" / "summary.json")
    glm53 = load(ROOT / "results" / "public_glm53_smoke" / "summary.json")
    gatemem = load(ROOT / "results" / "gatemem_round" / "summary.json")
    mab = load(ROOT / "results" / "public_mab_smoke" / "summary.json")
    aggregate = load(ROOT / "results" / "text_baselines" / "text_baselines_v1" / "aggregate.json")
    rsi = load(ROOT / "results" / "rsi_qfocus" / "summary.json")
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
        ("MemoryAgentBench", "data/MemoryAgentBench/data"),
        ("MARBLE", "data/MARBLE/repo"),
        ("GateMem", "data/GateMem/data"),
    ]:
        full = ROOT / path
        size = shell(f"du -sh {full} 2>/dev/null | cut -f1") if full.exists() else "missing"
        status = "run" if name in ("LoCoMo", "LongMemEval (oracle)", "MemoryAgentBench", "GateMem") else "downloaded; adapter not run"
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
        ("Letta/MemGPT", "letta-ai/letta", "official source cloned; runtime not run (heavy)"),
        ("RAPTOR", "parthsarthi03/raptor", "official source cloned; reimpl used in this round"),
        ("GraphRAG", "microsoft/graphrag", "official source cloned; reimpl used in this round"),
        ("A-MEM", "agiresearch/A-mem", "official source cloned; not run"),
        ("Graphiti", "getzep/graphiti", "official source cloned; not run"),
        ("AutoGen", "microsoft/autogen", "official source cloned; not run"),
        ("MemLens", "LIUHA1ZHU/MemLens", "official source cloned; not run"),
        ("ClawGang", "sept-usc/ClawGang", "official source cloned; not run"),
        ("CAMA/GovMem/RCR/MemPrice/MemGPT", "-", "faithful reimplementation; run and labelled official=false"),
        ("MARBLE", "ulab-uiuc/MARBLE", "official source cloned; not run"),
        ("GateMem", "rzhub/GateMem", "official source + data downloaded; not run"),
    ]
    for name, repo, status in baseline_rows:
        sha = commit_map.get(repo, "")
        lines.append(f"| {name} | `{repo}` | `{sha[:12]}` | {status} |")
    lines += [
        "",
        "## Public QA matrix (LoCoMo + LongMemEval + MemoryAgentBench)",
        "",
        "Reader: DeepSeek flash (`deepseek-chat`); top-k=5; LoCoMo 2 conversations x 15 QA;",
        "LongMemEval oracle 20 instances; MemoryAgentBench 2 rows x 5 QA (no evidence labels).",
        "",
    ]
    if public:
        for dataset, payload in public.get("datasets", {}).items():
            lines += public_table(dataset, payload)
    if mab:
        for dataset, payload in mab.get("datasets", {}).items():
            lines += public_table(dataset, payload)
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
    lines += [
        "## GateMem governance (official scorer; education, 2 episodes / 40 checkpoints)",
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
        result = rsi["results"][0]
        cert = result["rsi_certificate"]
        lines += [
            "| mode | test success | final success | seen | unseen | U version | structural edits |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, row in result["results"].items():
            lines.append(
                f"| {mode} | {fmt(row['final_test']['success'])} | {fmt(row['final_test']['final_success'])} | "
                f"{fmt(row['final_test_seen']['success'])} | {fmt(row['final_test_unseen']['success'])} | "
                f"{len(row['u_versions']) - 1} | {len(row['structural_edits'])} |"
            )
        lines += [
            "",
            f"- certificate passed: `{cert['passed']}`; N1={cert['n1_persistence']}, "
            f"N2={cert['n2_meta_level_u_changed']}, N3={cert['n3_independent_validation']}, N4={cert['n4']}",
            f"- gain vs fixed: {fmt(cert['gain_vs_fixed'])} (LCB {fmt(cert['gain_vs_fixed_lcb'])})",
            f"- gain vs u_fixed: {fmt(cert['gain_vs_u_fixed'])} (LCB {fmt(cert['gain_vs_u_fixed_lcb'])})",
            f"- unseen gain vs u_fixed: {fmt(cert['unseen_gain_vs_u_fixed'])} (LCB {fmt(cert['unseen_gain_vs_u_fixed_lcb'])})",
            f"- meta gains: {cert['meta_gains']}",
            "",
            "U lineage:",
        ]
        for version in result["results"]["meta_rsi"]["u_versions"]:
            lines.append(f"- U v{version['version']}: " + " -> ".join(s["kind"] for s in version["stages"]))
        q_tokens = {
            key: value["valuation"]["tokens"]
            for key, value in result["results"]["meta_rsi"]["buyer_policies"].items()
        }
        lines.append(f"- final Q: `{json.dumps(q_tokens, ensure_ascii=False)}`")
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
            "",
            "Not covered in this round: MARBLE task KPIs, the official GateMem",
            "RAG-Policy / Naive-RAG agents (we report a prompt-only policy baseline",
            "with the official scorer), official RAPTOR/GraphRAG/Letta runtime",
            "integration, and USD cost (tokens/latency are reported instead).",
            "",
            "## Notes",
            "",
            "- Mem0 rows use the official package with `infer=False`, local Qdrant, and the",
            "  same deterministic hash embedder as the other systems; this isolates memory",
            "  management from embedding quality.",
            "- CAMA/GovMem/RCR/MemPrice/MemGPT rows are faithful reimplementations, not",
            "  official reproductions.  They are labelled `official=false`.",
            "- Official RAPTOR/GraphRAG/Letta source is cloned, but the one-round harness uses",
            "  the reimplementations for a controlled comparison; official integration is",
            "  the next step.",
            "- Qwen3-8B is served locally by vLLM; DeepSeek/GLM are API readers.",
        ]
    else:
        lines.append("(RSI run not complete yet)")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
