"""Stratified public-memory evaluation with per-instance isolation.

The original ``run_public_memory_round`` pools every selected LoCoMo
conversation, LongMemEval haystack, and MemoryAgentBench row into a single
retrieval world, so a question can retrieve evidence from another instance.
This runner builds one world per conversation / haystack / row, samples
questions per ability/category, and writes per-query rows that can be
aggregated by category.

Datasets
--------
LoCoMo       : 10 conversations, QA categories 1-5.
LongMemEval  : 500 questions in 6 question types + 30 abstention ids.
MemoryAgentBench : Accurate_Retrieval, Conflict_Resolution,
                   Long_Range_Understanding, Test_Time_Learning.

Outputs
-------
``per_query_<world>.jsonl`` per instance, ``summary.json`` with overall and
per-category means, and ``report.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in [str(ROOT / "src"), str(ROOT / "experiments")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from agora_planner.memory import MemoryAsset  # noqa: E402
from run_public_memory_round import (  # noqa: E402
    PublicQuery,
    PublicWorld,
    Reader,
    _baseline_names,
    run_world,
)

LOCOMO_CATEGORIES = {
    "1": "cat1_multi_hop",
    "2": "cat2_temporal",
    "3": "cat3_open_domain",
    "4": "cat4_single_hop",
    "5": "cat5_adversarial",
}


# ----------------------------------------------------------------------
# World builders (one world per instance)
# ----------------------------------------------------------------------


def build_locomo_worlds(
    path: str | Path, *, per_category: int = 2, max_conversations: int = 10
) -> list[PublicWorld]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    worlds: list[PublicWorld] = []
    for conv in data[:max_conversations]:
        sample_id = str(conv.get("sample_id", "conv"))
        conversation = conv.get("conversation", {})
        speakers = [
            conversation.get("speaker_a", "speaker_a"),
            conversation.get("speaker_b", "speaker_b"),
        ]
        memories: list[MemoryAsset] = []
        session_keys = sorted(
            [key for key in conversation if re.fullmatch(r"session_\d+", key)],
            key=lambda key: int(key.split("_")[1]),
        )
        turn_index = 0
        for session_key in session_keys:
            date = conversation.get(f"{session_key}_date_time", "")
            for turn in conversation.get(session_key, []):
                dia_id = str(turn.get("dia_id") or f"{session_key}:{turn_index}")
                memories.append(
                    MemoryAsset(
                        id=f"{sample_id}:{dia_id}",
                        content=f"[{date}] {turn.get('speaker', speakers[0])}: {turn.get('text', '')}",
                        facet=session_key,
                        source_class="conversation",
                        quality=0.8,
                        scope="public",
                        version=0,
                        owner=str(turn.get("speaker", speakers[0])),
                        production_cost=0.1,
                        created_round=turn_index,
                        correlation_risk=0.1,
                        metadata={"session_id": session_key, "dia_id": dia_id, "date": date},
                    )
                )
                turn_index += 1

        by_category: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for idx, qa in enumerate(conv.get("qa", [])):
            by_category[str(qa.get("category", ""))].append((idx, qa))
        selected: list[tuple[int, dict[str, Any]]] = []
        for category in sorted(by_category):
            selected.extend(by_category[category][:per_category])
        selected.sort(key=lambda item: item[0])

        queries: list[PublicQuery] = []
        for idx, qa in selected:
            evidence = frozenset(
                f"{sample_id}:{item}" for item in (qa.get("evidence") or [])
            )
            raw_category = str(qa.get("category", ""))
            queries.append(
                PublicQuery(
                    id=f"{sample_id}_q{idx}",
                    question=str(qa.get("question", "")),
                    answer=str(qa.get("answer", "")),
                    evidence_ids=evidence,
                    agent_id=speakers[idx % 2],
                    category=LOCOMO_CATEGORIES.get(raw_category, f"cat{raw_category}"),
                    dataset="locomo",
                )
            )
        if queries:
            worlds.append(
                PublicWorld(
                    name=f"locomo_{sample_id}",
                    memories=memories,
                    queries=queries,
                    notes={"source": str(path), "sample_id": sample_id},
                )
            )
    return worlds


def build_longmemeval_worlds(
    path: str | Path, *, per_type: int = 20
) -> list[PublicWorld]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for inst in data:
        qid = str(inst.get("question_id", "q"))
        qtype = "abstention" if "abs" in qid.lower() else str(inst.get("question_type", "unknown"))
        by_type[qtype].append(inst)

    selected: list[tuple[str, dict[str, Any]]] = []
    for qtype in sorted(by_type):
        selected.extend((qtype, inst) for inst in by_type[qtype][:per_type])

    worlds: list[PublicWorld] = []
    for qtype, inst in selected:
        qid = str(inst.get("question_id", "q"))
        sessions = inst.get("haystack_sessions") or []
        dates = inst.get("haystack_dates") or []
        session_ids = inst.get("haystack_session_ids") or [
            f"{qid}_s{i}" for i in range(len(sessions))
        ]
        memories: list[MemoryAsset] = []
        for si, session in enumerate(sessions):
            session_id = str(session_ids[si]) if si < len(session_ids) else f"{qid}_s{si}"
            date = str(dates[si]) if si < len(dates) else ""
            for ti, turn in enumerate(session):
                memories.append(
                    MemoryAsset(
                        id=f"{qid}:{session_id}:{ti}",
                        content=f"[{date}] {turn.get('role', 'user')}: {turn.get('content', '')}",
                        facet=session_id,
                        source_class="conversation",
                        quality=0.8,
                        scope="public",
                        version=0,
                        owner=str(turn.get("role", "user")),
                        production_cost=0.1,
                        created_round=si * 100 + ti,
                        correlation_risk=0.1,
                        metadata={
                            "session_id": session_id,
                            "date": date,
                            "has_answer": bool(turn.get("has_answer", False)),
                        },
                    )
                )
        evidence_sessions = {str(x) for x in (inst.get("answer_session_ids") or [])}
        query = PublicQuery(
            id=qid,
            question=str(inst.get("question", "")),
            answer=str(inst.get("answer", "")),
            evidence_ids=frozenset(evidence_sessions),
            agent_id="user",
            category=qtype,
            dataset="longmemeval",
        )
        worlds.append(
            PublicWorld(
                name=f"lme_{qid}",
                memories=memories,
                queries=[query],
                notes={"source": str(path), "question_type": qtype},
            )
        )
    return worlds


def _normalise_answer(answer: Any) -> str:
    if hasattr(answer, "tolist"):
        answer = answer.tolist()
    if isinstance(answer, (list, tuple)):
        return str(answer[0]) if answer else ""
    return str(answer)


def build_memoryagentbench_worlds(
    data_dir: str | Path, *, rows_per_competency: int = 4, queries_per_row: int = 10
) -> list[PublicWorld]:
    import pandas as pd

    worlds: list[PublicWorld] = []
    for parquet in sorted(glob.glob(str(Path(data_dir) / "*.parquet"))):
        competency = Path(parquet).stem.split("-")[0]
        frame = pd.read_parquet(parquet)
        for row_index, row in frame.head(rows_per_competency).iterrows():
            lines = [line.strip() for line in str(row["context"]).splitlines() if line.strip()]
            memories = [
                MemoryAsset(
                    id=f"mab_{competency}_{row_index}_{line_index}",
                    content=re.sub(r"^\d+\.\s*", "", line),
                    facet=f"row{row_index}",
                    source_class="fact",
                    quality=0.8,
                    scope="public",
                    version=0,
                    owner="agent",
                    production_cost=0.1,
                    created_round=line_index,
                    correlation_risk=0.1,
                    metadata={"row": int(row_index), "competency": competency},
                )
                for line_index, line in enumerate(lines)
            ]
            questions = list(row["questions"])[:queries_per_row]
            answers = list(row["answers"])[:queries_per_row]
            queries: list[PublicQuery] = []
            for query_index, (question, answer) in enumerate(zip(questions, answers)):
                queries.append(
                    PublicQuery(
                        id=f"mab_{competency}_{row_index}_{query_index}",
                        question=str(question),
                        answer=_normalise_answer(answer),
                        evidence_ids=frozenset(),
                        agent_id="agent",
                        category=competency,
                        dataset="memoryagentbench",
                    )
                )
            if queries:
                worlds.append(
                    PublicWorld(
                        name=f"mab_{competency}_{row_index}",
                        memories=memories,
                        queries=queries,
                        notes={"source": parquet, "competency": competency},
                    )
                )
    return worlds


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values]
    values = [value for value in values if value == value]  # drop NaN
    return sum(values) / len(values) if values else None


REFUSAL_MARKERS = (
    "unknown",
    "not provided",
    "insufficient",
    "no information",
    "not mentioned",
    "cannot",
    "can't",
    "not answerable",
    "unanswerable",
)


def _is_refusal(prediction: str) -> bool:
    lowered = str(prediction).lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _bootstrap_ci(values: Sequence[float], *, seed: int = 13, samples: int = 1000) -> list[float] | None:
    import random

    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    return [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]


def _row_metrics(rows: Sequence[dict[str, Any]], top_k: int) -> dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    n = max(1, len(valid))
    tokens = sum(row.get("total_tokens", 0) for row in valid)
    exact_values = [float(row.get("answer_exact", 0.0)) for row in valid]
    f1_values = [float(row.get("answer_f1", 0.0)) for row in valid]
    return {
        "n_queries": len(valid),
        "errors": len(rows) - len(valid),
        f"recall@{top_k}": _mean(
            row.get(f"recall@{top_k}", 0.0) for row in valid
        ),
        f"context_precision@{top_k}": _mean(
            row.get(f"context_precision@{top_k}", 0.0) for row in valid
        ),
        "mrr": _mean(row.get("mrr", 0.0) for row in valid),
        f"ndcg@{top_k}": _mean(row.get(f"ndcg@{top_k}", 0.0) for row in valid),
        "answer_exact": _mean(exact_values),
        "answer_exact_ci95": _bootstrap_ci(exact_values),
        "answer_f1": _mean(f1_values),
        "answer_f1_ci95": _bootstrap_ci(f1_values),
        "answer_contains": _mean(row.get("answer_contains", 0.0) for row in valid),
        "refusal_rate": _mean(
            1.0 if _is_refusal(row.get("prediction", "")) else 0.0 for row in valid
        ),
        "tokens_per_query": tokens / n,
        "latency_ms_per_query": _mean(row.get("latency_ms", 0.0) for row in valid),
        "success_per_1k_tokens": (_mean(exact_values) or 0.0)
        / max(1e-9, tokens / 1000.0 / n),
    }


def aggregate(dirs: Sequence[Path], *, top_k: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for directory in dirs:
        for path in sorted(directory.glob("per_query_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row.get("dataset"), row.get("query_id"), row.get("baseline"))
                current = seen.get(key)
                if current is None or ("error" in current and "error" not in row):
                    seen[key] = row
    rows = list(seen.values())

    datasets: dict[str, Any] = {}
    for dataset in sorted({row.get("dataset") for row in rows}):
        ds_rows = [row for row in rows if row.get("dataset") == dataset]
        baselines = sorted({row.get("baseline") for row in ds_rows})
        categories = sorted({row.get("category", "") for row in ds_rows})
        prefix = {"longmemeval": "lme_", "memoryagentbench": "mab_"}.get(
            dataset, f"{dataset}_"
        )
        payload: dict[str, Any] = {
            "n_queries": len({row.get("query_id") for row in ds_rows}),
            "n_instances": len(
                {
                    path.name
                    for directory in dirs
                    for path in directory.glob(f"per_query_{prefix}*.jsonl")
                }
            ),
            "baselines": {},
            "per_category": {},
        }
        for baseline in baselines:
            payload["baselines"][baseline] = _row_metrics(
                [row for row in ds_rows if row.get("baseline") == baseline], top_k
            )
        for category in categories:
            cat_rows = [row for row in ds_rows if row.get("category", "") == category]
            payload["per_category"][category] = {
                baseline: _row_metrics(
                    [row for row in cat_rows if row.get("baseline") == baseline], top_k
                )
                for baseline in baselines
            }
        datasets[dataset] = payload
    return {"datasets": datasets, "top_k": top_k}


def write_report(payload: dict[str, Any], path: Path) -> None:
    k = payload["top_k"]
    lines = ["# Stratified public-memory evaluation", ""]
    lines.append("Per-instance isolation; one retrieval world per conversation / haystack / row.")
    lines.append("")
    for dataset, body in payload["datasets"].items():
        lines += [
            f"## {dataset}",
            "",
            f"Queries: {body['n_queries']}; instances: {body['n_instances']}",
            "",
            "| baseline | n | exact | F1 | contains | recall@k | MRR | nDCG@k | tokens/q |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for baseline, metrics in sorted(body["baselines"].items()):
            def fmt(value, digits=3):
                return "n/a" if value is None else f"{value:.{digits}f}"

            lines.append(
                f"| {baseline} | {metrics['n_queries']} | "
                f"{fmt(metrics['answer_exact'])} | {fmt(metrics['answer_f1'])} | "
                f"{fmt(metrics['answer_contains'])} | "
                f"{fmt(metrics[f'recall@{k}'])} | {fmt(metrics['mrr'])} | "
                f"{fmt(metrics[f'ndcg@{k}'])} | "
                f"{fmt(metrics['tokens_per_query'], 0)} |"
            )
        lines += ["", "### Per category", ""]
        for category, by_baseline in sorted(body["per_category"].items()):
            lines.append(f"**{category}**")
            lines.append("")
            lines.append("| baseline | n | exact | F1 | recall@k |")
            lines.append("|---|---:|---:|---:|---:|")
            for baseline, metrics in sorted(by_baseline.items()):
                def fmt(value, digits=3):
                    return "n/a" if value is None else f"{value:.{digits}f}"

                lines.append(
                    f"| {baseline} | {metrics['n_queries']} | "
                    f"{fmt(metrics['answer_exact'])} | {fmt(metrics['answer_f1'])} | "
                    f"{fmt(metrics[f'recall@{k}'])} |"
                )
            lines.append("")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified public-memory evaluation")
    parser.add_argument(
        "--locomo",
        default="/root/autodl-tmp/agora_planner_cli/data/LoCoMo/repo/data/locomo10.json",
    )
    parser.add_argument(
        "--longmemeval",
        default="/root/autodl-tmp/agora_planner_cli/data/LongMemEval/data/longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--memoryagentbench",
        default="/root/autodl-tmp/agora_planner_cli/data/MemoryAgentBench/data",
    )
    parser.add_argument("--datasets", default="locomo,longmemeval,memoryagentbench")
    parser.add_argument("--baselines", default="raw_rag,private,selective,oracle,govmem_reimpl,mem0_shared")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reader", default="deepseek", choices=("deepseek", "glm", "qwen"))
    parser.add_argument("--reader-model", default=None)
    parser.add_argument("--reader-max-tokens", type=int, default=512)
    parser.add_argument("--locomo-per-category", type=int, default=2)
    parser.add_argument("--locomo-max-conversations", type=int, default=10)
    parser.add_argument("--lme-per-type", type=int, default=20)
    parser.add_argument("--world-offset", type=int, default=0)
    parser.add_argument("--world-limit", type=int, default=0, help="0 = all worlds")
    parser.add_argument("--mab-rows-per-competency", type=int, default=4)
    parser.add_argument("--mab-queries-per-row", type=int, default=10)
    parser.add_argument(
        "--aggregate-dirs",
        default="",
        help="Comma-separated directories of per_query_*.jsonl to merge without running.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "public_stratified"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_dirs:
        dirs = [Path(item) for item in args.aggregate_dirs.split(",") if item]
        payload = aggregate(dirs, top_k=args.top_k)
        payload["config"] = vars(args)
        payload["reader"] = {"backend": args.reader, "model": args.reader_model or args.reader}
        (output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        write_report(payload, output_dir / "report.md")
        print(f"Wrote {output_dir / 'summary.json'}")
        return 0
    reader = Reader(
        args.reader, model=args.reader_model, max_tokens=args.reader_max_tokens
    )
    baselines = [name for name in args.baselines.split(",") if name] if args.baselines != "all" else _baseline_names()
    selected_datasets = {name.strip() for name in args.datasets.split(",") if name.strip()}

    worlds: list[PublicWorld] = []
    if "locomo" in selected_datasets:
        worlds += build_locomo_worlds(
            args.locomo,
            per_category=args.locomo_per_category,
            max_conversations=args.locomo_max_conversations,
        )
    if "longmemeval" in selected_datasets:
        worlds += build_longmemeval_worlds(args.longmemeval, per_type=args.lme_per_type)
    if "memoryagentbench" in selected_datasets:
        worlds += build_memoryagentbench_worlds(
            args.memoryagentbench,
            rows_per_competency=args.mab_rows_per_competency,
            queries_per_row=args.mab_queries_per_row,
        )

    if args.world_offset or args.world_limit:
        end = args.world_offset + args.world_limit if args.world_limit else None
        worlds = worlds[args.world_offset:end]
    print(
        f"worlds={len(worlds)} queries={sum(len(w.queries) for w in worlds)} "
        f"baselines={baselines}",
        flush=True,
    )
    for index, world in enumerate(worlds, 1):
        print(
            f"[{index}/{len(worlds)}] {world.name} memories={len(world.memories)} "
            f"queries={len(world.queries)}",
            flush=True,
        )
        run_world(
            world,
            baselines=baselines,
            reader=reader,
            top_k=args.top_k,
            output_dir=output_dir,
        )

    payload = aggregate([output_dir], top_k=args.top_k)
    payload["config"] = vars(args)
    payload["reader"] = {"backend": args.reader, "model": reader.model}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_report(payload, output_dir / "report.md")
    print(f"Wrote {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
