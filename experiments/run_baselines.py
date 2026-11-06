"""Reproduce the baseline suite on the synthetic memory-allocation benchmark.

This script runs every baseline in ``agora_planner.baselines`` plus the
ground-truth and noisy-bid VCG auctions, aggregates metrics over seeds, and
writes ``summary.json``, ``per_seed.csv``, and ``report.md``.

The Mem0 / MemGPT / RAPTOR / GraphRAG entries are *offline policy proxies* on
the synthetic benchmark, not official reproductions.  Official systems need
their own datasets and LLM APIs; they will be handled by separate adapters.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agora_planner.auction import MemoryAuction  # noqa: E402
from agora_planner.baselines import baseline_allocations, baseline_metadata  # noqa: E402
from agora_planner.benchmark import MemoryBenchmark  # noqa: E402
from agora_planner.config import DEFAULT_RESULTS_DIR, load_config  # noqa: E402
from agora_planner.metrics import summarize_allocation, summarize_auction  # noqa: E402
from agora_planner.valuation import GroundTruthValuator, NoisyBidder  # noqa: E402


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgoraPlanner baseline reproduction")
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "configs"
            / "synthetic_fresh.yaml"
        ),
    )
    parser.add_argument("--seeds", default=None, help="comma-separated seed override")
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    seeds = (
        [int(x) for x in args.seeds.split(",") if x.strip()]
        if args.seeds
        else list(config.seeds)
    )
    output_dir = Path(args.output_dir or DEFAULT_RESULTS_DIR / "baselines_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        bench = MemoryBenchmark(replace(config.benchmark, seed=int(seed)))
        allocations = baseline_allocations(
            bench, seed=int(seed), include_proxies=True, include_oracle=True
        )

        # Add the two market variants (ours) for reference.
        gt_auction = MemoryAuction(
            bench.memories,
            bench.agents,
            GroundTruthValuator(bench),
            bench.config,
        )
        gt_result = gt_auction.run(split="eval")
        allocations["auction_gt"] = gt_result.allocation

        noisy_auction = MemoryAuction(
            bench.memories,
            bench.agents,
            NoisyBidder(bench, bench.config, seed=int(seed)),
            bench.config,
        )
        noisy_result = noisy_auction.run(split="eval")
        allocations["auction_noisy"] = noisy_result.allocation

        for method, allocation in allocations.items():
            if method == "auction_gt":
                summary = summarize_auction(bench, gt_result)
            elif method == "auction_noisy":
                summary = summarize_auction(bench, noisy_result)
            else:
                summary = summarize_allocation(bench, allocation)
            row = {"method": method, "seed": int(seed)}
            row.update(
                {
                    "task_success": float(summary["task_success"]),
                    "gt_welfare": float(summary["gt_welfare"]),
                    "gt_min_utility": float(summary["gt_min_utility"]),
                    "memory_count": float(summary["memory_count"]),
                    "redundancy_rate": float(summary["redundancy_rate"]),
                    "zero_price_rate": float(summary["zero_price_rate"]),
                    "independence_index": float(summary["independence_index"]),
                    "between_agent_error_corr": float(
                        summary["between_agent_error_corr"]
                    ),
                    "total_cost": float(summary["total_cost"]),
                }
            )
            rows.append(row)

    methods = sorted({row["method"] for row in rows})
    aggregate: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate[method] = {}
        for key in [
            "task_success",
            "gt_welfare",
            "gt_min_utility",
            "memory_count",
            "redundancy_rate",
            "zero_price_rate",
            "independence_index",
            "between_agent_error_corr",
            "total_cost",
        ]:
            values = [float(row[key]) for row in method_rows]
            aggregate[method][key] = {
                "mean": _mean(values),
                "std": _std(values),
                "values": [round(v, 6) for v in values],
            }
        aggregate[method]["n_seeds"] = len(method_rows)

    payload = {
        "config": config.to_dict(),
        "seeds": seeds,
        "metadata": baseline_metadata(),
        "aggregate": aggregate,
        "per_seed": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        keys = [
            "method",
            "seed",
            "task_success",
            "gt_welfare",
            "gt_min_utility",
            "memory_count",
            "redundancy_rate",
            "zero_price_rate",
            "independence_index",
            "between_agent_error_corr",
            "total_cost",
        ]
        writer.writerow(keys)
        for row in rows:
            writer.writerow([row[key] for key in keys])
    (output_dir / "report.md").write_text(
        _render_report(payload), encoding="utf-8"
    )
    print(f"Saved: {output_dir}")
    print(_render_console(aggregate))
    return 0


def _render_console(aggregate: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"{'method':22s} {'success':>9s} {'welfare':>10s} {'minutil':>9s} {'mem':>6s} {'red':>7s} {'indep':>7s} {'errcorr':>8s}"
    ]
    for method, metrics in sorted(
        aggregate.items(), key=lambda kv: kv[1]["gt_welfare"]["mean"], reverse=True
    ):
        lines.append(
            f"{method:22s} "
            f"{metrics['task_success']['mean']:9.4f} "
            f"{metrics['gt_welfare']['mean']:10.4f} "
            f"{metrics['gt_min_utility']['mean']:9.4f} "
            f"{metrics['memory_count']['mean']:6.2f} "
            f"{metrics['redundancy_rate']['mean']:7.3f} "
            f"{metrics['independence_index']['mean']:7.3f} "
            f"{metrics['between_agent_error_corr']['mean']:8.4f}"
        )
    return "\n".join(lines)


def _render_report(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    metadata = payload["metadata"]
    lines = [
        "# Baseline reproduction (synthetic memory-allocation benchmark)",
        "",
        f"Seeds: {payload['seeds']}.  The Mem0 / MemGPT / RAPTOR / GraphRAG entries",
        "are offline policy proxies, not official reproductions.",
        "",
        "| method | type | official | success | welfare | min util | mem | redundancy | independence | err corr |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in sorted(
        aggregate.items(), key=lambda kv: kv[1]["gt_welfare"]["mean"], reverse=True
    ):
        meta = metadata.get(method, {"type": "method", "official": "n/a"})
        lines.append(
            f"| {method} | {meta.get('type','')} | {meta.get('official','')} | "
            f"{metrics['task_success']['mean']:.4f} | "
            f"{metrics['gt_welfare']['mean']:.4f} | "
            f"{metrics['gt_min_utility']['mean']:.4f} | "
            f"{metrics['memory_count']['mean']:.2f} | "
            f"{metrics['redundancy_rate']['mean']:.3f} | "
            f"{metrics['independence_index']['mean']:.3f} | "
            f"{metrics['between_agent_error_corr']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "* `shared_all` is intentionally unconstrained (over-sharing lower bound).",
            "* `central_oracle` / `oracle` use ground-truth values; `central_reported`",
            "  uses the same noisy bids as the auction.",
            "* `composition` is a proxy for RCR + CAMA + GovMem + MemPrice.",
            "* `mem0_like` / `memgpt_like` / `raptor_like` / `graphrag_like` are",
            "  policy proxies on the synthetic benchmark; official reproduction",
            "  requires the upstream code, datasets, and LLM APIs.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
