"""Command-line entry point for AgoraPlanner.

Examples
--------
Synthetic demo (no API calls)::

    python -m agora_planner.cli demo --mode synthetic --seed 0

Full experiment from a YAML config::

    python -m agora_planner.cli run --config experiments/configs/synthetic_small.yaml

DeepSeek flash smoke test::

    python -m agora_planner.cli test-llm

Small LLM-backed market demo::

    python -m agora_planner.cli demo --mode llm --max-calls 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ExperimentConfig, load_config
from .runner import ExperimentRunner


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def _load(args: argparse.Namespace) -> ExperimentConfig:
    config = load_config(args.config) if getattr(args, "config", None) else ExperimentConfig()
    if getattr(args, "seed", None) is not None:
        config.seeds = [args.seed]
    if getattr(args, "max_calls", None) is not None:
        config.llm.max_calls = args.max_calls
    return config


def cmd_demo(args: argparse.Namespace) -> int:
    config = _load(args)
    if args.mode == "llm":
        config.llm.enabled = True
        runner = ExperimentRunner(config)
        payload = runner.run_llm_demo(max_calls=args.max_calls, seed=args.seed or 0)
        print(json.dumps(
            {
                "mode": payload["mode"],
                "model": payload["model"],
                "llm_usage": payload["llm_usage"],
                "auction": {
                    "welfare": payload["auction"].get("mechanism_welfare"),
                    "gt_welfare": payload["auction"].get("gt_welfare"),
                    "task_success": payload["auction"].get("task_success"),
                    "allocation": payload["auction"].get("allocation"),
                    "prices": payload["auction"].get("prices"),
                },
                "winning_agent": payload["winning_agent"],
                "winning_bundle": payload["winning_bundle"],
                "bid_rationale": payload["bid_rationale"],
                "rsi": payload["rsi"],
            },
            ensure_ascii=False,
            indent=2,
        ))
        print("\nSaved: results/llm_demo_results.json")
        return 0

    # Synthetic demo: one seed, all baselines, market, and RSI.
    runner = ExperimentRunner(config)
    result = runner.run_seed(args.seed or 0)
    print("\n=== Benchmark ===")
    print(json.dumps(result["benchmark"], ensure_ascii=False, indent=2)[:4000])
    print("\n=== Baseline / auction metrics ===")
    for name, summary in result["baselines"].items():
        print(
            f"{name:16s} success={summary['task_success']:.4f} "
            f"gt_welfare={summary['gt_welfare']:.4f} "
            f"redundancy={summary['redundancy_rate']:.3f} "
            f"indep={summary['independence_index']:.3f} "
            f"err_corr={summary['between_agent_error_corr']:.3f} "
            f"mem={summary['memory_count']}"
        )
    summary = result["auction"]
    print(
        f"{'auction':16s} success={summary['task_success']:.4f} "
        f"gt_welfare={summary['gt_welfare']:.4f} "
        f"redundancy={summary['redundancy_rate']:.3f} "
        f"indep={summary['independence_index']:.3f} "
        f"err_corr={summary['between_agent_error_corr']:.3f} "
        f"mem={summary['memory_count']}"
    )
    if "rsi" in result:
        print("\n=== Market-grounded RSI ===")
        for row in result["rsi"]["rounds"]:
            print(
                f"round {row['round']}: accepted={row['accepted']} "
                f"op={row['op']} gain={row['mean_gain']:.4f} "
                f"lcb={row['lcb']:.4f} mem={row['memory_count']}"
            )
    if "rsi_self_validated" in result:
        print("\n=== Self-validated RSI baseline ===")
        for row in result["rsi_self_validated"]["rounds"]:
            print(
                f"round {row['round']}: accepted={row['accepted']} "
                f"op={row['op']} heldout_gain={row['mean_gain']:.4f} "
                f"mem={row['memory_count']}"
            )
    path = runner.save_json(result, f"demo_seed{args.seed or 0}.json")
    print(f"\nSaved: {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _load(args)
    runner = ExperimentRunner(config)
    payload = runner.run()
    print(f"\nSaved: {runner.output_dir / (config.name + '_results.json')}")
    print(f"Saved: {runner.output_dir / (config.name + '_summary.csv')}")
    if payload.get("llm_usage"):
        print("LLM usage:", payload["llm_usage"])
    return 0


def cmd_test_llm(args: argparse.Namespace) -> int:
    config = _load(args)
    config.llm.enabled = True
    runner = ExperimentRunner(config)
    result = runner.test_llm()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


def cmd_protocol_rsi(args: argparse.Namespace) -> int:
    """Run the v9 bounded-RI protocol experiment."""

    from .protocol_rsi import BoundedRIRunner, ExperimentSettings

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    default = ExperimentSettings()
    settings = ExperimentSettings(
        seeds=seeds,
        meta_rounds=args.rounds,
        replay_seeds=args.replay_seeds,
        candidates_per_round=args.candidates,
        n_memories=args.n_memories,
        n_train_tasks=args.n_train_tasks,
        n_eval_tasks=args.n_eval_tasks,
        redundancy_fraction=args.redundancy,
        evaluation_common_mode_lambda=args.evaluation_lambda,
        output_dir=args.output_dir or default.output_dir,
    )
    runner = BoundedRIRunner(settings)
    payload = runner.run(modes=modes)
    out = runner.save(payload)
    print(f"Saved: {out}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agora-planner",
        description="AgoraPlanner: auction-based memory asset planning",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run a small demo")
    demo.add_argument("--mode", choices=["synthetic", "llm"], default="synthetic")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--max-calls", type=int, default=8)
    demo.add_argument("--config", type=str, default=None)
    demo.set_defaults(func=cmd_demo)

    run = sub.add_parser("run", help="run a full experiment from a config")
    run.add_argument("--config", type=str, required=True)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--max-calls", type=int, default=None)
    run.set_defaults(func=cmd_run)

    test = sub.add_parser("test-llm", help="test the DeepSeek endpoint")
    test.add_argument("--config", type=str, default=None)
    test.set_defaults(func=cmd_test_llm)

    proto = sub.add_parser(
        "protocol-rsi",
        help="run the v9 bounded-RI protocol experiment",
    )
    proto.add_argument("--seeds", default="0,1,2")
    proto.add_argument("--rounds", type=int, default=3)
    proto.add_argument("--replay-seeds", type=int, default=3)
    proto.add_argument("--candidates", type=int, default=6)
    proto.add_argument(
        "--modes",
        default=(
            "fixed,meta_structural,meta_scalar,random_selection,"
            "random_replay,diagnosis_no_replay,self_validated,no_anchor,"
            "closed_library"
        ),
    )
    proto.add_argument("--n-memories", type=int, default=16)
    proto.add_argument("--n-train-tasks", type=int, default=32)
    proto.add_argument("--n-eval-tasks", type=int, default=32)
    proto.add_argument("--redundancy", type=float, default=0.50)
    proto.add_argument("--evaluation-lambda", type=float, default=0.50)
    proto.add_argument("--output-dir", default=None)
    proto.set_defaults(func=cmd_protocol_rsi)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
