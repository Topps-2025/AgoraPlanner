"""Run the AgoraPlanner v9 protocol-RI experiment.

Examples
--------
Fast smoke test::

    PYTHONPATH=src python experiments/run_protocol_rsi.py \
        --seeds 0,1 --rounds 2 --replay-seeds 2 --candidates 4

Full experiment::

    PYTHONPATH=src python experiments/run_protocol_rsi.py \
        --seeds 0,1,2 --rounds 3 --replay-seeds 3 --candidates 6

The script writes ``summary.json``, ``per_seed.csv``, and ``report.md`` to the
output directory.  All reported test gains are on a held-out seed that is not
used for protocol selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``src/`` importable when the script is run from the repository root.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agora_planner.protocol_rsi import BoundedRIRunner, ExperimentSettings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgoraPlanner v9 protocol-RI")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replay-seeds", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument(
        "--modes",
        default=(
            "fixed,meta_structural,meta_scalar,random_selection,"
            "random_replay,diagnosis_no_replay,self_validated,no_anchor,"
            "closed_library"
        ),
    )
    parser.add_argument("--n-memories", type=int, default=16)
    parser.add_argument("--n-train-tasks", type=int, default=32)
    parser.add_argument("--n-eval-tasks", type=int, default=32)
    parser.add_argument("--redundancy", type=float, default=0.50)
    parser.add_argument("--evaluation-lambda", type=float, default=0.50)
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    default_settings = ExperimentSettings()
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
        output_dir=args.output_dir or default_settings.output_dir,
    )
    runner = BoundedRIRunner(settings)
    payload = runner.run(modes=modes)
    out = runner.save(payload)
    print(f"Saved: {out}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
