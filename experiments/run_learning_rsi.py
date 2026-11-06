"""Run the bilateral buyer-seller bounded-RSI experiment.

Buyers learn demand-based valuations and bids; sellers learn cost-based
valuations and asks.  The update rule is a program; the meta-level generates
new programs by structural mutations, validates them on held-out meta-episodes,
and reuses the adopted program.  This is the algorithm-level RSI construction,
not a selection from a fixed library of complete update rules.

Examples
--------
Fast smoke test::

    PYTHONPATH=src python experiments/run_learning_rsi.py \
        --seeds 0,1 --rounds 2 --meta-steps 20 --meta-seeds 2 \
        --candidates 4 --test-seeds 1

Full run (the reported v2 experiment)::

    PYTHONPATH=src python experiments/run_learning_rsi.py \
        --seeds 0,1,2,3,4 --rounds 4 --meta-steps 40 --meta-seeds 4 \
        --candidates 10 --test-steps 40 --test-seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agora_planner.learning_rsi import AgentRSIRunner, LearningRSISettings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgoraPlanner agent-level RSI")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--meta-steps", type=int, default=40)
    parser.add_argument("--meta-seeds", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--test-steps", type=int, default=40)
    parser.add_argument("--test-seeds", type=int, default=5)
    parser.add_argument("--test-shift", type=float, default=0.5)
    parser.add_argument(
        "--modes",
        default=(
            "fixed,meta_structural,meta_param,random_mutation,"
            "self_validated,closed_library"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    default = LearningRSISettings()
    settings = LearningRSISettings(
        seeds=seeds,
        meta_rounds=args.rounds,
        meta_steps=args.meta_steps,
        meta_seeds=args.meta_seeds,
        candidates=args.candidates,
        test_steps=args.test_steps,
        test_seeds=args.test_seeds,
        test_shift=args.test_shift,
        output_dir=args.output_dir or default.output_dir,
    )
    runner = AgentRSIRunner(settings)
    payload = runner.run(modes=modes)
    out = runner.save(payload)
    print(f"Saved: {out}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
