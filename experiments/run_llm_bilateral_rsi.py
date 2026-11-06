"""Run the LLM-backed bilateral buyer--seller bounded-RSI experiment.

Buyers use DeepSeek to value a memory from their demand; sellers use DeepSeek
to estimate reservation cost.  The inner loop updates valuation/quote
parameters from realized outcomes.  The outer loop generates new update
programs (including LLM prompt revisions), validates them on held-out
meta-episodes, and versions/reuses the adopted program.

Live run (requires DEEPSEEK_API_KEY in .env or the environment)::

    PYTHONPATH=src python experiments/run_llm_bilateral_rsi.py \
        --seeds 0 --rounds 4 --steps 6 --meta-seeds 5 --candidates 7 \
        --test-seeds 4 --model deepseek-chat \
        --output-dir results/llm_bilateral_rsi_v1

Offline smoke test (deterministic mock, no API calls)::

    PYTHONPATH=src python experiments/run_llm_bilateral_rsi.py --offline \
        --seeds 0 --rounds 1 --steps 3 --meta-seeds 1 --candidates 2 \
        --test-seeds 1 --output-dir results/llm_bilateral_rsi_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agora_planner.config import LLMConfig  # noqa: E402
from agora_planner.llm import DeepSeekClient  # noqa: E402
from agora_planner.llm_bilateral_rsi import (  # noqa: E402
    LLMBilateralClient,
    LLMBilateralRSIRunner,
    LLMRSISettings,
)


class OfflineClient:
    """A deterministic stand-in for the DeepSeek JSON client."""

    available = True

    def json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        text = messages[-1]["content"]
        if "buyer agent" in text:
            # A simple content-sensitive value: longer, more facet-rich
            # memories get a higher score.
            score = min(9.0, 3.0 + 0.15 * len(text) % 6.0)
            return {
                "semantic_value": round(score, 2),
                "confidence": 0.6,
                "matched_facets": ["compliance", "supply", "risk"],
                "redundancy_risk": 0.2,
                "rationale": "offline mock",
            }
        if "seller agent" in text:
            return {
                "cost_estimate": 1.2,
                "scarcity": 0.3,
                "quality": 0.6,
                "rationale": "offline mock",
            }
        return {"addendum": "Prefer specific evidence and discount stale sources."}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM bilateral buyer-seller bounded RSI")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--meta-seeds", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=7)
    parser.add_argument("--test-steps", type=int, default=6)
    parser.add_argument("--test-seeds", type=int, default=4)
    parser.add_argument("--test-shift", type=float, default=0.3)
    parser.add_argument(
        "--modes",
        default="fixed,meta_structural",
    )
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--max-calls", type=int, default=7000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    default = LLMRSISettings()
    settings = LLMRSISettings(
        seeds=seeds,
        meta_rounds=args.rounds,
        meta_steps=args.steps,
        meta_seeds=args.meta_seeds,
        candidates=args.candidates,
        test_steps=args.test_steps,
        test_seeds=args.test_seeds,
        test_shift=args.test_shift,
        model=args.model,
        max_llm_tokens=args.max_tokens,
        max_llm_calls=args.max_calls,
        output_dir=args.output_dir or default.output_dir,
    )
    if args.offline:
        llm = LLMBilateralClient(
            OfflineClient(), max_tokens=args.max_tokens, temperature=0.0
        )
    else:
        client = DeepSeekClient(
            LLMConfig(
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=0.0,
                max_calls=args.max_calls,
            )
        )
        llm = LLMBilateralClient(
            client, max_tokens=args.max_tokens, temperature=0.0
        )
    runner = LLMBilateralRSIRunner(settings, llm=llm)
    payload = runner.run(modes=modes)
    out = runner.save(payload)
    print(f"Saved: {out}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["llm_usage"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
