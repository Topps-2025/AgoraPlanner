"""Quality-metadata ablation for the text-memory reimplementations."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agora_planner.config import DEFAULT_RESULTS_DIR  # noqa: E402
from agora_planner.memory_systems import (  # noqa: E402
    CAMAReimpl,
    GraphRAGReimpl,
    MemPriceReimpl,
    RAPTORReimpl,
    RCRRouterReimpl,
)
from run_text_baselines import _bench, _run_system  # noqa: E402


def main() -> None:
    seeds = [0, 1, 2, 3, 4]
    output_dir = Path(DEFAULT_RESULTS_DIR / "text_ablation_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        bench = _bench(seed)
        systems = {
            "raptor_quality": RAPTORReimpl(branch_facts=4, use_quality=True),
            "raptor_noquality": RAPTORReimpl(branch_facts=4, use_quality=False),
            "graphrag_quality": GraphRAGReimpl(expand_neighbors=3, use_quality=True),
            "graphrag_noquality": GraphRAGReimpl(expand_neighbors=3, use_quality=False),
            "cama_quality": CAMAReimpl(candidate_k=12, use_quality=True),
            "cama_noquality": CAMAReimpl(candidate_k=12, use_quality=False),
            "rcr_quality": RCRRouterReimpl(token_budget=48, use_quality=True),
            "rcr_noquality": RCRRouterReimpl(token_budget=48, use_quality=False),
            "memprice_quality": MemPriceReimpl(target_tokens=48, use_quality=True),
            "memprice_noquality": MemPriceReimpl(target_tokens=48, use_quality=False),
        }
        row = {"seed": seed, "systems": {}}
        for name, system in systems.items():
            row["systems"][name] = _run_system(bench, system, name=name, top_k=3)
        rows.append(row)
        print(f"seed {seed} done", flush=True)
    with open(output_dir / "ablation.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    keys = sorted({f"{name}.{metric}" for row in rows for name in row["systems"] for metric in ["retrieval.answer_accuracy@1", "retrieval.recall@k", "retrieval.decoy_rate@1", "retrieval.context_tokens"]})
    aggregate = {}
    for key in keys:
        name, metric = key.split(".", 1)
        values = []
        for row in rows:
            node = row["systems"][name]["retrieval"]
            metric_key = metric.split(".", 1)[1]
            values.append(float(node[metric_key]))
        aggregate[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    with open(output_dir / "aggregate.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
