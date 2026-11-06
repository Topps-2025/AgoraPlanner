"""Public-pool vs bilateral-market experiment on a medium-long OLS derivation.

The task is a ten-step derivation of the simple linear regression solution:
model -> loss -> expansion -> partial derivatives -> determinant -> a -> b
-> Hessian -> prediction -> numeric check.

The public pool is a reviewed catalog (review quorum, quality/correlation
admission, source cap, embargo, reputation).  It is NOT priced by bid/ask.
Configurations:

    market      private sellers only
    pool        public pool only
    hybrid      public pool + private sellers
    all_public  unreviewed all-memory pool (negative control)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agora_planner.config import DEFAULT_RESULTS_DIR, LLMConfig  # noqa: E402
from agora_planner.demand_first_rsi import MemoryRecord, QuoteProgram, RPNAST, SellerAgent  # noqa: E402
from agora_planner.llm import DeepSeekClient, _load_dotenv  # noqa: E402
from agora_planner.long_horizon_nl_rsi import (  # noqa: E402
    EpisodeResult,
    LHEval,
    LongHorizonWorld,
    RoundRobinAllocator,
    SubTask,
    TaskGraph,
    apply_allocator,
    evaluate_graphs,
    run_graph,
)
from agora_planner.nl_demand_first_rsi import (  # noqa: E402
    DEFAULT_DEMAND_INSTRUCTION,
    DEFAULT_RETRIEVAL_INSTRUCTION,
    NLBuyerPolicy,
    NLSellerPolicy,
    NLWorkflow,
)
from agora_planner.public_pool import MemoryPool, PoolPolicy  # noqa: E402


# ----------------------------------------------------------------------
# OLS derivation world
# ----------------------------------------------------------------------


def _ols_specs() -> list[tuple[str, str, str, str, str, str, tuple[str, ...], str]]:
    return [
        (
            "s1",
            "Write the per-observation squared residual for y = a*x + b. Give (y - a*x - b)^2.",
            "(y - a*x - b)**2",
            "ols_loss",
            "OLS loss term: (y - a*x - b)^2; source: current regression notes.",
            "OLS loss term: (y + a*x + b)^2; wrong sign.",
            (),
            "sympy",
        ),
        (
            "s2",
            "Expand the loss in terms of n, Sx, Sy, Sxx, Sxy, a, b. Give the expanded formula.",
            "Syy - 2*a*Sxy - 2*b*Sy + a**2*Sxx + 2*a*b*Sx + n*b**2",
            "ols_expand",
            "Expansion: sum (y-a*x-b)^2 = Syy - 2a Sxy - 2b Sy + a^2 Sxx + 2ab Sx + n b^2.",
            "Expansion has cross term -2ab Sx instead of +2ab Sx; wrong sign.",
            ("s1",),
            "sympy",
        ),
        (
            "s3",
            "Differentiate the expanded loss with respect to a. Give dL/da.",
            "2*(a*Sxx + b*Sx - Sxy)",
            "ols_grad_a",
            "dL/da = 2(a Sxx + b Sx - Sxy).",
            "dL/da = 2(a Sxx - b Sx - Sxy); wrong sign on b Sx.",
            ("s2",),
            "sympy",
        ),
        (
            "s4",
            "Differentiate the expanded loss with respect to b. Give dL/db.",
            "2*(a*Sx + n*b - Sy)",
            "ols_grad_b",
            "dL/db = 2(a Sx + n b - Sy).",
            "dL/db = 2(a Sx - n b - Sy); wrong sign on n b.",
            ("s2",),
            "sympy",
        ),
        (
            "s5",
            "The normal-equation matrix is [[Sxx,Sx],[Sx,n]]. Give its determinant D.",
            "n*Sxx - Sx**2",
            "ols_det",
            "Normal-equation determinant D = n Sxx - Sx^2.",
            "Normal-equation determinant D = n Sxx + Sx^2; wrong sign.",
            ("s3", "s4"),
            "sympy",
        ),
        (
            "s6",
            "Solve the normal equations for a. Give the closed form using n, Sx, Sy, Sxx, Sxy.",
            "(n*Sxy - Sx*Sy)/(n*Sxx - Sx**2)",
            "ols_a",
            "Cramer's rule: a = (n Sxy - Sx Sy)/(n Sxx - Sx^2).",
            "Cramer's rule: a = (Sx Sxy - n Sy)/(n Sxx - Sx^2); wrong numerator.",
            ("s5",),
            "sympy",
        ),
        (
            "s7",
            "Solve the normal equations for b. Give the closed form using n, Sx, Sy, Sxx, Sxy.",
            "(Sxx*Sy - Sx*Sxy)/(n*Sxx - Sx**2)",
            "ols_b",
            "Cramer's rule: b = (Sxx Sy - Sx Sxy)/(n Sxx - Sx^2).",
            "Cramer's rule: b = (Sxx Sxy - Sx Sy)/(n Sxx - Sx^2); wrong numerator.",
            ("s5",),
            "sympy",
        ),
        (
            "s8",
            "Give the determinant of the Hessian of the loss (the 2x2 matrix of second derivatives).",
            "4*(n*Sxx - Sx**2)",
            "ols_hessian",
            "Hessian = [[2Sxx,2Sx],[2Sx,2n]]; det = 4(n Sxx - Sx^2).",
            "Hessian determinant = 2(n Sxx - Sx^2); missing factor 4.",
            ("s3", "s4"),
            "sympy",
        ),
        (
            "s9",
            "Using a and b from s6/s7, give the fitted value yhat(x)=a*x+b as one formula.",
            "((n*Sxy - Sx*Sy)*x + (Sxx*Sy - Sx*Sxy))/(n*Sxx - Sx**2)",
            "ols_predict",
            "yhat(x) = a x + b, with a and b from Cramer's rule.",
            "yhat(x) = a x - b; wrong sign on intercept.",
            ("s6", "s7"),
            "sympy",
        ),
        (
            "s10",
            "For x=[1,2], y=[3,5], evaluate yhat(3). Give the number.",
            "7",
            "ols_numeric",
            "n=2, Sx=3, Sy=8, Sxx=5, Sxy=11; a=2, b=1; yhat(3)=7.",
            "n=2, Sx=3, Sy=8, Sxx=5, Sxy=11; a=1, b=2; yhat(3)=5. Wrong values.",
            ("s9",),
            "numeric",
        ),
    ]


def build_ols_world(n_buyers: int = 2) -> LongHorizonWorld:
    graph_id = "ols"
    specs = _ols_specs()
    tasks: list[SubTask] = []
    memories: list[MemoryRecord] = []
    for idx, (sid, question, gold, facet, true_text, decoy_text, deps, verifier) in enumerate(specs):
        true_id = f"{graph_id}_{sid}_true"
        decoy_id = f"{graph_id}_{sid}_decoy"
        seller_true = "sellerA" if idx % 2 == 0 else "sellerB"
        seller_decoy = "sellerB" if seller_true == "sellerA" else "sellerA"
        memories.append(
            MemoryRecord(
                id=true_id,
                seller_id=seller_true,
                content=true_text,
                facet=facet,
                source="internal" if idx % 2 == 0 else "regulator",
                quality=0.92,
                freshness=0.9,
                production_cost=0.10,
                correlation_risk=0.1,
                is_true=True,
                is_decoy=False,
                kind="lemma",
                depends_on=tuple(f"{graph_id}_{d}" for d in deps),
                assumptions=("regular design",),
                valid_scope="current",
            )
        )
        memories.append(
            MemoryRecord(
                id=decoy_id,
                seller_id=seller_decoy,
                content=decoy_text,
                facet=facet,
                source="vendor" if idx % 2 == 0 else "market",
                quality=0.22,
                freshness=0.3,
                production_cost=0.02,
                correlation_risk=0.8,
                is_true=False,
                is_decoy=True,
                kind="lemma",
                depends_on=(),
                assumptions=("stale",),
                valid_scope="superseded",
            )
        )
        tasks.append(
            SubTask(
                id=f"{graph_id}_{sid}",
                question=question,
                gold_answer=gold,
                buyer_id=f"buyer{idx % max(1, n_buyers)}",
                facet=facet,
                required_memory_ids=frozenset({true_id}),
                decoy_memory_ids=frozenset({decoy_id}),
                depends_on=tuple(f"{graph_id}_{d}" for d in deps),
                verifier=verifier,
                answer_format="a mathematical expression or a number, plain text (no LaTeX)",
            )
        )
    graph = TaskGraph(
        id=graph_id,
        subtasks=tuple(tasks),
        final_subtask_id=f"{graph_id}_s10",
    )
    buyers = tuple(f"buyer{i}" for i in range(max(1, n_buyers)))
    graph = apply_allocator(graph, RoundRobinAllocator(), buyers)
    sellers = tuple(
        SellerAgent(sid, tuple(m for m in memories if m.seller_id == sid))
        for sid in ("sellerA", "sellerB")
    )
    # The test graph is the same derivation; this experiment is a fixed-policy
    # pool/market comparison, not a held-out RSI claim.
    return LongHorizonWorld(
        meta_graph=graph,
        test_graph=graph,
        meta_sellers=sellers,
        test_sellers=sellers,
        buyer_ids=buyers,
        pool=None,
    )


def build_pool(memories: Sequence[MemoryRecord], mode: str) -> MemoryPool:
    if mode == "all_public":
        pool = MemoryPool(
            PoolPolicy(
                review_quorum=0,
                review_threshold=0.0,
                quality_threshold=0.0,
                max_correlation=1.0,
                embargo_steps=0,
                source_cap=2,
            )
        )
        for memory in memories:
            entry = pool.publish(memory, step=0, license="public")
            pool.force_approve(entry.memory.id)
        return pool
    pool = MemoryPool(
        PoolPolicy(
            review_quorum=2,
            review_threshold=0.5,
            quality_threshold=0.5,
            max_correlation=0.6,
            embargo_steps=0,
            source_cap=2,
        )
    )
    for memory in memories:
        if not memory.is_true:
            continue
        pool.publish(memory, step=0, license="team")
        pool.review(memory.id, "reviewer1", 0.9)
        pool.review(memory.id, "reviewer2", 0.85)
    return pool


def _memories(world: LongHorizonWorld) -> list[MemoryRecord]:
    seen: dict[str, MemoryRecord] = {}
    for seller in world.meta_sellers:
        for memory in seller.inventory:
            seen[memory.id] = memory
    return list(seen.values())


# ----------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------


def run_config(
    mode: str,
    n_agents: int,
    workflow: NLWorkflow,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    world = build_ols_world(n_agents)
    calls_start = workflow.calls
    failures_start = workflow.failures
    memories = _memories(world)
    pool = None
    sellers = {s.id: s for s in world.meta_sellers}
    seller_policies = {
        s.id: NLSellerPolicy(DEFAULT_RETRIEVAL_INSTRUCTION, top_k=10)
        for s in world.meta_sellers
    }
    if mode in ("pool", "hybrid", "all_public"):
        pool = build_pool(memories, mode)
    if mode == "pool":
        sellers = {}
        seller_policies = {}
    if mode == "all_public":
        sellers = {}
        seller_policies = {}

    buyer_policies = {
        bid: NLBuyerPolicy(
            DEFAULT_DEMAND_INSTRUCTION,
            RPNAST(("llm_utility", "facet_match", "MUL"), version=0, name="seed_llm_facet"),
            QuoteProgram(price_rule="ask"),
        )
        for bid in world.buyer_ids
    }
    evaluation = evaluate_graphs(
        [world.meta_graph], buyer_policies, seller_policies, sellers, workflow, pool=pool
    )
    rows = evaluation.rows
    pool_usage = sum(
        1 for row in rows for offer in row.selected if offer.seller_id == "pool"
    )
    market_usage = sum(
        1 for row in rows for offer in row.selected if offer.seller_id != "pool"
    )
    if pool is not None:
        for row in rows:
            pool_ids = [o.record.id for o in row.selected if o.seller_id == "pool"]
            if pool_ids:
                pool.record_usage(pool_ids, row.success >= 1.0)
    per_agent: dict[str, int] = {bid: 0 for bid in world.buyer_ids}
    for row in rows:
        per_agent[row.buyer_id] = per_agent.get(row.buyer_id, 0) + 1
    return {
        "mode": mode,
        "n_agents": n_agents,
        "seed": seed,
        "success": evaluation.success,
        "final_success": evaluation.final_success,
        "pollution": evaluation.pollution,
        "objective": evaluation.objective,
        "spend": evaluation.spend,
        "n_steps": len(rows),
        "pool_usage": pool_usage,
        "market_usage": market_usage,
        "per_agent_tasks": per_agent,
        "pool_stats": pool.stats() if pool is not None else None,
        "rows": [r.to_dict() for r in rows],
        "llm_calls": workflow.calls - calls_start,
        "llm_failures": workflow.failures - failures_start,
    }


def run(
    modes: Sequence[str],
    agent_counts: Sequence[int],
    workflow: NLWorkflow,
) -> dict[str, Any]:
    results = []
    for mode in modes:
        for n_agents in agent_counts:
            results.append(run_config(mode, n_agents, workflow))
    return {"results": results}


def save(payload: dict[str, Any], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "per_config.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["mode", "n_agents", "success", "final_success", "pollution", "objective", "spend", "pool_usage", "market_usage", "llm_calls", "llm_failures"]
        )
        for r in payload["results"]:
            writer.writerow(
                [
                    r["mode"], r["n_agents"], r["success"], r["final_success"],
                    r["pollution"], r["objective"], r["spend"], r["pool_usage"],
                    r["market_usage"], r["llm_calls"], r["llm_failures"],
                ]
            )
    lines = [
        "# Public pool vs market on a ten-step OLS derivation",
        "",
        "| mode | agents | success | final | pollution | spend | pool usage | market usage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["results"]:
        lines.append(
            f"| {r['mode']} | {r['n_agents']} | {r['success']:.3f} | {r['final_success']:.3f} | "
            f"{r['pollution']:.3f} | {r['spend']:.3f} | {r['pool_usage']} | {r['market_usage']} |"
        )
    lines += ["", "Ten-step OLS derivation; fixed policy; not a public benchmark.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Public pool vs market OLS experiment")
    parser.add_argument("--modes", default="market,pool,hybrid,all_public")
    parser.add_argument("--agent-counts", default="1,2,4")
    parser.add_argument("--max-calls", type=int, default=1200)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR / "pool_math_ols"))
    args = parser.parse_args(argv)
    _load_dotenv()
    if args.offline:
        workflow = NLWorkflow(None, use_llm=False)
    else:
        client = DeepSeekClient(
            LLMConfig(
                model="deepseek-flash",
                max_tokens=2500,
                temperature=0.0,
                max_calls=args.max_calls,
            )
        )
        workflow = NLWorkflow(client, use_llm=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    counts = [int(x) for x in args.agent_counts.split(",") if x.strip()]
    payload = run(modes, counts, workflow)
    payload["settings"] = {
        "modes": modes,
        "agent_counts": counts,
        "model": "offline-mock" if args.offline else "deepseek-flash",
        "dataset": "ten_step_ols",
        "decomposition": "fixed_gold_dag",
        "allocator": "round_robin",
    }
    out = save(payload, args.output_dir)
    print(f"Saved: {out}")
    print(json.dumps(
        [
            {
                "mode": r["mode"], "agents": r["n_agents"],
                "success": round(r["success"], 4), "final": round(r["final_success"], 4),
                "pollution": round(r["pollution"], 4), "spend": round(r["spend"], 4),
                "pool_usage": r["pool_usage"], "market_usage": r["market_usage"],
            }
            for r in payload["results"]
        ],
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
