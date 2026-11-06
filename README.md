# AgoraPlanner

AgoraPlanner is a research prototype for **market-based memory allocation** and
**task-collaboration-grounded recursive self-improvement (RSI)** in multi-agent
systems.

The central question is not "which memory is relevant?" but "which memory should
an agent acquire, from whom, at what cost, and how should the procurement policy
itself improve from downstream task outcomes?" The code treats memory as an
asset: buyers write demands, sellers expose scoped offers, a transaction bridges
the two sides, and the final signal is **verified task completion**, not trade
surplus.

> Status: research prototype, not production software. The current live
> experiments are single-seed and use a controlled long-horizon diagnostic, not
> a public benchmark. See [Known limitations](#known-limitations).

## What is in this repository

| Line | Modules | Purpose |
|---|---|---|
| Memory market | `memory.py`, `benchmark.py`, `auction.py`, `valuation.py`, `agents.py`, `agent_market.py` | Synthetic latent-facet memory market with ground-truth values, allocators, and pricing |
| Demand-first procurement | `demand_first_rsi.py`, `nl_demand_first_rsi.py`, `long_horizon_nl_rsi.py` | Buyer demand contracts, seller-scoped retrieval, symbolic valuation (RPN), long-horizon task graphs |
| Task-collaboration RSI | `task_collab_rsi.py` | Bounded U-level RSI: the update program itself is structurally edited and validated on held-out task families |
| Team commons | `public_pool.py` | Reviewed, embargoed, source-capped team memory pool |
| Protocol / agent RSI | `protocol_rsi.py`, `learning_rsi.py`, `llm_bilateral_rsi.py` | Earlier bounded-RSI prototypes at the protocol and agent levels |
| Baselines / diagnostics | `baselines.py`, `memory_systems.py`, `text_benchmark.py`, `embeddings.py` | Baseline allocators, memory-system reimplementations, and a text-memory benchmark |
| Optional adapters | `mem0_adapter.py`, `autogen_adapter.py` | Vendor integrations with lazy imports; not required for the core experiments |

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Python 3.10+ is required. The base dependencies are `numpy`, `PyYAML`, and
`requests`; `pytest` is the only dev dependency. Optional vendor adapters import
their packages lazily.

## Configuration

Copy `.env.example` to `.env` and fill in the key:

```text
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
# Optional: data/results directory. Default: ~/.agora_planner/results
# AGORA_DATA_ROOT=/path/to/agora-data
```

`AGORA_DATA_ROOT` is the canonical data override. If it is unset, the code uses
`AGORA_RESULTS_DIR` or `~/.agora_planner/results`, so a fresh checkout never
writes into the source tree.

## Quick start

### Offline smoke test

```bash
python examples/quickstart.py
```

This builds the long-horizon world and runs one offline round with deterministic
fallbacks. It prints the per-mode summary; no API key is required.

### Live task-collaboration RSI

```bash
python experiments/run_task_collab_rsi.py \
  --seeds 0 --rounds 4 --steps 10 \
  --modes fixed,u_fixed,meta_rsi \
  --max-calls 3500 \
  --output-dir results/task_collab_n4
```

The runner writes `summary.json`, `per_seed.csv`, `report.md`, and
`u_lineage.json` under the output directory.

### Forced-Q diagnostic

```bash
python experiments/run_task_collab_rsi.py \
  --seeds 0 --rounds 4 --steps 10 \
  --modes fixed,u_fixed,meta_rsi \
  --seed-q constant --q-focus --fit-episodes 1 \
  --max-calls 3000 \
  --output-dir results/task_collab_n4_qforced
```

In this world the stale decoy is cheap, high-quality, high-provenance, and
low-freshness, so provenance/quality/budget filters cannot remove it. The seed
buyer Q is constant; the task-completion signal must learn a Q that separates
true from stale memory. This is the live evidence for the joint demand/Q update.

### Other experiments

```bash
# Long-horizon demand-first RSI diagnostic
python experiments/run_long_horizon_nl_rsi.py --seeds 0 --modes fixed,joint

# Single-hop natural-language demand-first RSI
python experiments/run_nl_demand_first_rsi.py --seeds 0 --dataset controlled

# Synthetic market baselines
python experiments/run_baselines.py --seeds 0,1,2

# Text-memory benchmark
python experiments/run_text_baselines.py --seeds 0
```

See `experiments/README.md` for the full command list.

## Core interfaces

All public objects are importable from their modules; the package itself keeps
`__init__` light.

### Configuration and LLM

| Interface | Module | Purpose |
|---|---|---|
| `LLMConfig` | `agora_planner.config` | Model, endpoint, token budget, cache path |
| `BenchmarkConfig` | `agora_planner.config` | Synthetic benchmark parameters |
| `DeepSeekClient` | `agora_planner.llm` | OpenAI-compatible chat client with on-disk cache, retries, token accounting |
| `NLWorkflow` | `agora_planner.nl_demand_first_rsi` | Prompt layer: `demand`, `retrieve`, `value`, `answer_general`; deterministic fallbacks |

### Memory, demand, and market

| Interface | Module | Purpose |
|---|---|---|
| `MemoryRecord` | `agora_planner.demand_first_rsi` | A memory asset (facet, source, quality, cost, provenance, dependencies) |
| `MemoryOffer` | `agora_planner.demand_first_rsi` | Seller offer over a memory with ask/relevance |
| `SellerAgent` | `agora_planner.demand_first_rsi` | A seller with an inventory |
| `DemandContract` | `agora_planner.demand_first_rsi` | Buyer demand (facets, budget, bundle/source limits, provenance/quality floors) |
| `DemandOp` | `agora_planner.demand_first_rsi` | Task-relative demand operator |
| `RPNAST` | `agora_planner.demand_first_rsi` | Symbolic valuation expression over fixed factors |
| `QuoteProgram` | `agora_planner.demand_first_rsi` | Quote/bridge policy (shading, markup, price rule) |
| `select_offers` | `agora_planner.demand_first_rsi` | Budget-aware bundle selection |

### Policies

| Interface | Module | Purpose |
|---|---|---|
| `NLBuyerPolicy` | `agora_planner.nl_demand_first_rsi` | Demand instruction, demand ops, valuation RPN, quote program, version/lineage |
| `NLSellerPolicy` | `agora_planner.nl_demand_first_rsi` | Retrieval instruction, top-k, quality floor, seller valuation RPN |
| `SellerRPNMutator` | `agora_planner.nl_demand_first_rsi` | Structural edits over seller valuation RPN |

### Task-collaboration RSI (current main prototype)

| Interface | Module | Purpose |
|---|---|---|
| `TaskCollabRSIRunner` | `agora_planner.task_collab_rsi` | Main runner: traces, policy updates, meta-RSI, live tests, certificate |
| `UpdateProgram` | `agora_planner.task_collab_rsi` | The editable update procedure U (typed stage sequence) |
| `UpdateStage` | `agora_planner.task_collab_rsi` | One typed stage (`calibrate_q`, `repair_demand`, `repair_seller`, ...) |
| `MetaMutator` | `agora_planner.task_collab_rsi` | Structural proposals over U |
| `meta_step` | `agora_planner.task_collab_rsi` | Held-out validation and adoption of a U edit |
| `run_update_program` | `agora_planner.task_collab_rsi` | Execute U on a policy state |
| `build_world_split` | `agora_planner.task_collab_rsi` | Fit/gate/validation/test family split |
| `collect_trace_bundle` | `agora_planner.task_collab_rsi` | Counterfactual per-memory task-completion probes |
| `run_episode_live` | `agora_planner.task_collab_rsi` | End-to-end long-horizon rollout with the policy's own context |

The CLI entry point is `agora-task-collab-rsi` (installed by `pip install -e .`),
or `python experiments/run_task_collab_rsi.py`.

## How the task-collaboration signal works

For every step `s` and every seller offer `o`, the runner runs the real reader
LLM once per offer under a gold prior context and verifies the answer:

```text
gain_s(o) = success(reader with o) - success(reader without o)
```

A ridge regression fits `Q(o) = w · phi(o)` on these measured gains and compiles
it to an RPN valuation AST. Candidate policy edits are then evaluated by replay
on held-out traces, and candidate **update programs U** are evaluated by the
task completion of the policies they produce on a held-out family. Trade prices
only decide whether an offer can be acquired; they are not the acceptance
objective.

The update program is a typed stage sequence. The meta-mutator inserts, deletes,
replaces, or swaps stages; the meta-controller adopts a candidate U only if it
improves verified task completion on held-out episodes.

## Known limitations

- **Bounded RSI, not open-ended RSI.** The stage library and the meta-mutation
  language are fixed; the meta-procedure is not edited by the system. N5
  (autonomy over the evaluator/objective) is not claimed.
- **Controlled diagnostic, not a public benchmark.** The live task is a
  10-step, multi-family memory-procurement chain, not GAIA / MultiAgentBench /
  LongMemEval.
- **Single-seed live evidence.** The live DeepSeek run is seed 0; a three-seed
  mechanism replication is scripted, not live. API latency prevented additional
  live seeds in the first campaign.
- **Q update is task-gated.** In the LLM-utility-seed run the seller floor hid
  the stale decoy, so `calibrate_q` was rejected (correctly: Q had no marginal
  task-completion effect). A separate forced-Q diagnostic
  (`--q-focus --seed-q constant`) makes the stale decoy cheap, high-quality,
  high-provenance, and low-freshness; `calibrate_q(single_gain)` is then
  accepted and Q learns freshness / source-correlation / utility weights.
  Reported live result: fixed 0.00, u_fixed 0.50, meta_rsi 1.00.
- **N4 is cumulative capability + cross-family transfer**, not accelerating
  compounding; the meta-gain slope is negative in the reported run.
- **Text policies are fixed.** Demand instruction text and seller retrieval
  instruction are not edited in the main run because their effect is not
  replayable without new LLM traces.
- **No internal research docs.** The Chinese weekly reports, working memory, and
  experiment archives are deliberately excluded from this repository.

## Repository layout

```text
agora-planner/
├── src/agora_planner/        # core package
├── tests/                    # pytest suite
├── experiments/              # runnable experiment entry points + configs
├── examples/quickstart.py    # offline smoke example
├── pyproject.toml
├── requirements.txt
└── README.md
```

## License

No license has been chosen yet. Treat the code as an all-rights-reserved
research prototype until a license is added.
