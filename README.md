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

> Status: research prototype, not production software. The live RSI evidence is
> now a 3-seed run on a controlled long-horizon diagnostic; the public QA matrix
> is a one-round pilot plus a per-instance stratified LoCoMo / LongMemEval /
> MemoryAgentBench run, with official runtime checks for RAPTOR/GraphRAG/Letta,
> official GateMem agents on four domains, RAPTOR on public LoCoMo, and the full
> official MARBLE MultiAgentBench Bargaining sweep. See
> [Known limitations](#known-limitations).

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

## Cloud one-round evaluation

The `scripts/cloud/` and `experiments/run_public_memory_round.py` /
`experiments/run_gatemem_round.py` entry points implement a one-round public
evaluation matrix for a Linux GPU box:

- **Datasets**: LoCoMo, LongMemEval (oracle and s_cleaned), MemoryAgentBench,
  MARBLE, GateMem; Qwen3-8B is served locally by vLLM.
- **Named baselines**: Raw-RAG, private/shared/selective allocation, official
  Mem0 (`infer=False`, local Qdrant, hash embeddings), **official RAPTOR,
  GraphRAG, and Letta/MemGPT runtimes**, official GateMem Naive-RAG and
  Policy-RAG agents on education plus three domain subsets, official RAPTOR on
  public LoCoMo, the full official MARBLE MultiAgentBench Bargaining sweep and
  Base/Coding runtime smoke tests, and faithful reimplementations of
  CAMA/GovMem/RCR-Router/MemPrice/MemGPT (labelled `official=false`).
- **Metrics**: answer exact/F1, recall@k, nDCG@k, MRR, context precision,
  tokens/query, latency/query, success per 1k tokens; synthetic welfare,
  between-agent error correlation, common-mode penalty; GateMem Utility /
  Access-Control / Forgetting / MGS; MARBLE buyer/seller effectiveness,
  progress, interaction and planning scores; RSI N1-N4 / held-out gain /
  transfer; measured/estimated USD cost.

```bash
# Public QA matrix (LoCoMo + LongMemEval + MemoryAgentBench)
python experiments/run_public_memory_round.py   --datasets locomo,longmemeval --max-conversations 2 --max-queries 15   --max-instances 20 --top-k 5 --reader deepseek   --baselines all --output-dir results/public_round

# Stratified per-instance run (one retrieval world per conversation/haystack/row)
python experiments/run_public_stratified.py \
  --datasets locomo,longmemeval,memoryagentbench \
  --locomo-per-category 2 --lme-per-type 20 \
  --mab-rows-per-competency 4 --mab-queries-per-row 10 \
  --baselines raw_rag,private,selective,oracle,govmem_reimpl,mem0_shared \
  --output-dir results/public_stratified

# GateMem governance with the official scorer
python experiments/run_gatemem_round.py --max-episodes 2 --max-checkpoints 40   --reader deepseek --output-dir results/gatemem_round

# Synthetic allocation/team metrics and RSI q-focus (3 seeds)
python experiments/run_text_baselines.py --seeds 0,1,2,3,4 --top-k 3
python experiments/run_task_collab_rsi.py --seeds 0,1,2 --rounds 3 --steps 10   --modes fixed,u_fixed,meta_rsi --seed-q constant --q-focus --fit-episodes 1

# Official RAPTOR runtime (requires the official repo on PYTHONPATH)
RAPTOR_REPO=/path/to/raptor python experiments/run_raptor_official.py

# Official GraphRAG 2.7.2 runtime (install graphrag into an isolated path)
PYTHONPATH=/path/to/graphrag:src python experiments/run_graphrag_official.py   --seed 0 --local --output-root results/graphrag_official

# Official Letta 0.16.8 server + embedded PostgreSQL/pgvector
DEEPSEEK_API_KEY=... bash scripts/cloud/start_letta_stack.sh
python experiments/run_letta_official.py

# Official GateMem agents (from the GateMem repo)
python bench/scripts/run_eval.py --data_dir bench/data/education   --agent rag_policy --retrieval_backend tfidf   --llm_provider deepseek --llm_model deepseek-flash   --out_dir results/gatemem_official --run_name education_rag_policy
```

The cloud helper expects `AGORA_SSH_HOST`, `AGORA_SSH_PORT`, `AGORA_SSH_USER`,
and `AGORA_SSH_PASSWORD` in the environment; the password is never stored in the
repository.  Data and results live under `/root/autodl-tmp/agora_planner_cli/`
by default (override with `AGORA_CLOUD_ROOT`).

> One-round status: official Mem0, RAPTOR, GraphRAG, Letta/MemGPT, GateMem
> agents, RAPTOR on LoCoMo, and the official MARBLE Bargaining sweep (100 tasks)
> are integrated.  The public QA numbers are a pilot, not SOTA claims.

### One-round results (2026-11-08, 3-seed RSI / pilot QA)

Selected numbers from `RESULTS_SUMMARY_2026-11-08.md` (DeepSeek flash reader,
top-k=5; official systems labelled; reimplementations labelled `official=false`):

| setting | metric | value |
|---|---|---:|
| LoCoMo (30 QA) | oracle answer F1 | 0.494 |
| LoCoMo (30 QA) | Raw-RAG answer F1 | 0.250 |
| LoCoMo (30 QA) | Mem0 answer F1 | 0.276 |
| LongMemEval oracle (20 QA) | private answer F1 | 0.477 |
| LongMemEval oracle (20 QA) | oracle answer F1 | 0.659 |
| LongMemEval oracle (20 QA) | Mem0 answer F1 | 0.225 |
| LongMemEval s_cleaned (2 questions, full haystack) | raw-RAG / oracle answer F1 | 1.000 / 1.000 |
| LongMemEval m_cleaned (1 question, full haystack) | raw-RAG / oracle answer F1 | 0.783 / 0.667 |
| GLM-5.3-flash LoCoMo (30 QA, 13 baselines) | raw-RAG / oracle / Mem0 F1 | .304 / .446 / .295 |
| GLM-5.3-flash LongMemEval (20 QA, 13 baselines) | raw-RAG / oracle / Mem0 F1 | .161 / .236 / .128 |
| GateMem official Naive-RAG (540 ckpt) | U / A / F / MGS | .050 / .033 / .044 / .046 |
| GateMem official Policy-RAG (540 ckpt) | U / A / F / MGS | .061 / .039 / .028 / .057 |
| GateMem Naive-RAG, first 2 episodes (medical 56 / household 48 / office 64) | U / A / F / MGS | med .050/.056/.000/.047; hh .312/.125/.062/.256; off .500/.150/.115/.376 |
| GateMem Policy-RAG, same subsets | U / A / F / MGS | med .000/.056/.000/.000; hh .250/.125/.000/.219; off .389/.200/.038/.299 |
| synthetic 5-seed | selective task success / welfare / rho_e | .530 / 10.055 / .067 |
| synthetic 5-seed | shared_all task success / welfare / rho_e | .676 / -14.536 / -.004 |
| RSI q-focus (3 seeds, 3 rounds) | fixed / u_fixed / meta_rsi success | 0.00 / 0.50 / 1.00 |
| RSI q-focus (3 seeds) | N1 / N2 / N3 / N4 pass rate | 1.00 / 1.00 / 1.00 / 1.00 |
| official RAPTOR (5 seeds) | 26-fact accuracy | 1.000 ± 0.000 |
| official RAPTOR on public LoCoMo (30 QA) | exact / F1 / contains | 0.000 / 0.150 / 0.200 |
| official GraphRAG basic (5 seeds) | 26-fact accuracy / context recall | 0.633 ± 0.090 / 1.000 |
| official Letta/MemGPT (5 seeds) | 26-fact accuracy | 0.633 ± 0.340 |
| official A-MEM (26 facts) | accuracy / context recall | 0.542 / 1.000 |
| official Graphiti + Kuzu (26 facts) | accuracy / context recall | 0.458 / 0.583 |
| official MemLens + Milvus Lite (26 facts) | accuracy / context recall | 0.792 / 1.000 |
| official AutoGen (private / shared / central / GovMem) | accuracy | .800 / 1.000 / .800 / .600 |
| official MARBLE Base | final summary / planning / tokens | BANANA / 5 / 619 |
| official MARBLE Coding env | valid solution.py / tokens | yes / 1546 |
| official MARBLE Bargaining task 1 | buyer / seller rubric (E/P/I), planning | 1/1/1; 1/1/1; 1.000 |
| official MARBLE Bargaining, 100 tasks | buyer E/P/I; seller E/P/I; planning | 1.467/1.333/1.378; 1.433/1.300/1.344; 2.763 |
| official MARBLE Bargaining, 100 tasks | valid tasks / tokens per task | 90/100; 136,138 |
| stratified LoCoMo (98 QA, 10 conv, per-instance) | raw-RAG / oracle / Mem0 F1 | .132 / .369 / .112 |
| stratified LongMemEval s_cleaned (140 QA, 7 types) | raw-RAG / oracle / Mem0 F1 | .329 / .393 / .275 |
| stratified MemoryAgentBench (124 QA, 4 competencies) | raw-RAG / oracle / Mem0 F1 | .220 / .218 / .180 (Mem0 84 QA) |
| stratified LongMemEval | single-session-user / multi-session oracle F1 | .805 / .272 |
| stratified LongMemEval | knowledge-update / temporal / abstention oracle F1 | .578 / .229 / .020 |
| stratified MemoryAgentBench | Accurate Retrieval / Conflict Resolution raw F1 | .659 / .008 |
| full LoCoMo (1,986 QA, 10 conv, per-instance) | raw-RAG / oracle / Mem0 / BGE-small dense F1 | .234 / .415 / .187 / .263 |
| full LoCoMo oracle | single-hop / multi-hop / temporal / adversarial F1 | .659 / .484 / .343 / .001 |
| full LongMemEval s_cleaned (500 QA) | raw-RAG / oracle / selective / BGE-small dense F1 | .380 / .490 / .338 / .361 |
| full LongMemEval oracle | knowledge-update / multi-session / temporal / abstention F1 | .662 / .397 / .365 / .016 |
| full MemoryAgentBench (3,671 QA, 146 rows) | raw-RAG / oracle F1 | .403 / .402 |
| full MemoryAgentBench raw-RAG | Accurate Retrieval / Conflict Resolution / Long Range / TTL F1 | .647 / .210 / .086 / .000 |
| API cost estimate | all one-round components | ~USD 8.22 |

These are one-round numbers and are not public-benchmark SOTA claims.  The
stratified rows use one retrieval world per conversation / haystack / row and
sample questions per ability/category: LoCoMo first 2 QA per category per
conversation (98 QA), LongMemEval first 20 per question type plus 20 abstention
ids (140 QA), MemoryAgentBench first 4 rows per competency x 10 QA (124 QA).
They report bootstrap 95% CIs; the LME abstention and LoCoMo adversarial rows
also serve as abstention checks.  The earlier pooled pilot had a retrieval
isolation bug (all selected conversations/haystacks shared one world), so use
the per-instance tables.  The full-set rows cover all released LoCoMo QA, all
500 LongMemEval s_cleaned questions, and all 3,671 MemoryAgentBench questions;
Mem0 covers all 1,986 LoCoMo QA after per-conversation reruns. The dense row
is a BGE-small bi-encoder added as an embedding-sensitivity check, not an
official system. The first-4-rows stratified MemoryAgentBench sample
understated Conflict Resolution (.008) relative to the full set (.210), so use
the full row for that competency.  GateMem domain rows use the first two episodes of each domain, and MARBLE
rubric means cover the 90/100 tasks where both role evaluations parsed.

## Known limitations

- **Bounded RSI, not open-ended RSI.** The stage library and the meta-mutation
  language are fixed; the meta-procedure is not edited by the system. N5
  (autonomy over the evaluator/objective) is not claimed.
- **Controlled RSI task + one-round public pilot.** The RSI live task is a
  10-step, multi-family memory-procurement chain. Public QA now includes the
  full LoCoMo set (1,986 QA), the full LongMemEval s_cleaned set (500 QA), and
  the full MemoryAgentBench set (3,671 QA; raw-RAG/oracle, with a 124-QA
  stratified run for the other baselines). It is not a SOTA claim.
  MemoryAgentBench has no released per-question evidence ids, so its retrieval
  recall is N/A; its Test-Time Learning rows are zero and Long-Range
  Understanding is low for every method with this top-k reader.
- **Three-seed live evidence, not a large-scale study.** The live DeepSeek RSI
  run is seeds 0-2; the public QA matrix is a one-round pilot on small query
  subsets. It is mechanism evidence, not a public-benchmark SOTA claim.
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
- **GateMem scope.** The governance table uses the official GateMem scorer and
  the official Naive-RAG / Policy-RAG agents on the full education domain (540
  checkpoints) plus the first two episodes of the medical, household, and office
  domains (168 checkpoints total), not a full four-domain sweep. The legacy
  prompt-only subset is kept as a separate line.
- **Official runtime scope.** Official RAPTOR/GraphRAG/Letta/A-MEM/Graphiti/
  MemLens are run on the 26-fact text-memory diagnostic with hash embeddings;
  RAPTOR is additionally run on 30 public LoCoMo QA. AutoGen is an orchestration
  pilot. MARBLE runs the full official MultiAgentBench Bargaining sweep (100
  tasks; 90 have both role rubrics parsed) plus Base/Coding smoke tests; the
  sweep is not a leaderboard submission. ClawGang's upstream repository contains
  only a whitepaper and assets, so it has no runnable baseline.
- **Hash embeddings handicap the named systems.** The official memory baselines
  use the same deterministic hash embedder to isolate memory management from
  embedding quality. A BGE-small dense retriever is added on the full sets as an
  embedding-sensitivity check (LoCoMo F1 .263 vs .234 raw-RAG; LongMemEval .361
  vs .380 raw-RAG), but the official systems themselves were not rerun with
  neural embeddings, so absolute numbers are not comparable to their papers.
- **No internal research docs.** The Chinese weekly reports, working memory, and
  experiment archives are deliberately excluded from this repository.

## Repository layout

```text
agora-planner/
├── src/agora_planner/        # core package
├── tests/                    # pytest suite
├── experiments/              # runnable experiment entry points + configs
├── scripts/cloud/            # SSH/data download/round-summary helpers
├── examples/quickstart.py    # offline smoke example
├── pyproject.toml
├── requirements.txt
└── README.md
```

## License

No license has been chosen yet. Treat the code as an all-rights-reserved
research prototype until a license is added.
