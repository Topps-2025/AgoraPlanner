# Experiments

All scripts are runnable from the repository root. The base install is enough
for the offline/symbolic experiments; the live scripts need `DEEPSEEK_API_KEY`.

```bash
pip install -e ".[dev]"
```

## Task-collaboration RSI (current main prototype)

```bash
# Live DeepSeek run: fixed policy, fixed update program, and evolving U.
python experiments/run_task_collab_rsi.py \
  --seeds 0 --rounds 4 --steps 10 \
  --modes fixed,u_fixed,meta_rsi \
  --max-calls 3500 \
  --output-dir results/task_collab_n4

# Ablations: parameter-only U, self-validated U, no counterfactual probes,
# surplus-gated acceptance, random structural edits.
python experiments/run_task_collab_rsi.py \
  --seeds 0 --rounds 4 --steps 10 \
  --modes meta_rsi,meta_param,self_validated,no_probe,surplus_gate,random_meta \
  --output-dir results/task_collab_n4_controls

# Forced-Q diagnostic: stale decoy passes provenance/quality/budget filters;
# the buyer Q must be learned from per-memory task-completion gains.
python experiments/run_task_collab_rsi.py \
  --seeds 0 --rounds 4 --steps 10 \
  --modes fixed,u_fixed,meta_rsi \
  --seed-q constant --q-focus --fit-episodes 1 \
  --output-dir results/task_collab_n4_qforced

# Offline smoke test (no API key).
python experiments/run_task_collab_rsi.py --offline --steps 6 --rounds 1
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--seeds 0,1,2` | Multiple seeds |
| `--rounds 4` | Meta-RSI rounds |
| `--steps 10` | Sub-steps per long-horizon episode |
| `--meta-gate strict\|mean\|lcb` | U-level acceptance rule |
| `--gate-signal success\|surplus` | Inner acceptance signal (ablation) |
| `--seed-q llm_utility\|constant` | Seed buyer valuation (constant forces Q learning) |
| `--q-focus` | Forced-Q world: decoy passes filters + scoped retrieval |
| `--fit-episodes 1` | Use one fit episode (faster diagnostic) |
| `--offline` | Deterministic fallbacks, no API calls |

## Long-horizon / multi-hop demand-first RSI

```bash
python experiments/run_long_horizon_nl_rsi.py \
  --seeds 0 --modes fixed,joint --max-calls 700 \
  --output-dir results/long_horizon_nl_rsi
```

## Single-hop natural-language demand-first RSI

```bash
python experiments/run_nl_demand_first_rsi.py \
  --seeds 0 --rounds 1 --meta-tasks 4 --test-tasks 4 \
  --dataset controlled --max-calls 700 \
  --output-dir results/nl_demand_first_rsi
```

## Symbolic demand-first RSI

```bash
python experiments/run_demand_first_rsi.py \
  --seeds 0,1,2 --rounds 3 --meta-tasks 6 --test-tasks 6 --candidates 8 \
  --output-dir results/demand_first_rsi
```

## Bounded RSI prototypes

```bash
# Agent-level update-program RSI (synthetic two-sided market)
python experiments/run_learning_rsi.py --seeds 0,1,2 --rounds 4

# Protocol-level bounded RI
python experiments/run_protocol_rsi.py --seeds 0,1,2 --rounds 3

# Real-LLM bilateral market RSI
python experiments/run_llm_bilateral_rsi.py --seeds 0 --rounds 4 --steps 6
```

## Baselines and text-memory benchmark

```bash
python experiments/run_baselines.py --seeds 0,1,2
python experiments/run_text_baselines.py --seeds 0,1,2
python experiments/run_text_ablation.py
```

## Public-pool / market diagnostic

```bash
python experiments/run_pool_math_experiment.py \
  --modes market,pool,hybrid,all_public --agent-counts 1,2,4
```

## Outputs

Every runner writes a `summary.json`, `per_seed.csv`, and `report.md` under the
requested output directory. The default root is `~/.agora_planner/results`
(override with `AGORA_DATA_ROOT` or `--output-dir`).

## Official named-baseline runtimes

```bash
# RAPTOR (official repo on PYTHONPATH or RAPTOR_REPO)
RAPTOR_REPO=/path/to/raptor python experiments/run_raptor_official.py

# GraphRAG 2.7.2 (install it into an isolated target; local hash embeddings)
PYTHONPATH=/path/to/graphrag:src python experiments/run_graphrag_official.py \
  --seed 0 --local --output-root results/graphrag_official

# Letta/MemGPT 0.16.8 REST server + embedded PostgreSQL/pgvector
DEEPSEEK_API_KEY=... bash scripts/cloud/start_letta_stack.sh
python experiments/run_letta_official.py

# MARBLE Base / Coding smoke (official MARBLE checkout + its dependencies)
PYTHONPATH=/path/to/MARBLE python -m marble.main \
  --config_path experiments/marble_pilot/base_config.yaml

# A-MEM (official agentic_memory; hash embedding + DeepSeek JSON mode)
AMEM_REPO=/path/to/a-mem python experiments/run_amem_official.py --seed 0

# Graphiti (official graphiti-core + embedded Kuzu)
GRAPHITI_REPO=/path/to/graphiti python experiments/run_graphiti_official.py \
  --seed 0 --max-facts 26 --max-queries 24

# MemLens (official pipeline + embedded Milvus Lite)
MEMLENS_REPO=/path/to/MemLens python experiments/run_memlens_official.py --seed 0

# AutoGen orchestration pilot
PYTHONPATH=/path/to/autogen python experiments/run_autogen_pilot.py \
  --seed 0 --n-queries 5 --top-k 3
```

GraphRAG, Letta, MARBLE, A-MEM, Graphiti, MemLens, and AutoGen need their
upstream packages installed; the base install deliberately does not vendor them.

## Public memory benchmarks

```bash
# Original pooled pilot (kept for reference; use the stratified runner for
# per-ability numbers).
python experiments/run_public_memory_round.py \
  --datasets locomo,longmemeval,memoryagentbench \
  --max-conversations 2 --max-queries 15 --max-instances 20 \
  --mab-rows 2 --mab-queries 5 --top-k 5 --reader deepseek \
  --baselines all --output-dir results/public_round

# Per-instance stratified run: one retrieval world per conversation /
# LongMemEval haystack / MemoryAgentBench row, sampled per category.
python experiments/run_public_stratified.py \
  --datasets locomo,longmemeval,memoryagentbench \
  --locomo-per-category 2 --lme-per-type 20 \
  --mab-rows-per-competency 4 --mab-queries-per-row 10 \
  --baselines raw_rag,private,selective,oracle,govmem_reimpl,mem0_shared \
  --output-dir results/public_stratified

# Merge per-query rows from several shards without rerunning.
python experiments/run_public_stratified.py \
  --aggregate-dirs results/public_stratified_locomo,results/public_stratified_lme \
  --output-dir results/public_stratified
```

`run_public_stratified.py` writes `per_query_<world>.jsonl`, `summary.json`
(overall and per-category means, bootstrap 95% CIs, refusal rate), and
`report.md`. MemoryAgentBench has no released evidence ids, so recall is N/A.

## Optional vendor adapters

The public repository includes `mem0_adapter.py` and `autogen_adapter.py`, but
does not vendor their third-party packages. Install `mem0ai` / `autogen` in your
environment if you want to run those adapters. The core tests skip them when the
packages are absent.
