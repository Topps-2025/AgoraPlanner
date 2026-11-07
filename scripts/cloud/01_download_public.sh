#!/usr/bin/env bash
# Download public datasets and named baseline source tarballs on the cloud box.
# GitHub git clone is unreliable here; codeload.github.com works, and HF is
# reached through hf-mirror.com. Qwen3-8B already exists; never re-download it.
set -u

ROOT="${ROOT:-/root/autodl-tmp/agora_planner_cli}"
DATA="$ROOT/data"
BASE="$ROOT/baselines"
LOG="$ROOT/logs/download_public.log"
mkdir -p "$DATA" "$BASE" "$ROOT/logs"

echo "[$(date -u +%FT%TZ)] start download" | tee -a "$LOG"
df -h "$ROOT" | tail -1 | tee -a "$LOG"

fetch() {
  local url="$1" out="$2"
  if [ -s "$out" ]; then
    echo "[skip] $out" | tee -a "$LOG"
    return 0
  fi
  echo "[get ] $url" | tee -a "$LOG"
  mkdir -p "$(dirname "$out")"
  if curl -L --fail --retry 3 --retry-delay 3 --max-time 1800 -o "$out.part" "$url"; then
    mv "$out.part" "$out"
    echo "[ok  ] $(du -h "$out" | cut -f1) $out" | tee -a "$LOG"
  else
    rm -f "$out.part"
    echo "[FAIL] $url" | tee -a "$LOG"
  fi
}

fetch_repo() {
  local owner_repo="$1" name="$2" dest="$3"
  local out="$dest/repo.tar.gz"
  if [ -f "$dest/.ok" ]; then
    echo "[skip] repo $name" | tee -a "$LOG"
    return 0
  fi
  mkdir -p "$dest"
  fetch "https://codeload.github.com/$owner_repo/tar.gz/refs/heads/main" "$out"
  if [ ! -s "$out" ]; then
    fetch "https://codeload.github.com/$owner_repo/tar.gz/refs/heads/master" "$out"
  fi
  if [ -s "$out" ]; then
    tar -xzf "$out" -C "$dest" --strip-components=1 2>>"$LOG" || true
    touch "$dest/.ok"
  fi
}

# ----------------------------------------------------------------------
# Benchmark repositories
# ----------------------------------------------------------------------
fetch_repo xiaowu0162/LongMemEval LongMemEval "$DATA/LongMemEval/repo"
fetch_repo snap-research/locomo locomo "$DATA/LoCoMo/repo"
fetch_repo HUST-AI-HYZ/MemoryAgentBench MemoryAgentBench "$DATA/MemoryAgentBench/repo"
fetch_repo ulab-uiuc/MARBLE MARBLE "$DATA/MARBLE/repo"
fetch_repo rzhub/GateMem GateMem "$DATA/GateMem/repo"

# ----------------------------------------------------------------------
# Official named baseline source
# ----------------------------------------------------------------------
fetch_repo mem0ai/mem0 mem0 "$BASE/mem0"
fetch_repo letta-ai/letta letta "$BASE/letta"
fetch_repo parthsarthi03/raptor raptor "$BASE/raptor"
fetch_repo microsoft/graphrag graphrag "$BASE/graphrag"
fetch_repo agiresearch/A-mem a-mem "$BASE/a-mem"
fetch_repo getzep/graphiti graphiti "$BASE/graphiti"
fetch_repo microsoft/autogen autogen "$BASE/autogen"
fetch_repo LIUHA1ZHU/MemLens MemLens "$BASE/MemLens"
fetch_repo sept-usc/ClawGang ClawGang "$BASE/ClawGang"

# ----------------------------------------------------------------------
# Dataset files (HF via mirror)
# ----------------------------------------------------------------------
LME="$DATA/LongMemEval/data"
for f in longmemeval_oracle.json longmemeval_s_cleaned.json longmemeval_m_cleaned.json; do
  fetch "https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/$f" "$LME/$f"
done

MAB="$DATA/MemoryAgentBench/data"
for f in Accurate_Retrieval-00000-of-00001.parquet \
         Conflict_Resolution-00000-of-00001.parquet \
         Long_Range_Understanding-00000-of-00001.parquet \
         Test_Time_Learning-00000-of-00001.parquet; do
  fetch "https://hf-mirror.com/datasets/ai-hyz/MemoryAgentBench/resolve/main/data/$f" "$MAB/$f"
done
fetch "https://hf-mirror.com/datasets/ai-hyz/MemoryAgentBench/resolve/main/entity2id.json" "$MAB/entity2id.json"

GATE="$DATA/GateMem/data"
for domain in education household medical office; do
  for f in checkpoints.jsonl episodes.jsonl; do
    fetch "https://hf-mirror.com/datasets/Ray368/GateMem/resolve/main/gatemem/data/$domain/$f" "$GATE/$domain/$f"
  done
done

# LoCoMo data lives in the repo; MARBLE/GateMem code tarballs are in place.

# ----------------------------------------------------------------------
# Record upstream commit SHAs where the GitHub API is reachable.
# ----------------------------------------------------------------------
COMMITS="$BASE/commits.csv"
echo "repo,commit" > "$COMMITS"
for pair in \
  xiaowu0162/LongMemEval \
  snap-research/locomo \
  HUST-AI-HYZ/MemoryAgentBench \
  ulab-uiuc/MARBLE \
  rzhub/GateMem \
  mem0ai/mem0 \
  letta-ai/letta \
  parthsarthi03/raptor \
  microsoft/graphrag \
  agiresearch/A-mem \
  getzep/graphiti \
  microsoft/autogen \
  LIUHA1ZHU/MemLens \
  sept-usc/ClawGang; do
  sha=$(curl -s --max-time 20 "https://api.github.com/repos/$pair/commits/main" | \
        /root/miniconda3/bin/python -c "import json,sys; print(json.load(sys.stdin).get('sha',''))" 2>/dev/null || true)
  if [ -z "$sha" ]; then
    sha=$(curl -s --max-time 20 "https://api.github.com/repos/$pair/commits/master" | \
          /root/miniconda3/bin/python -c "import json,sys; print(json.load(sys.stdin).get('sha',''))" 2>/dev/null || true)
  fi
  echo "$pair,$sha" >> "$COMMITS"
done

echo "[$(date -u +%FT%TZ)] done" | tee -a "$LOG"
du -sh "$DATA" "$BASE" 2>/dev/null | tee -a "$LOG"
df -h "$ROOT" | tail -1 | tee -a "$LOG"
