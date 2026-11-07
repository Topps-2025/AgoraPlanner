#!/usr/bin/env bash
# Start the official Letta 0.16.8 REST server with embedded PostgreSQL/pgvector
# and the local hash-embedding endpoint.
set -u
ROOT="${AGORA_CLOUD_ROOT:-/root/autodl-tmp/agora_planner_cli}"
VENV="${LETTA_VENV:-/root/autodl-tmp/venvs/letta}"
LOG="$ROOT/logs"
mkdir -p "$LOG"

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export LETTA_LLM_PROVIDER="${LETTA_LLM_PROVIDER:-deepseek}"
export NO_PROXY="${NO_PROXY:-api.deepseek.com,127.0.0.1}"

# 1. embedded Postgres
nohup "$VENV/bin/python" "$ROOT/scripts/cloud/start_letta_pg.py" \
  > "$LOG/letta_pg_launcher.log" 2>&1 &
echo $! > "$LOG/letta_pg.pid"

for _ in $(seq 1 60); do
  if [ -s "$ROOT/letta_pg_uri.txt" ]; then break; fi
  sleep 2
done
if [ ! -s "$ROOT/letta_pg_uri.txt" ]; then
  echo "Postgres did not start; see $LOG/letta_pg_launcher.log" >&2
  exit 1
fi
export LETTA_PG_URI="$(cat "$ROOT/letta_pg_uri.txt")"
echo "LETTA_PG_URI=$LETTA_PG_URI"

# 2. Letta REST server
nohup "$VENV/bin/letta" server --host 127.0.0.1 --port 8283 \
  > "$LOG/letta_server.log" 2>&1 &
echo $! > "$LOG/letta_server.pid"

# 3. hash embedding endpoint
source /root/autodl-tmp/venvs/agora_cli/bin/activate
nohup python "$ROOT/experiments/hash_embedding_server.py" --port 8766 \
  > "$LOG/letta_embed_server.log" 2>&1 &
echo $! > "$LOG/letta_embed.pid"

for _ in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8283/v1/health || true)
  if [ "$code" = "200" ]; then
    echo "Letta server healthy"
    exit 0
  fi
  sleep 2
done
echo "Letta server did not become healthy; see $LOG/letta_server.log" >&2
exit 1
