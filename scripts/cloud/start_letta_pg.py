"""Start an embedded PostgreSQL + pgvector for the official Letta server.

Keeps the Postgres process alive and writes its URI to ``letta_pg_uri.txt``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pgserver

ROOT = Path(os.environ.get("AGORA_CLOUD_ROOT", "/root/autodl-tmp/agora_planner_cli"))
PGDATA = ROOT / "letta_pgdata"
URI_FILE = ROOT / "letta_pg_uri.txt"


def main() -> None:
    PGDATA.mkdir(parents=True, exist_ok=True)
    # cleanup_mode=None leaves the postmaster alive after this process exits;
    # this process also stays alive so the handle is never the last one.
    server = pgserver.get_server(PGDATA, cleanup_mode=None)
    uri = server.get_uri()
    URI_FILE.write_text(uri + "\n", encoding="utf-8")
    print(f"PG_URI {uri}", flush=True)
    try:
        print(server.psql("CREATE EXTENSION IF NOT EXISTS vector;"), flush=True)
    except Exception as exc:  # pragma: no cover - pgvector may already exist
        print(f"vector extension warning: {exc}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
