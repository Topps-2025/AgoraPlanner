"""Merge per-baseline public-QA chunk summaries into one matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("AGORA_CLOUD_ROOT", "/root/autodl-tmp/agora_planner_cli"))
CHUNKS = ROOT / "results" / "public_glm53_chunks"
OUT = ROOT / "results" / "public_glm53_main"


def load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    datasets: dict[str, Any] = {}
    config = {}
    reader = {}
    for chunk in sorted(p for p in CHUNKS.iterdir() if p.is_dir()):
        summary = load(chunk / "summary.json")
        if not summary:
            continue
        config = summary.get("config", config)
        reader = summary.get("reader", reader)
        for dataset, body in summary.get("datasets", {}).items():
            target = datasets.setdefault(
                dataset,
                {
                    "n_memories": body.get("n_memories"),
                    "n_queries": body.get("n_queries"),
                    "baselines": {},
                },
            )
            target["baselines"].update(body.get("baselines", {}))
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "reader": reader, "datasets": datasets}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("merged", {name: len(body["baselines"]) for name, body in datasets.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
