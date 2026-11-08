"""Aggregate official MultiAgentBench bargaining task results."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("AGORA_CLOUD_ROOT", "/root/autodl-tmp/agora_planner_cli"))
BASE = ROOT / "results" / "marble_bench"


def mean(values):
    return sum(values) / len(values) if values else None


def record(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = rows[-1]
    evaluation = data.get("task_evaluation", {})
    buyer = evaluation.get("buyer", {})
    seller = evaluation.get("seller", {})
    valid = (
        buyer.get("effectiveness_of_strategies", -1) >= 0
        and seller.get("effectiveness_of_strategies", -1) >= 0
    )
    return {
        "valid": valid,
        "planning_scores": data.get("planning_scores", []),
        "communication_scores": data.get("communication_scores", []),
        "buyer": buyer,
        "seller": seller,
        "token_usage": data.get("token_usage", 0),
        "total_milestones": data.get("total_milestones", 0),
        "path": path.name,
        "mtime": path.stat().st_mtime,
    }


def main() -> int:
    # One task can have both ``bargaining_task1_result.jsonl`` (early smoke) and
    # ``bargaining_task_1_result.jsonl`` (a later failed rerun); prefer a record
    # with a valid official evaluation, otherwise the newest file.
    candidates: dict[int | None, list[dict]] = {}
    for path in BASE.glob("bargaining_task*_result.jsonl"):
        match = re.search(r"task_?(\d+)_result", path.name)
        task_id = int(match.group(1)) if match else None
        try:
            item = record(path)
        except Exception as exc:
            item = {"valid": False, "error": str(exc), "path": path.name, "mtime": path.stat().st_mtime}
        candidates.setdefault(task_id, []).append(item)
    tasks = []
    for task_id, items in sorted(candidates.items(), key=lambda item: (item[0] is None, item[0])):
        items = sorted(items, key=lambda item: (item["valid"], item["mtime"]), reverse=True)
        task = dict(items[0])
        task["task_id"] = task_id
        tasks.append(task)
    valid = [task for task in tasks if task.get("valid")]
    metrics = {}
    for role in ("buyer", "seller"):
        for field in ("effectiveness_of_strategies", "progress_and_outcome", "interaction_dynamics"):
            metrics[f"{role}.{field}"] = mean(
                [task[role].get(field) for task in valid if isinstance(task.get(role), dict)]
            )
    metrics["planning"] = mean(
        [score for task in valid for score in task.get("planning_scores", []) if score >= 0]
    )
    metrics["tokens_per_task"] = mean([task.get("token_usage", 0) for task in valid])
    metrics["n_tasks"] = len(tasks)
    metrics["n_valid"] = len(valid)
    metrics["n_invalid"] = len(tasks) - len(valid)
    metrics["n_buyer_invalid"] = sum(
        1 for task in tasks if isinstance(task.get("buyer"), dict)
        and task["buyer"].get("effectiveness_of_strategies", -1) < 0
    )
    metrics["n_seller_invalid"] = sum(
        1 for task in tasks if isinstance(task.get("seller"), dict)
        and task["seller"].get("effectiveness_of_strategies", -1) < 0
    )
    payload = {"engine": "official MultiAgentBench Bargaining (WorldSimulation)", "metrics": metrics, "tasks": tasks}
    (BASE / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
