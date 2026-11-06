"""AgoraPlanner: market-based memory allocation and task-collaboration RSI.

This repository is a research prototype.  The main lines are:

* :mod:`agora_planner.memory`, :mod:`agora_planner.benchmark`,
  :mod:`agora_planner.auction` — a synthetic market for memory assets;
* :mod:`agora_planner.demand_first_rsi`, :mod:`agora_planner.nl_demand_first_rsi`,
  :mod:`agora_planner.long_horizon_nl_rsi` — demand-first memory procurement;
* :mod:`agora_planner.task_collab_rsi` — bounded U-level RSI driven by
  per-memory task-completion gains (the current main prototype);
* :mod:`agora_planner.public_pool` — a reviewed team-memory commons;
* :mod:`agora_planner.protocol_rsi`, :mod:`agora_planner.learning_rsi` —
  protocol- and agent-level bounded RSI prototypes.

The code is intentionally small and dependency-light.  Optional vendor
adapters (Mem0, AutoGen, Letta, RAPTOR, GraphRAG) are isolated in their own
modules and import their third-party packages lazily.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
