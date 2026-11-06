"""Configuration objects for AgoraPlanner.

The prototype is intentionally configuration-driven so that the same code can
run a deterministic synthetic experiment, a noisy-bid robustness experiment, and
a small DeepSeek-backed demo.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Portable data layer.  Override with AGORA_DATA_ROOT (or AGORA_RESULTS_DIR).
# The default is a per-user directory so a fresh checkout does not write into
# the source tree.
_DATA_ROOT_ENV = os.environ.get("AGORA_DATA_ROOT")
if _DATA_ROOT_ENV:
    DEFAULT_RESULTS_DIR = Path(_DATA_ROOT_ENV) / "results"
else:
    DEFAULT_RESULTS_DIR = Path(
        os.environ.get("AGORA_RESULTS_DIR", Path.home() / ".agora_planner" / "results")
    )


@dataclass
class LLMConfig:
    """DeepSeek / OpenAI-compatible client configuration."""

    enabled: bool = True
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 60
    max_retries: int = 3
    cache_path: str | None = str(DEFAULT_RESULTS_DIR / "llm_cache.json")
    max_calls: int = 40

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """Parameters of the latent-facet memory benchmark."""

    seed: int = 0
    n_agents: int = 3
    n_facets: int = 10
    n_memories: int = 12
    n_train_tasks: int = 40
    n_eval_tasks: int = 40

    # Per-agent context / attention budget: how many assets an agent may hold.
    max_memories_per_agent: int = 2

    # At most one memory per source class per agent (partition matroid).
    one_per_source: bool = True

    source_classes: list[str] = field(
        default_factory=lambda: ["doc", "tool", "human", "derived"]
    )

    # Population structure.
    common_knowledge_fraction: float = 0.20
    cold_start_fraction: float = 0.20
    redundancy_fraction: float = 0.30
    poison_fraction: float = 0.10

    # Task structure.
    task_facet_min: int = 1
    task_facet_max: int = 3
    task_scopes: list[str] = field(
        default_factory=lambda: ["research", "coding", "sql", "ops"]
    )
    # Empty means uniform sampling.  A non-empty dict is used by the
    # distribution-shift experiment to emphasise a different facet mix.
    task_facet_weights: dict[str, float] = field(default_factory=dict)

    # Memory economics.
    utility_scale: float = 10.0
    memory_cost: float = 0.50
    usage_cost: float = 0.10
    common_mode_lambda: float = 0.35
    common_mode_risk_same_source: float = 0.85
    common_mode_risk_cross_source: float = 0.25

    # Success model.
    success_alpha: float = 3.2
    success_beta: float = 0.55
    success_gamma: float = 1.4
    poison_penalty: float = 0.9
    success_temperature: float = 1.0

    # Auction solver.
    max_combinations: int = 2_000_000
    beam_width: int = 96
    solver: str = "auto"  # auto | exact | beam | greedy

    # Valuation.
    valuation: str = "ground_truth"  # ground_truth | additive_noisy | llm
    bid_noise: float = 0.08
    bid_overconfidence: float = 0.10
    probe_samples: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RSIConfig:
    """Market-grounded recursive self-improvement configuration."""

    rounds: int = 3
    validation_seeds: int = 3
    delta: float = 0.002
    pareto_epsilon: float = 0.01
    mdl_lambda: float = 0.01
    max_memories: int = 20
    use_llm_proposer: bool = False
    max_proposals: int = 6
    self_validated_baseline: bool = True
    require_lower_confidence_bound: bool = True
    # Optional whitelist of transformation ops.  Empty means all ops.
    # Used by the v9 protocol layer to make ``update_rule`` a structural
    # protocol component rather than a free-form heuristic.
    allowed_ops: tuple[str, ...] = ()
    # Distribution-shift validation: held-out tasks emphasise facets that are
    # rare in the training stream.  This makes training-only edits visibly
    # unsafe while market-grounded validation should reject them.
    validation_shift: bool = False
    shift_strength: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Top-level configuration."""

    name: str = "agora_demo"
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rsi: RSIConfig = field(default_factory=RSIConfig)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    output_dir: str = str(DEFAULT_RESULTS_DIR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "benchmark": self.benchmark.to_dict(),
            "llm": self.llm.to_dict(),
            "rsi": self.rsi.to_dict(),
            "seeds": list(self.seeds),
            "output_dir": self.output_dir,
        }


def _update_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    """Recursively update a dataclass from a dict, ignoring unknown keys."""

    for key, value in values.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load a YAML config; missing fields keep their defaults."""

    config = ExperimentConfig()
    if path is None:
        return config
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _update_dataclass(config, raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict(), fh, allow_unicode=True, sort_keys=False)
