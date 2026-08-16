"""Simulation configuration and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..component.rollout.engine import Request


@dataclass
class SimConfig:
    """Configuration for a single simulation run."""

    batch_size: int = 8
    max_staleness: int = 1
    n_versions: int = 20
    train_time: float = 1.0
    sync_time: float = 0.0
    recompute_time: float = 0.0
    partial_rollout: bool = True
    token_budget: int = 8192
    kv_blocks: int = 100_000
    block_size: int = 16
    max_concurrent: int | None = None
    max_concurrency: int | None = None
    reject_if_kv_full: bool = True
    reject_if_waiting: bool = False
    reject_if_running_full: bool = False
    prompt_len: int = 512
    n_instances: int = 1
    livelock_rounds: int = 50
    max_engine_iters: int = 10_000


@dataclass
class SimResult:
    """Simulation output."""

    consumed: list[Request] = field(default_factory=list)
    dropped: list[Request] = field(default_factory=list)
    sim_time: float = 0.0
    versions_done: int = 0
    livelocked: bool = False
    train_timestamps: list[tuple[float, float]] = field(default_factory=list)

    @property
    def n_consumed(self) -> int:
        return len(self.consumed)

    @property
    def staleness_true_values(self) -> list[int]:
        return [r.staleness_true() for r in self.consumed]

    @property
    def staleness_reported_values(self) -> list[int]:
        return [r.staleness_reported() for r in self.consumed]

    @property
    def max_staleness_true(self) -> int:
        vals = self.staleness_true_values
        return max(vals) if vals else 0

    @property
    def max_staleness_reported(self) -> int:
        vals = self.staleness_reported_values
        return max(vals) if vals else 0

    @property
    def throughput(self) -> float:
        if self.sim_time <= 0:
            return 0.0
        total_tokens = sum(r.generated for r in self.consumed)
        return total_tokens / self.sim_time
