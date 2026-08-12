"""Hydra structured configs for DryRun-RL."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostModelConfig:
    """Rollout inference cost model configuration."""

    name: str = "roofline"
    F: float = 0.005
    W: float = 0.001
    G: float = 0.0001
    A_p: float = 0.0
    A_d: float = 1e-7
    b: int = 16
    path: str = ""
    key: str = "TP1_PP1"
    h: int = 4096
    m: int = 11008
    C1: float = 0.0
    C2: float = 0.0
    C3: float = 0.0
    C4: float = 0.0
    C5: float = 0.0


@dataclass
class PolicyConfig:
    """Staleness policy configuration."""

    name: str = "psrl"
    proactive_filter: bool = False
    proactive_threshold: int = 0
    max_concurrent: int | None = None
    update_weights_interval: int = 1
    over_sampling_ratio: float = 1.0
    staleness_threshold: float = 0.1
    trigger_parameter_sync_step: int = 4
    require_batches: int = 1


@dataclass
class WorkloadConfig:
    """Workload length distribution configuration."""

    name: str = "uniform"
    n: int = 200
    seed: int = 42
    lo: int = 50
    hi: int = 200
    short_len: int = 50
    long_len: int = 500
    long_frac: float = 0.25
    mu: float = 4.0
    sigma: float = 1.0
    alpha: float = 2.0


@dataclass
class TrainCostConfig:
    """Training step cost configuration."""

    name: str = "fixed"
    train_time: float = 1.0
    model_params: int = 0
    peak_flops: float = 312e12
    memory_bandwidth: float = 2e12
    overhead_factor: float = 1.1


@dataclass
class SyncCostConfig:
    """Weight sync cost configuration."""

    name: str = "fixed"
    sync_time: float = 0.0


@dataclass
class RecomputeCostConfig:
    """Recompute cost configuration."""

    name: str = "fixed"
    time_per_token: float = 0.0
    model_params: int = 0
    peak_flops: float = 312e12
    overhead_factor: float = 1.1


@dataclass
class SimulateConfig:
    """Top-level simulation configuration."""

    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    train_cost: TrainCostConfig = field(default_factory=TrainCostConfig)
    sync_cost: SyncCostConfig = field(default_factory=SyncCostConfig)
    recompute_cost: RecomputeCostConfig = field(default_factory=RecomputeCostConfig)

    batch_size: int = 8
    max_staleness: int = 2
    n_versions: int = 20
    partial_rollout: bool = True
    token_budget: int = 8192
    kv_blocks: int = 100_000
    block_size: int = 16
    n_instances: int = 1
    prompt_len: int = 512
    livelock_rounds: int = 50
    max_engine_iters: int = 10_000
    output_dir: str = "outputs"
