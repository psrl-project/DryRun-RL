"""Hydra configuration for DryRun-RL."""

from hydra.core.config_store import ConfigStore

from .schema import (
    AdmissionControlConfig,
    CostModelConfig,
    JobConfig,
    PolicyConfig,
    RecomputeCostConfig,
    RolloutConfig,
    SimulateConfig,
    SyncCostConfig,
    TrainCostConfig,
    WorkloadConfig,
)


def register_configs() -> None:
    """Register structured configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="simulate_schema", node=SimulateConfig)
    cs.store(group="cost_model", name="roofline_schema", node=CostModelConfig)
    cs.store(group="policy", name="psrl_schema", node=PolicyConfig)
    cs.store(group="workload", name="uniform_schema", node=WorkloadConfig)
    cs.store(group="train_cost", name="fixed_schema", node=TrainCostConfig)
    cs.store(group="sync_cost", name="fixed_schema", node=SyncCostConfig)
    cs.store(group="recompute_cost", name="fixed_schema", node=RecomputeCostConfig)
