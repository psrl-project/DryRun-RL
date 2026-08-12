"""Hydra-based CLI entrypoint for DryRun-RL."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ..config import register_configs
from ..cost.analytical import UnifiedRoofline
from ..cost.base import CostModel, RecomputeCostModel, SyncCostModel, TrainCostModel
from ..cost.empirical import DistServe, PSRLFitted
from ..cost.train_cost import (
    AnalyticalRecomputeCost,
    AnalyticalTrainCost,
    BandwidthSyncCost,
    FixedRecomputeCost,
    FixedSyncCost,
    FixedTrainCost,
)
from ..policy.areal import ArealPolicy
from ..policy.base import StalenessPolicy
from ..policy.psrl import PSRLPolicy
from ..policy.roll import RollPolicy
from ..policy.slime import SlimePolicy
from ..policy.verl import VerlPolicy
from ..sim import SimConfig, Simulator
from ..viz.plots import SimPlotter
from ..workload.distributions import bimodal, lognormal, powerlaw, uniform

dryrun_logger = logging.getLogger("dryrun")

register_configs()


def build_cost_model(cfg: DictConfig) -> CostModel:
    """Instantiate a rollout cost model from config."""
    name = cfg.get("name", "roofline")
    if name == "roofline":
        return UnifiedRoofline(
            F=cfg.get("F", 0.005),
            W=cfg.get("W", 0.001),
            G=cfg.get("G", 0.0001),
            A_p=cfg.get("A_p", 0.0),
            A_d=cfg.get("A_d", 1e-7),
        )
    if name == "psrl_fitted":
        path = cfg.get("path", "")
        assert path, "cost_model.path is required for psrl_fitted."
        return PSRLFitted.from_json(path, key=cfg.get("key", "TP1_PP1"))
    if name == "distserve":
        return DistServe(
            h=cfg.get("h", 4096),
            m=cfg.get("m", 11008),
            C1=cfg.get("C1", 0.0),
            C2=cfg.get("C2", 0.0),
            C3=cfg.get("C3", 0.0),
            C4=cfg.get("C4", 0.0),
            C5=cfg.get("C5", 0.0),
            b=cfg.get("b", 16),
            F=cfg.get("F", 0.0),
        )
    raise ValueError(f"Unknown cost model: {name!r}.")


def build_policy(cfg: DictConfig, max_staleness: int) -> StalenessPolicy:
    """Instantiate a staleness policy from config."""
    name = cfg.get("name", "psrl")
    if name == "psrl":
        return PSRLPolicy(
            proactive_filter=cfg.get("proactive_filter", False),
            proactive_threshold=cfg.get("proactive_threshold", 0),
        )
    if name == "areal":
        mc = cfg.get("max_concurrent", None)
        return ArealPolicy(max_concurrent=mc)
    if name == "roll":
        mc = cfg.get("max_concurrent", None)
        return RollPolicy(max_concurrent=mc)
    if name == "slime":
        return SlimePolicy(
            update_weights_interval=cfg.get("update_weights_interval", 1),
            over_sampling_ratio=cfg.get("over_sampling_ratio", 1.0),
            max_concurrent=cfg.get("max_concurrent", None),
        )
    if name == "verl":
        return VerlPolicy(
            staleness_threshold=cfg.get("staleness_threshold", 0.1),
            trigger_parameter_sync_step=cfg.get("trigger_parameter_sync_step", 4),
            require_batches=cfg.get("require_batches", 1),
            max_concurrent=cfg.get("max_concurrent", None),
        )
    raise ValueError(f"Unknown policy: {name!r}.")


def build_workload(cfg: DictConfig) -> list[int]:
    """Generate workload lengths from config."""
    name = cfg.get("name", "uniform")
    n = cfg.get("n", 200)
    seed = cfg.get("seed", 42)
    if name == "uniform":
        return uniform(n, cfg.get("lo", 50), cfg.get("hi", 200), seed=seed)
    if name == "bimodal":
        return bimodal(
            n,
            cfg.get("short_len", 50),
            cfg.get("long_len", 500),
            cfg.get("long_frac", 0.25),
            seed=seed,
        )
    if name == "lognormal":
        return lognormal(n, cfg.get("mu", 4.0), cfg.get("sigma", 1.0), seed=seed)
    if name == "powerlaw":
        return powerlaw(
            n, cfg.get("alpha", 2.0), cfg.get("lo", 10), cfg.get("hi", 1000), seed=seed
        )
    raise ValueError(f"Unknown workload: {name!r}.")


def build_train_cost(cfg: DictConfig) -> TrainCostModel:
    """Instantiate a training cost model from config."""
    name = cfg.get("name", "fixed")
    if name == "fixed":
        return FixedTrainCost(train_time=cfg.get("train_time", 1.0))
    if name == "analytical":
        return AnalyticalTrainCost(
            model_params=cfg.get("model_params", 0),
            peak_flops=cfg.get("peak_flops", 312e12),
            memory_bandwidth=cfg.get("memory_bandwidth", 2e12),
            overhead_factor=cfg.get("overhead_factor", 1.1),
        )
    raise ValueError(f"Unknown train cost model: {name!r}.")


def build_sync_cost(cfg: DictConfig) -> SyncCostModel:
    """Instantiate a sync cost model from config."""
    name = cfg.get("name", "fixed")
    if name == "fixed":
        return FixedSyncCost(sync_time=cfg.get("sync_time", 0.0))
    if name == "bandwidth":
        return BandwidthSyncCost()
    raise ValueError(f"Unknown sync cost model: {name!r}.")


def build_recompute_cost(cfg: DictConfig) -> RecomputeCostModel:
    """Instantiate a recompute cost model from config."""
    name = cfg.get("name", "fixed")
    if name == "fixed":
        return FixedRecomputeCost(time_per_token=cfg.get("time_per_token", 0.0))
    if name == "analytical":
        return AnalyticalRecomputeCost(
            model_params=cfg.get("model_params", 0),
            peak_flops=cfg.get("peak_flops", 312e12),
            overhead_factor=cfg.get("overhead_factor", 1.1),
        )
    raise ValueError(f"Unknown recompute cost model: {name!r}.")


def run_simulate(cfg: DictConfig) -> None:
    """Run a simulation from a resolved Hydra config."""
    sim_cfg = SimConfig(
        batch_size=cfg.batch_size,
        max_staleness=cfg.max_staleness,
        n_versions=cfg.n_versions,
        train_time=cfg.train_cost.get("train_time", 1.0),
        sync_time=cfg.sync_cost.get("sync_time", 0.0),
        recompute_time=0.0,
        partial_rollout=cfg.partial_rollout,
        token_budget=cfg.token_budget,
        kv_blocks=cfg.kv_blocks,
        block_size=cfg.block_size,
        n_instances=cfg.n_instances,
        prompt_len=cfg.prompt_len,
        livelock_rounds=cfg.get("livelock_rounds", 50),
        max_engine_iters=cfg.get("max_engine_iters", 10_000),
    )

    cost_model = build_cost_model(cfg.cost_model)
    policy = build_policy(cfg.policy, cfg.max_staleness)
    lengths = build_workload(cfg.workload)
    train_cost = build_train_cost(cfg.train_cost)
    sync_cost = build_sync_cost(cfg.sync_cost)
    recompute_cost = build_recompute_cost(cfg.recompute_cost)

    dryrun_logger.info(
        f"Starting simulation: policy={policy.name}, n_versions={sim_cfg.n_versions}, "
        f"n_instances={sim_cfg.n_instances}, batch_size={sim_cfg.batch_size}."
    )

    sim = Simulator(
        cost=cost_model,
        policy=policy,
        lengths=lengths,
        cfg=sim_cfg,
        train_cost=train_cost,
        sync_cost=sync_cost,
        recompute_cost=recompute_cost,
    )
    result = sim.run()

    print(f"Simulation complete: {result.versions_done} versions in {result.sim_time:.2f}s")
    print(f"  Consumed: {result.n_consumed} requests")
    print(f"  Dropped: {len(result.dropped)} requests")
    print(f"  Throughput: {result.throughput:.1f} tokens/s")
    if result.consumed:
        print(f"  Max staleness (true): {result.max_staleness_true}")
        print(f"  Max staleness (reported): {result.max_staleness_reported}")
    if result.livelocked:
        print("  WARNING: Simulation livelocked!")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plotter = SimPlotter(result, output_dir / "plots")
    paths = plotter.plot_all()
    print(f"\n  Plots saved to: {output_dir / 'plots'}")
    for p in paths:
        print(f"    {p.name}")


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "defaults")


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="simulate")
def main(cfg: DictConfig) -> None:
    """DryRun-RL simulation entrypoint."""
    run_simulate(cfg)


if __name__ == "__main__":
    main()
