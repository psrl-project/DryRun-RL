"""Tests for Hydra configuration system."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dryrun.config.schema import (
    CostModelConfig,
    PolicyConfig,
    SimulateConfig,
    WorkloadConfig,
)
from dryrun.cli.main import (
    build_cost_model,
    build_policy,
    build_workload,
    build_train_cost,
    build_sync_cost,
    build_recompute_cost,
)
from dryrun.cost.analytical import UnifiedRoofline
from dryrun.cost.empirical import DistServe
from dryrun.cost.train_cost import AnalyticalTrainCost, FixedTrainCost, FixedSyncCost, FixedRecomputeCost
from dryrun.policy.psrl import PSRLPolicy
from dryrun.policy.areal import ArealPolicy
from dryrun.policy.roll import RollPolicy
from dryrun.policy.slime import SlimePolicy
from dryrun.policy.verl import VerlPolicy


class TestStructuredConfigs:
    """Test that dataclass configs have correct defaults."""

    def test_simulate_defaults(self):
        cfg = SimulateConfig()
        assert cfg.batch_size == 8
        assert cfg.max_staleness == 2
        assert cfg.n_versions == 20
        assert cfg.n_instances == 1
        assert cfg.partial_rollout is True
        assert cfg.cost_model.name == "roofline"
        assert cfg.policy.name == "psrl"
        assert cfg.workload.name == "uniform"

    def test_cost_model_defaults(self):
        cfg = CostModelConfig()
        assert cfg.name == "roofline"
        assert cfg.F == 0.005
        assert cfg.W == 0.001
        assert cfg.G == 0.0001
        assert cfg.A_d == 1e-7

    def test_policy_defaults(self):
        cfg = PolicyConfig()
        assert cfg.name == "psrl"

    def test_workload_defaults(self):
        cfg = WorkloadConfig()
        assert cfg.name == "uniform"
        assert cfg.n == 200
        assert cfg.seed == 42


class TestFactoryFunctions:
    """Test that factory functions produce correct instances."""

    def test_build_roofline(self):
        cfg = OmegaConf.create({"name": "roofline", "F": 0.01, "W": 0.002, "G": 0.0002, "A_p": 0, "A_d": 2e-7})
        model = build_cost_model(cfg)
        assert isinstance(model, UnifiedRoofline)
        assert model.F == 0.01
        assert model.A_d == 2e-7

    def test_build_distserve(self):
        cfg = OmegaConf.create({
            "name": "distserve", "h": 4096, "m": 11008,
            "C1": 1e-13, "C2": 1e-13, "C3": 0.001, "C4": 0, "C5": 1e-13,
        })
        model = build_cost_model(cfg)
        assert isinstance(model, DistServe)

    def test_build_unknown_cost_model(self):
        cfg = OmegaConf.create({"name": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown cost model"):
            build_cost_model(cfg)

    def test_build_all_policies(self):
        for name, cls in [
            ("psrl", PSRLPolicy),
            ("areal", ArealPolicy),
            ("roll", RollPolicy),
            ("slime", SlimePolicy),
            ("verl", VerlPolicy),
        ]:
            cfg = OmegaConf.create({"name": name})
            policy = build_policy(cfg, max_staleness=2)
            assert isinstance(policy, cls), f"Expected {cls}, got {type(policy)}."
            assert policy.name == name

    def test_build_unknown_policy(self):
        cfg = OmegaConf.create({"name": "nonexistent"})
        with pytest.raises(ValueError, match="Unknown policy"):
            build_policy(cfg, max_staleness=2)

    def test_build_all_workloads(self):
        for name in ["uniform", "bimodal", "lognormal", "powerlaw"]:
            cfg = OmegaConf.create({"name": name, "n": 50, "seed": 42})
            lengths = build_workload(cfg)
            assert len(lengths) == 50
            assert all(isinstance(l, int) for l in lengths)

    def test_build_fixed_train_cost(self):
        cfg = OmegaConf.create({"name": "fixed", "train_time": 2.5})
        model = build_train_cost(cfg)
        assert isinstance(model, FixedTrainCost)
        assert model.step_time(8, 1000, 1) == 2.5

    def test_build_analytical_train_cost(self):
        cfg = OmegaConf.create({
            "name": "analytical",
            "model_params": 8_000_000_000,
            "peak_flops": 312e12,
            "memory_bandwidth": 2e12,
        })
        model = build_train_cost(cfg)
        assert isinstance(model, AnalyticalTrainCost)
        t = model.step_time(8, 8000, 1)
        assert t > 0

    def test_build_sync_cost(self):
        cfg = OmegaConf.create({"name": "fixed", "sync_time": 0.5})
        model = build_sync_cost(cfg)
        assert isinstance(model, FixedSyncCost)

    def test_build_recompute_cost(self):
        cfg = OmegaConf.create({"name": "fixed", "time_per_token": 0.001})
        model = build_recompute_cost(cfg)
        assert isinstance(model, FixedRecomputeCost)


class TestOmegaConfIntegration:
    """Test that OmegaConf properly resolves structured configs."""

    def test_from_dataclass(self):
        cfg = OmegaConf.structured(SimulateConfig())
        assert cfg.batch_size == 8
        assert cfg.cost_model.name == "roofline"
        assert cfg.policy.name == "psrl"

    def test_override(self):
        cfg = OmegaConf.structured(SimulateConfig())
        cfg = OmegaConf.merge(cfg, {"batch_size": 16, "policy": {"name": "areal"}})
        assert cfg.batch_size == 16
        assert cfg.policy.name == "areal"

    def test_yaml_roundtrip(self, tmp_path):
        cfg = OmegaConf.structured(SimulateConfig())
        yaml_path = tmp_path / "test.yaml"
        OmegaConf.save(cfg, yaml_path)
        loaded = OmegaConf.load(yaml_path)
        assert loaded.batch_size == cfg.batch_size
        assert loaded.cost_model.name == cfg.cost_model.name
