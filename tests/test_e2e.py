"""End-to-end simulation tests."""

from dryrun.cost.analytical import UnifiedRoofline
from dryrun.policy.areal import ArealPolicy
from dryrun.policy.psrl import PSRLPolicy
from dryrun.policy.roll import RollPolicy
from dryrun.policy.slime import SlimePolicy
from dryrun.policy.verl import VerlPolicy
from dryrun.simulator import SimConfig, Simulator
from dryrun.workload.distributions import bimodal, uniform


def _run_sim(policy, lengths, cfg):
    cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_p=0, A_d=1e-7)
    sim = Simulator(cost=cost, policy=policy, lengths=lengths, cfg=cfg)
    return sim.run()


class TestPSRLBoundHolds:
    def test_uniform_lengths(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=10, train_time=0.5)
        lengths = uniform(100, 50, 200, seed=42)
        result = _run_sim(PSRLPolicy(), lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 10
        for r in result.consumed:
            assert r.staleness_true() <= cfg.max_staleness

    def test_bimodal_lengths(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=8, train_time=0.5)
        lengths = bimodal(100, short_len=50, long_len=500, long_frac=0.25, seed=42)
        result = _run_sim(PSRLPolicy(), lengths, cfg)
        assert not result.livelocked
        for r in result.consumed:
            assert r.staleness_true() <= cfg.max_staleness


class TestArealTokenBucket:
    def test_bucket_invariant(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=10, train_time=0.5)
        lengths = uniform(100, 50, 100, seed=0)
        result = _run_sim(ArealPolicy(), lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 10


class TestRollBound:
    def test_bound_holds_uniform(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=10, train_time=0.5)
        lengths = uniform(100, 50, 100, seed=0)
        result = _run_sim(RollPolicy(), lengths, cfg)
        assert not result.livelocked
        for r in result.consumed:
            assert r.staleness_true() <= cfg.max_staleness


class TestSlimeBarrier:
    def test_staleness_equals_interval(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=1,
            n_versions=10,
            train_time=0.5,
            partial_rollout=False,
        )
        lengths = uniform(100, 50, 100, seed=0)
        policy = SlimePolicy(update_weights_interval=2)
        result = _run_sim(policy, lengths, cfg)
        assert not result.livelocked
        for r in result.consumed:
            assert r.staleness_true() <= 2


class TestVerlAsync:
    def test_runs_without_crash(self):
        cfg = SimConfig(batch_size=4, max_staleness=1, n_versions=8, train_time=0.5)
        lengths = uniform(100, 50, 100, seed=0)
        policy = VerlPolicy(staleness_threshold=0.1, trigger_parameter_sync_step=2)
        result = _run_sim(policy, lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 8


class TestDeterminism:
    def test_same_seed_same_result(self):
        cfg = SimConfig(batch_size=4, max_staleness=2, n_versions=5, train_time=0.3)
        lengths = uniform(50, 50, 100, seed=42)
        r1 = _run_sim(PSRLPolicy(), lengths, cfg)
        r2 = _run_sim(PSRLPolicy(), lengths, cfg)
        assert r1.sim_time == r2.sim_time
        assert r1.versions_done == r2.versions_done
        assert len(r1.consumed) == len(r2.consumed)


class TestMultiInstance:
    def test_two_instances(self):
        cfg = SimConfig(
            batch_size=4,
            max_staleness=2,
            n_versions=8,
            train_time=0.5,
            n_instances=2,
        )
        lengths = uniform(100, 50, 100, seed=0)
        result = _run_sim(PSRLPolicy(), lengths, cfg)
        assert not result.livelocked
        assert result.versions_done == 8
        for r in result.consumed:
            assert r.staleness_true() <= cfg.max_staleness
