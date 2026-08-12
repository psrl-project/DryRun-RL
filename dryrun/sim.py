"""The simulation driver: rollout generation interleaved with training.

This module provides the Simulator class that orchestrates the full RL
training loop using the event-driven architecture. It mirrors the sim_staleness
driver but with the multi-instance event coordinator infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core.types import ReqStatus, Request
from .cost.base import CostModel, RecomputeCostModel, SyncCostModel, TrainCostModel
from .cost.train_cost import FixedRecomputeCost, FixedSyncCost, FixedTrainCost
from .engine.instance import NativeInstance
from .policy.base import CompleteAction, SimState, StalenessPolicy


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


class Simulator:
    """
    Single or multi-instance simulation driver.

    Runs the full RL training loop: admit -> generate -> complete ->
    select batch -> (recompute) -> train -> weight update -> abort/re-prefill.
    """

    def __init__(
        self,
        cost: CostModel,
        policy: StalenessPolicy,
        lengths: list[int],
        cfg: SimConfig,
        train_cost: TrainCostModel | None = None,
        sync_cost: SyncCostModel | None = None,
        recompute_cost: RecomputeCostModel | None = None,
    ):
        self.cost = cost
        self.policy = policy
        self.lengths = list(lengths)
        self.cfg = cfg
        self.train_cost = train_cost or FixedTrainCost(cfg.train_time)
        self.sync_cost = sync_cost or FixedSyncCost(cfg.sync_time)
        self.recompute_cost = recompute_cost or FixedRecomputeCost(
            cfg.recompute_time / max(1, cfg.batch_size * cfg.prompt_len) if cfg.recompute_time > 0 else 0.0
        )

        self.instances: list[NativeInstance] = []
        for i in range(cfg.n_instances):
            self.instances.append(
                NativeInstance(
                    cost,
                    token_budget=cfg.token_budget,
                    kv_blocks=cfg.kv_blocks,
                    block_size=cfg.block_size,
                    max_running=cfg.max_concurrent,
                    instance_id=i,
                )
            )

        self.t: float = 0.0
        self.version: int = 0
        self.engine_version: int = 0
        self.next_rid: int = 0
        self.inflight: list[Request] = []
        self.ready: list[Request] = []
        self.consumed: list[Request] = []
        self.dropped: list[Request] = []
        self.n_admitted: int = 0
        self.n_consumed: int = 0
        self.train_timestamps: list[tuple[float, float]] = []

    def _state(self) -> SimState:
        return SimState(
            now=self.t,
            version=self.version,
            engine_version=self.engine_version,
            batch_size=self.cfg.batch_size,
            max_staleness=self.cfg.max_staleness,
            inflight=self.inflight,
            ready=self.ready,
            n_admitted=self.n_admitted,
            n_consumed=self.n_consumed,
            n_instances=self.cfg.n_instances,
        )

    def _next_length(self) -> int:
        return int(self.lengths[self.next_rid % len(self.lengths)])

    def _select_instance(self, req: Request) -> int:
        loads = [inst.n_active for inst in self.instances]
        return loads.index(min(loads))

    def _admit(self) -> int:
        st = self._state()
        quota = self.policy.admit_quota(st)
        admitted = 0
        while admitted < quota:
            req = Request(
                rid=self.next_rid,
                prompt_len=self.cfg.prompt_len,
                target_len=self._next_length(),
                v_traj=self.engine_version,
                admit_time=self.t,
            )
            instance_id = self._select_instance(req)
            if not self.instances[instance_id].add_request(req, self.t):
                break
            self.next_rid += 1
            self.inflight.append(req)
            self.n_admitted += 1
            self.policy.on_admit(req, st)
            admitted += 1
        return admitted

    def run(self) -> SimResult:
        idle_rounds = 0
        while self.version < self.cfg.n_versions:
            self._admit()

            batch = self.policy.select_batch(self._state())
            if batch is None:
                if not self.inflight and self._admit() == 0:
                    has_waiting = any(inst.waiting for inst in self.instances)
                    if not has_waiting:
                        idle_rounds += 1
                        if idle_rounds > self.cfg.livelock_rounds:
                            return self._result(livelocked=True)
                        continue
                before = len(self.ready)
                self._generate_until_progress()
                idle_rounds = 0 if len(self.ready) > before else idle_rounds + 1
                if idle_rounds > self.cfg.livelock_rounds:
                    return self._result(livelocked=True)
                continue

            idle_rounds = 0
            self._train(batch)

        return self._result()

    def _collect(self, done: list[Request]) -> None:
        st = self._state()
        for req in done:
            if req in self.inflight:
                self.inflight.remove(req)
            req.complete_time = self.t
            if self.policy.on_complete(req, st) is CompleteAction.DROP:
                req.status = ReqStatus.DROPPED
                self.dropped.append(req)
            else:
                self.ready.append(req)

    def _discard(self, reqs: list[Request]) -> None:
        seen: set[int] = set()
        to_cancel: list[Request] = []
        for req in reqs:
            if id(req) in seen:
                continue
            seen.add(id(req))
            if req in self.inflight:
                self.inflight.remove(req)
                to_cancel.append(req)
            if req in self.ready:
                self.ready.remove(req)
            req.status = ReqStatus.DROPPED
            self.dropped.append(req)
        if to_cancel:
            for inst in self.instances:
                inst.cancel(
                    [r for r in to_cancel if r.instance_id == inst.instance_id],
                    keep_tokens=False,
                )

    def _generate_until_progress(self) -> None:
        before = len(self.ready)
        guard = 0
        while len(self.ready) == before:
            guard += 1
            has_active = any(inst.running or inst.waiting for inst in self.instances)
            if guard > self.cfg.max_engine_iters or not has_active:
                return
            for inst in self.instances:
                if not inst.running and not inst.waiting:
                    continue
                done, _ = inst.advance_to(self.t + 10**9, self.engine_version)
                self.t = max(self.t, inst.t_local)
                if done:
                    self._collect(done)
                    if len(self.ready) > before:
                        return

    def _generate_during(self, duration: float) -> None:
        if duration <= 0:
            return
        deadline = self.t + duration
        guard = 0
        while guard <= self.cfg.max_engine_iters:
            guard += 1
            all_done = True
            for inst in self.instances:
                if not inst.running and not inst.waiting:
                    continue
                all_done = False
                done, reached = inst.advance_to(deadline, self.engine_version)
                self.t = max(self.t, inst.t_local)
                if done:
                    self._collect(done)
            if all_done or all(inst.t_local >= deadline for inst in self.instances):
                break
        self.t = max(self.t, deadline)
        for inst in self.instances:
            inst.t_local = max(inst.t_local, deadline)

    def _train(self, batch: list[Request]) -> None:
        train_start = self.t
        for req in batch:
            req.consume_version = self.version
            if req in self.ready:
                self.ready.remove(req)
            self.consumed.append(req)
            self.n_consumed += 1

        self.version += 1
        self.policy.on_version_advance(self.version, self._state())
        self.engine_version = self.policy.engine_version_after_train(self._state())

        expired = self.policy.expire(self._state())
        if expired:
            self._discard(expired)

        if self.cfg.partial_rollout and self.inflight:
            for inst in self.instances:
                inst.abort_all()
            self.policy.on_abort(list(self.inflight), self._state())

        self._admit()

        total_tokens = sum(r.ctx for r in batch)
        train_time = self.train_cost.step_time(len(batch), total_tokens, 1)
        sync_time = self.sync_cost.sync_time(0, self.cfg.n_instances, 0)
        recompute_time = self.recompute_cost.recompute_time(len(batch), total_tokens, 1)

        total_duration = train_time + sync_time + recompute_time
        self._generate_during(total_duration)
        self.train_timestamps.append((train_start, self.t))
        self.policy.check_invariants(self._state())

    def _result(self, livelocked: bool = False) -> SimResult:
        return SimResult(
            consumed=self.consumed,
            dropped=self.dropped,
            sim_time=self.t,
            versions_done=self.version,
            livelocked=livelocked,
            train_timestamps=self.train_timestamps,
        )
