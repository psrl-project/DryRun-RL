"""slime loop barrier staleness policy."""

from __future__ import annotations

from .base import CompleteAction, SimState, StalenessPolicy
from ..core.types import Request


class SlimePolicy(StalenessPolicy):
    """
    slime's staleness control via training loop barrier structure.

    No staleness parameter. The bound comes purely from the loop structure:
    generation dispatched BEFORE weight push, consumed AFTER. Max staleness
    equals update_weights_interval exactly.
    """

    name = "slime"

    def __init__(
        self,
        update_weights_interval: int = 1,
        over_sampling_ratio: float = 1.0,
        partial_rollout: bool = False,
        max_concurrent: int | None = None,
    ):
        assert update_weights_interval >= 1, "update_weights_interval must be >= 1."
        assert over_sampling_ratio >= 1.0, "over_sampling_ratio must be >= 1.0."
        self.interval = update_weights_interval
        self.over_sampling_ratio = over_sampling_ratio
        self.partial_rollout = partial_rollout
        self.max_concurrent = max_concurrent

        self.remaining = 0
        self._engine_v = 0
        self._armed_push: int | None = None
        self._steps_since_push = 0
        self.stall_since: float | None = None
        self.stall_time = 0.0

    def admit_quota(self, st: SimState) -> int:
        if self.remaining >= st.batch_size:
            return 0
        room = max(1, int(round(self.over_sampling_ratio * st.batch_size)))
        if self.max_concurrent is not None:
            room = min(room, self.max_concurrent - len(st.inflight))
        return max(0, room)

    def on_admit(self, req: Request, st: SimState) -> None:
        self.remaining += 1

    def on_complete(self, req: Request, st: SimState) -> CompleteAction:
        if len(st.ready) < st.batch_size:
            return CompleteAction.KEEP
        return CompleteAction.DROP

    def select_batch(self, st: SimState) -> list[Request] | None:
        if len(st.ready) < st.batch_size:
            if st.inflight and self.stall_since is None:
                self.stall_since = st.now
            return None
        if self.stall_since is not None:
            self.stall_time += st.now - self.stall_since
            self.stall_since = None
        chosen = list(st.ready[:st.batch_size])
        self.remaining = 0
        return chosen

    def engine_version_after_train(self, st: SimState) -> int:
        if self._armed_push is not None:
            self._engine_v = self._armed_push
            self._armed_push = None

        self._steps_since_push += 1
        if self._steps_since_push >= self.interval:
            self._steps_since_push = 0
            self._armed_push = st.version

        return self._engine_v

    def expire(self, st: SimState) -> list[Request]:
        if self.partial_rollout:
            return []
        return list(st.inflight)

    def check_invariants(self, st: SimState) -> None:
        if not self.partial_rollout:
            assert st.version - self._engine_v <= self.interval, (
                f"Version gap {st.version - self._engine_v} exceeds interval {self.interval}."
            )
