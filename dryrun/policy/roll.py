"""ROLL admission + GC discard staleness policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .base import CompleteAction, SimState, StalenessPolicy
from ..core.types import Request


@dataclass
class _Group:
    """One FIFO of finished prompts per admission step."""

    running: set[int] = field(default_factory=set)
    finished: deque = field(default_factory=deque)


class RollPolicy(StalenessPolicy):
    """
    ROLL's RLVR async admission with GC discard enforcement.

    Same token bucket as AReaL for admission, but enforces the bound via:
    1. Consumption floor (min_step) prevents old batches.
    2. Per-step GC purges stragglers whose group aged past the window.
    """

    name = "roll"

    def __init__(self, max_concurrent: int | None = None):
        self.max_concurrent = max_concurrent
        self.budget = 0
        self.running_prompts = 0
        self.completed_prompts = 0
        self._configured = False
        self.groups: dict[int, _Group] = {}
        self.gc_dropped = 0

    def _ensure_group(self, step: int) -> _Group:
        return self.groups.setdefault(step, _Group())

    def _configure(self, st: SimState) -> None:
        ratio = st.max_staleness
        if ratio > 0:
            self.budget = ratio * st.batch_size
        self.budget += st.batch_size
        self._ensure_group(st.version)
        self._configured = True

    def admit_quota(self, st: SimState) -> int:
        if not self._configured:
            self._configure(st)
        self._ensure_group(st.version)
        room = self.budget - (self.running_prompts + self.completed_prompts)
        if self.max_concurrent is not None:
            room = min(room, self.max_concurrent - len(st.inflight))
        return max(0, room)

    def on_admit(self, req: Request, st: SimState) -> None:
        self.running_prompts += 1
        self._ensure_group(req.v_traj).running.add(req.rid)

    def on_complete(self, req: Request, st: SimState) -> CompleteAction:
        group = self.groups.get(req.v_traj)
        if group is None or req.rid not in group.running:
            return CompleteAction.DROP
        group.running.discard(req.rid)
        group.finished.append(req)
        self.running_prompts -= 1
        self.completed_prompts += 1
        return CompleteAction.KEEP

    def select_batch(self, st: SimState) -> list[Request] | None:
        min_step = st.version - st.max_staleness
        steps = [s for s in sorted(self.groups) if min_step <= s < st.version]
        if st.version in self.groups:
            steps.append(st.version)

        available = sum(len(self.groups[s].finished) for s in steps)
        if available < st.batch_size:
            return None

        collected: list[Request] = []
        for step in steps:
            group = self.groups[step]
            while group.finished and len(collected) < st.batch_size:
                collected.append(group.finished.popleft())
            if len(collected) >= st.batch_size:
                break
        return collected[:st.batch_size]

    def on_version_advance(self, version: int, st: SimState) -> None:
        self.budget += st.batch_size
        self._ensure_group(version)

    def expire(self, st: SimState) -> list[Request]:
        max_gc_step = (st.version - 1) - st.max_staleness
        if max_gc_step < 0:
            return []
        by_rid = {r.rid: r for r in st.inflight}
        victims = []
        for step in [s for s in self.groups if s <= max_gc_step]:
            group = self.groups.pop(step)
            assert not group.finished, (
                f"ROLL invariant violated: group {step} still holds "
                f"{len(group.finished)} uncollected finished prompt(s) at gc time."
            )
            for rid in group.running:
                req = by_rid.get(rid)
                if req is not None:
                    victims.append(req)
            self.running_prompts -= len(group.running)
            self.gc_dropped += len(group.running)
        return victims

    def check_invariants(self, st: SimState) -> None:
        floor = st.version - st.max_staleness
        for step, group in self.groups.items():
            if step < floor:
                assert not group.running and not group.finished, (
                    f"Group {step} below floor {floor} (version={st.version}) "
                    f"still has {len(group.running)} running / "
                    f"{len(group.finished)} finished."
                )
