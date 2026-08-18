"""Value objects describing engine instance state.

`StepStat` accumulates what happened during engine steps (telemetry), while
`InstanceLoad` is the view a router uses to pick a target instance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepStat:
    """Aggregated engine steps.

    `dt`, `phase`, and `t` describe wall-clock cost so telemetry (the
    `rollout_step` log stream) can report per-step prefill vs. decode timing,
    not just token counts. `phase` is `"prefill"` for a mixed prefill+decode
    engine step (any waiting request in the batch) and `"decode"` for a pure
    closed-form decode segment. `t` is the simulation time at which the step
    ends (commits), matching `NativeInstance.t_local` after the step.
    """

    n_tokens: int
    n_decode: int
    n_prefill: int
    saturated: bool
    steps: int = 1
    saturated_steps: int = 0
    dt: float = 0.0
    phase: str = "decode"
    t: float = 0.0

    def __post_init__(self) -> None:
        if self.saturated and self.saturated_steps == 0:
            self.saturated_steps = self.steps


@dataclass
class InstanceLoad:
    """Snapshot of instance load for routing decisions."""

    instance_id: int
    n_waiting: int
    n_running: int
    kv_utilization: float
    total_ctx: int
