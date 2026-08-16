"""Discrete events emitted by an engine instance."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .request import Request


class EventKind(Enum):
    PREFILL_STEP = "prefill_step"
    REQUEST_COMPLETE = "request_complete"
    KV_PREEMPT = "kv_preempt"
    DEADLOCK_PREEMPT = "deadlock_preempt"


@dataclass
class EngineEvent:
    """
    A single discrete event produced by one `NativeInstance`.

    Attributes:
        kind: Category of state change this event represents.
        instance_id: Which instance produced this event.
        t: Value of `t_local` on that instance immediately after the event.
        completed: Requests that finished generation as part of this event.
            Non-empty only for `REQUEST_COMPLETE` events.
    """

    kind: EventKind
    instance_id: int
    t: float
    completed: list[Request] = field(default_factory=list)
