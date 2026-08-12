"""Event types and event queue for the discrete-event simulation."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    REQUEST_ARRIVE = "request_arrive"
    REQUEST_DISPATCHED = "request_dispatched"
    REQUEST_PREFILL_DONE = "request_prefill_done"
    REQUEST_COMPLETE = "request_complete"
    TRAIN_BATCH_READY = "train_batch_ready"
    TRAIN_STEP_START = "train_step_start"
    TRAIN_STEP_DONE = "train_step_done"
    VERSION_ADVANCE = "version_advance"
    SYNC_START = "sync_start"
    SYNC_DONE = "sync_done"
    RECOMPUTE_START = "recompute_start"
    RECOMPUTE_DONE = "recompute_done"
    ADMISSION_CHECK = "admission_check"
    ABORT_ALL = "abort_all"
    EXPIRE_REQUESTS = "expire_requests"
    KV_EXHAUSTED = "kv_exhausted"
    PREEMPTION = "preemption"
    SIM_END = "sim_end"


EVENT_PRIORITY: dict[EventType, int] = {
    EventType.TRAIN_STEP_DONE: 0,
    EventType.VERSION_ADVANCE: 1,
    EventType.SYNC_DONE: 2,
    EventType.ABORT_ALL: 3,
    EventType.EXPIRE_REQUESTS: 4,
    EventType.RECOMPUTE_DONE: 5,
    EventType.TRAIN_BATCH_READY: 6,
    EventType.REQUEST_COMPLETE: 7,
    EventType.RECOMPUTE_START: 8,
    EventType.TRAIN_STEP_START: 9,
    EventType.SYNC_START: 10,
    EventType.REQUEST_PREFILL_DONE: 11,
    EventType.REQUEST_DISPATCHED: 12,
    EventType.REQUEST_ARRIVE: 13,
    EventType.ADMISSION_CHECK: 14,
    EventType.KV_EXHAUSTED: 15,
    EventType.PREEMPTION: 16,
    EventType.SIM_END: 99,
}


@dataclass
class Event:
    """
    A single simulation event.
    """

    time: float
    event_type: EventType
    source: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    _priority: int = field(init=False, repr=False)
    _seq: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._priority = EVENT_PRIORITY.get(self.event_type, 50)

    def __lt__(self, other: Event) -> bool:
        if self.time != other.time:
            return self.time < other.time
        if self._priority != other._priority:
            return self._priority < other._priority
        return self._seq < other._seq

    def __le__(self, other: Event) -> bool:
        return self == other or self < other


class EventQueue:
    """
    Priority queue of simulation events.
    """

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = itertools.count()

    def push(self, event: Event) -> None:
        event._seq = next(self._counter)
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek(self) -> Event | None:
        return self._heap[0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)
