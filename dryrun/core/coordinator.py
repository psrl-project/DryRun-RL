"""Event coordinator: the simulation timeline manager."""

from __future__ import annotations

from dataclasses import dataclass, field

from .component import SimComponent
from .events import Event, EventQueue, EventType
from .types import Request


@dataclass
class SimResult:
    """
    Aggregated simulation results.
    """

    consumed: list[Request] = field(default_factory=list)
    dropped: list[Request] = field(default_factory=list)
    inflight: list[Request] = field(default_factory=list)
    ready: list[Request] = field(default_factory=list)
    sim_time: float = 0.0
    versions_done: int = 0
    livelocked: bool = False
    events_processed: int = 0
    train_timestamps: list[tuple[float, float]] = field(default_factory=list)


class EventCoordinator:
    """
    Manages the global simulation timeline and dispatches events to registered
    components.
    """

    def __init__(self) -> None:
        self._queue = EventQueue()
        self._components: dict[str, SimComponent] = {}
        self._handlers: dict[EventType, list[SimComponent]] = {}
        self._clock: float = 0.0
        self._version: int = 0
        self._events_processed: int = 0

    @property
    def now(self) -> float:
        return self._clock

    @property
    def version(self) -> int:
        return self._version

    @version.setter
    def version(self, v: int) -> None:
        self._version = v

    def register_component(self, name: str, component: SimComponent) -> None:
        self._components[name] = component
        for event_type in component.subscriptions():
            self._handlers.setdefault(event_type, []).append(component)

    def get_component(self, name: str) -> SimComponent:
        return self._components[name]

    def schedule(self, event: Event) -> None:
        assert event.time >= self._clock, (
            f"Cannot schedule event at {event.time}, clock is at {self._clock}."
        )
        self._queue.push(event)

    def schedule_at(
        self,
        time: float,
        event_type: EventType,
        source: str = "",
        **payload,
    ) -> None:
        self.schedule(Event(time=time, event_type=event_type, source=source, payload=payload))

    def peek(self) -> Event | None:
        return self._queue.peek()

    def has_events(self) -> bool:
        return bool(self._queue)

    def run(
        self,
        max_versions: int | None = None,
        max_events: int = 1_000_000,
        until: float | None = None,
    ) -> None:
        """
        Run the event loop until a termination condition is met.
        """
        while self._queue and self._events_processed < max_events:
            if max_versions is not None and self._version >= max_versions:
                break
            if until is not None and self._queue.peek().time > until:
                break

            event = self._queue.pop()
            self._clock = event.time
            self._events_processed += 1

            handlers = self._handlers.get(event.event_type, [])
            for handler in handlers:
                handler.handle(event, self)

    @property
    def events_processed(self) -> int:
        return self._events_processed
