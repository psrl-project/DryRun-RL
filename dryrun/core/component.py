"""SimComponent abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .events import Event, EventType

if TYPE_CHECKING:
    from .coordinator import EventCoordinator


class SimComponent(ABC):
    """
    Base class for all simulation components.
    """

    @abstractmethod
    def subscriptions(self) -> list[EventType]:
        """Return the event types this component handles."""

    @abstractmethod
    def handle(self, event: Event, coordinator: EventCoordinator) -> None:
        """Handle a dispatched event."""
