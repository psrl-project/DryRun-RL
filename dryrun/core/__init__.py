"""Core module: types, events, components, and coordinator."""

from .component import SimComponent
from .coordinator import EventCoordinator, SimResult
from .events import Event, EventQueue, EventType
from .types import ReqStatus, Request, Segment
