"""Staleness policy plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from ..component.rollout.engine import Request


class CompleteAction(Enum):
    KEEP = "keep"
    DROP = "drop"


@dataclass
class SimState:
    """What a policy is allowed to see when making decisions."""

    now: float = 0.0
    version: int = 0
    engine_version: int = 0
    batch_size: int = 0
    max_staleness: int = 0
    inflight: list[Request] = field(default_factory=list)
    ready: list[Request] = field(default_factory=list)
    n_admitted: int = 0
    n_consumed: int = 0
    n_instances: int = 1
    sync_in_progress: bool = False


class StalenessPolicy(ABC):
    """
    Base class for staleness control policies.
    """

    name: str = "base"

    @abstractmethod
    def admit_quota(self, st: SimState) -> int:
        """How many new rollouts may be admitted at this instant."""

    def on_admit(self, req: Request, st: SimState) -> None:
        """Hook for policies that reserve resources at dispatch time."""

    def on_complete(self, req: Request, st: SimState) -> CompleteAction:
        return CompleteAction.KEEP

    @abstractmethod
    def peek_batch(self, st: SimState) -> list[Request] | None:
        """Return the next training batch without consuming policy state."""

    def take_batch(self, st: SimState) -> list[Request] | None:
        """Consume and return the next training batch."""
        return self.peek_batch(st)

    def on_batch_unavailable(self, st: SimState) -> None:
        """Called when no training batch is currently available."""

    def on_version_advance(self, version: int, st: SimState) -> None:
        """Called after a training step produces a new weight version."""

    def on_abort(self, reqs: list[Request], st: SimState) -> None:
        """Called when in-flight generation is aborted for a weight update."""

    def expire(self, st: SimState) -> list[Request]:
        """Requests to permanently give up on right now."""
        return []

    def engine_version_after_train(self, st: SimState) -> int:
        """The weight version the rollout engine is actually serving."""
        return st.version

    def check_invariants(self, st: SimState) -> None:
        """Assert this policy's own contract."""
