"""Interfaces and workload vocabulary for rollout latency models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutWorkload:
    """Describe one inference engine iteration."""

    phase: str
    n_tokens: float
    sum_l2: float
    ctxsum: float
    request_count: int

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode", "mixed"}:
            raise ValueError(f"Unsupported rollout phase {self.phase!r}.")
        if self.n_tokens < 0 or self.sum_l2 < 0 or self.ctxsum < 0 or self.request_count < 0:
            raise ValueError("Rollout workload values must be non-negative.")

    @classmethod
    def from_step(
        cls,
        n_tokens,
        sum_l2,
        ctxsum,
        request_count: int,
    ) -> RolloutWorkload:
        """Infer the phase for the simulator's compact step interface."""
        if sum_l2 > 0 and ctxsum > 0:
            phase = "mixed"
        elif sum_l2 > 0:
            phase = "prefill"
        else:
            phase = "decode"
        return cls(
            phase=phase,
            n_tokens=n_tokens,
            sum_l2=sum_l2,
            ctxsum=ctxsum,
            request_count=request_count,
        )


class RolloutCostModel(ABC):
    """Predict one rollout step and expose pure-decode closed-form terms."""

    @abstractmethod
    def predict(self, workload: RolloutWorkload):
        """Predict strictly positive wall-clock latency in seconds."""

    def step_time(self, n_tokens, t2, ctxsum, n_reqs=0):
        """Adapt the simulator's compact step arguments to `RolloutWorkload`."""
        return self.predict(RolloutWorkload.from_step(n_tokens, t2, ctxsum, n_reqs))

    @abstractmethod
    def decode_coeffs(self, n_d: int, ctxsum) -> tuple:
        """Return floor, intercept, and slope for a pure-decode segment."""

    def saturated(self, n_tokens, t2, ctxsum, n_reqs=0) -> bool:
        """Return whether the workload is above a fitted low-load floor."""
        return False
