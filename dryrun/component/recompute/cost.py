"""Recompute cost models.

Some RL algorithms (PPO, GRPO) require log-probabilities under the new
policy for the consumed batch. This is a forward pass on the training
side, cheaper than a full training step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RecomputeCostModel(ABC):
    """
    Log-prob recomputation latency model.

    Some RL algorithms (PPO, GRPO) require log-probabilities under the new
    policy for the consumed batch. This is a forward pass on the training
    side, cheaper than a full training step.
    """

    @abstractmethod
    def recompute_time(self, batch_size: int, total_tokens: int, tp_size: int) -> float:
        """
        Wall-clock time for recomputing log-probs on a batch.

        Args:
            batch_size: Number of sequences.
            total_tokens: Total tokens across all sequences.
            tp_size: Tensor-parallel degree used for the recompute pass.
        """


class FixedRecomputeCost(RecomputeCostModel):
    """Recompute time proportional to total tokens."""

    def __init__(self, time_per_token: float):
        self.time_per_token = time_per_token

    def recompute_time(self, batch_size: int, total_tokens: int, tp_size: int) -> float:
        return self.time_per_token * total_tokens / tp_size


class AnalyticalRecomputeCost(RecomputeCostModel):
    """
    Recompute as a forward pass: 2 * model_params * total_tokens / (tp_size * peak_flops).
    """

    def __init__(self, model_params: int, peak_flops: float, overhead_factor: float = 1.1):
        self.model_params = model_params
        self.peak_flops = peak_flops
        self.overhead_factor = overhead_factor

    def recompute_time(self, batch_size: int, total_tokens: int, tp_size: int) -> float:
        return 2 * self.model_params * total_tokens / (tp_size * self.peak_flops) * self.overhead_factor
