"""Training step cost models.

Predict the wall-clock time for one gradient accumulation + update step.
The dominant cost is the forward + backward pass over the batch, which
scales linearly with `total_tokens` and inversely with data-parallel degree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TrainCostModel(ABC):
    """
    Training step latency model.

    Predicts the wall-clock time for one gradient accumulation + update step.
    The dominant cost is the forward + backward pass over the batch, which
    scales linearly with `total_tokens` and inversely with data-parallel degree.
    """

    @abstractmethod
    def step_time(self, batch_size: int, total_tokens: int, dp_size: int) -> float:
        """
        Wall-clock time for one gradient step.

        Args:
            batch_size: Number of sequences in the training batch.
            total_tokens: Total tokens across all sequences (sum of lengths).
            dp_size: Data-parallel degree. Each DP rank processes
                `total_tokens / dp_size` tokens.
        """


class AnalyticalTrainCost(TrainCostModel):
    """
    Roofline-based training cost model.

    Formula:
        step_time = max(compute_time, memory_time) * overhead_factor

    Where:
        compute_time = 6 * model_params * total_tokens / (n_gpus * peak_flops)
        memory_time  = model_bytes / memory_bandwidth

    The factor of 6 comes from: 2 (forward FLOP/param) + 4 (backward
    FLOP/param) = 6 FLOPs per parameter per token.

    Args:
        model_params: Number of model parameters (e.g. 8e9 for 8B model).
        peak_flops: GPU peak FLOP/s (e.g. 312e12 for A100 BF16).
        memory_bandwidth: GPU memory bandwidth in bytes/s (e.g. 2e12 for A100).
        model_bytes: Model size in bytes. Defaults to `model_params * 2` (BF16).
        overhead_factor: Multiplicative overhead for communication, kernel
            gaps, etc. Typically 1.05-1.2.
    """

    def __init__(
        self,
        model_params: int,
        peak_flops: float,
        memory_bandwidth: float,
        model_bytes: int | None = None,
        overhead_factor: float = 1.1,
    ):
        self.model_params = model_params
        self.peak_flops = peak_flops
        self.memory_bandwidth = memory_bandwidth
        self.model_bytes = model_bytes or model_params * 2
        self.overhead_factor = overhead_factor

    def step_time(self, batch_size: int, total_tokens: int, dp_size: int) -> float:
        n_gpus = dp_size
        compute_time = 6 * self.model_params * total_tokens / (n_gpus * self.peak_flops)
        memory_time = self.model_bytes / self.memory_bandwidth
        return max(compute_time, memory_time) * self.overhead_factor


class FixedTrainCost(TrainCostModel):
    """Fixed training step time for simple simulations."""

    def __init__(self, train_time: float):
        self.train_time = train_time

    def step_time(self, batch_size: int, total_tokens: int, dp_size: int) -> float:
        return self.train_time
