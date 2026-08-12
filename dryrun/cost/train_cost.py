"""Training, sync, and recompute cost model implementations.

These models predict the latency of training-side operations:
- **Training step:** Forward + backward pass + optimizer update.
- **Weight sync:** Distributing new weights to rollout instances.
- **Recompute:** Forward pass to get log-probs under the new policy (for PPO/GRPO).
"""

from __future__ import annotations

from .base import RecomputeCostModel, SyncCostModel, TrainCostModel


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
        model_bytes: Model size in bytes. Defaults to `model_params * 2`
            (BF16).
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


class FixedSyncCost(SyncCostModel):
    """Fixed synchronization time."""

    def __init__(self, sync_time: float):
        self.sync_time_val = sync_time

    def sync_time(self, model_size_bytes: int, n_instances: int, bandwidth_gbps: float) -> float:
        return self.sync_time_val


class BandwidthSyncCost(SyncCostModel):
    """Bandwidth-based sync time: model_size / bandwidth."""

    def sync_time(self, model_size_bytes: int, n_instances: int, bandwidth_gbps: float) -> float:
        bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
        return model_size_bytes / bandwidth_bytes_per_sec


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
