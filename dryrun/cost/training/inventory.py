"""Static parameter, gradient, and optimizer state inventory."""

from __future__ import annotations

from dataclasses import dataclass

from .base import TrainingParallelism

_DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "tf32": 4,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp8": 1,
    "int8": 1,
}


def dtype_bytes(dtype: str) -> int:
    """Return storage bytes for a supported training dtype."""
    normalized = dtype.lower()
    if normalized not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported training dtype {dtype!r}.")
    return _DTYPE_BYTES[normalized]


@dataclass(frozen=True)
class ParameterInventory:
    """Break down per-rank static state bytes."""

    model_parameters: float
    gradients: float
    master_parameters: float
    optimizer_moments: float

    @property
    def total_bytes(self) -> float:
        return (
            self.model_parameters
            + self.gradients
            + self.master_parameters
            + self.optimizer_moments
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "model_parameters": self.model_parameters,
            "gradients": self.gradients,
            "master_parameters": self.master_parameters,
            "optimizer_moments": self.optimizer_moments,
            "total_bytes": self.total_bytes,
        }


def parameter_inventory(
    *,
    parameter_count: int,
    parallelism: TrainingParallelism,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_state_dtype: str,
    distributed_optimizer: bool,
) -> ParameterInventory:
    """Compute V1 uniform-layer static state bytes on one rank."""
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive.")
    model_parallel = parallelism.tp * parallelism.pp
    local_parameters = parameter_count / model_parallel
    optimizer_shard = parallelism.dp if distributed_optimizer else 1
    parameter_bytes = dtype_bytes(parameter_dtype)
    gradient_bytes = dtype_bytes(gradient_dtype)
    optimizer_bytes = dtype_bytes(optimizer_state_dtype)
    master_bytes = optimizer_bytes if parameter_bytes < optimizer_bytes else 0
    return ParameterInventory(
        model_parameters=local_parameters * parameter_bytes,
        gradients=local_parameters * gradient_bytes / optimizer_shard,
        master_parameters=local_parameters * master_bytes / optimizer_shard,
        optimizer_moments=local_parameters * 2 * optimizer_bytes / optimizer_shard,
    )
