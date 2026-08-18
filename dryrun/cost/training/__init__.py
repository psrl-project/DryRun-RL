"""Training latency and memory models."""

from .analytical import (
    AnalyticalTrainingCost,
    AnalyticalTrainingLatency,
    AnalyticalTrainingMemory,
)
from .base import (
    TrainingCostEstimate,
    TrainingCostModel,
    TrainingLatencyModel,
    TrainingMemoryModel,
    TrainingOutOfDomainError,
    TrainingParallelism,
    TrainingWorkload,
)
from .fixed import (
    FixedTrainingCost,
    FixedTrainingLatency,
    FixedTrainingMemory,
)
from .hydraulis import (
    HydraulisLatencyModel,
    HydraulisMemoryModel,
    HydraulisTrainingCost,
)
from .inventory import ParameterInventory, dtype_bytes, parameter_inventory
from .linear import (
    LinearTrainingCost,
    LinearTrainingLatency,
    LinearTrainingMemory,
)

TRAINING_MODEL_REGISTRY = {
    "hydraulis": HydraulisTrainingCost,
}

__all__ = [
    "AnalyticalTrainingCost",
    "AnalyticalTrainingLatency",
    "AnalyticalTrainingMemory",
    "FixedTrainingCost",
    "FixedTrainingLatency",
    "FixedTrainingMemory",
    "HydraulisLatencyModel",
    "HydraulisMemoryModel",
    "HydraulisTrainingCost",
    "LinearTrainingCost",
    "LinearTrainingLatency",
    "LinearTrainingMemory",
    "ParameterInventory",
    "TRAINING_MODEL_REGISTRY",
    "TrainingCostEstimate",
    "TrainingCostModel",
    "TrainingLatencyModel",
    "TrainingMemoryModel",
    "TrainingOutOfDomainError",
    "TrainingParallelism",
    "TrainingWorkload",
    "dtype_bytes",
    "parameter_inventory",
]
