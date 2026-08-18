"""Rollout latency models and workload vocabulary."""

from .base import RolloutCostModel, RolloutWorkload
from .distserve import DistServe
from .psrl import PSRLFitted
from .roofline import LinearLPS, UnifiedRoofline

ROLLOUT_MODEL_REGISTRY = {
    "roofline": UnifiedRoofline,
    "psrl": PSRLFitted,
    "distserve": DistServe,
}

__all__ = [
    "DistServe",
    "LinearLPS",
    "PSRLFitted",
    "ROLLOUT_MODEL_REGISTRY",
    "RolloutCostModel",
    "RolloutWorkload",
    "UnifiedRoofline",
]
