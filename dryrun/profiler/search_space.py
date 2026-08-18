"""Constrained candidate generation for Megatron-Bridge profiling sweeps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

from .schema import Parallelism, ProfileValidationError


@dataclass(frozen=True)
class ModelConstraints:
    """Describe architecture divisibility constraints used by Megatron-Core."""

    num_layers: int
    num_attention_heads: int
    num_query_groups: int

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_attention_heads", "num_query_groups"):
            if getattr(self, name) <= 0:
                raise ProfileValidationError(f"{name} must be positive.")


@dataclass(frozen=True)
class TrainingCandidate:
    """Describe one independently executable training profiling candidate."""

    parallelism: Parallelism
    micro_batch_size: int
    global_batch_size: int
    num_microbatches: int
    sequence_shape: tuple[int, ...]
    schedule: str = "1f1b"
    recompute: str = "none"
    repeat: int = 0

    def __post_init__(self) -> None:
        if self.parallelism.cp != 1:
            raise ProfileValidationError("Training candidate V1 only supports CP=1.")
        if self.schedule.lower() != "1f1b":
            raise ProfileValidationError("Training candidate V1 only supports the 1F1B schedule.")
        if self.micro_batch_size <= 0 or self.global_batch_size <= 0 or self.num_microbatches <= 0:
            raise ProfileValidationError("Batch sizes and num_microbatches must be positive.")
        expected = self.micro_batch_size * self.parallelism.dp * self.num_microbatches
        if self.global_batch_size != expected:
            raise ProfileValidationError(
                "global_batch_size must equal MBS * DP * num_microbatches, "
                f"got {self.global_batch_size} != {expected}."
            )
        if len(self.sequence_shape) != self.micro_batch_size:
            raise ProfileValidationError("sequence_shape must contain one length per microbatch sequence.")
        if any(length <= 0 for length in self.sequence_shape):
            raise ProfileValidationError("All sequence lengths must be positive.")
        if self.repeat < 0:
            raise ProfileValidationError("repeat must be non-negative.")

    @property
    def candidate_id(self) -> str:
        """Return a compact, deterministic identity without encoding a lookup protocol."""
        value = json.dumps(self.to_dict(include_id=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if include_id:
            value["candidate_id"] = self.candidate_id
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingCandidate:
        return cls(
            parallelism=Parallelism.from_dict(value["parallelism"]),
            micro_batch_size=int(value["micro_batch_size"]),
            global_batch_size=int(value["global_batch_size"]),
            num_microbatches=int(value["num_microbatches"]),
            sequence_shape=tuple(int(item) for item in value["sequence_shape"]),
            schedule=str(value.get("schedule", "1f1b")),
            recompute=str(value.get("recompute", "none")),
            repeat=int(value.get("repeat", 0)),
        )


def tensor_parallel_is_valid(tp: int, constraints: ModelConstraints) -> bool:
    """Check attention head and grouped-query attention divisibility."""
    if constraints.num_attention_heads % tp != 0:
        return False
    return constraints.num_query_groups % tp == 0 or tp % constraints.num_query_groups == 0


def enumerate_training_candidates(
    *,
    world_sizes: list[int],
    tensor_parallel_sizes: list[int],
    pipeline_parallel_sizes: list[int],
    micro_batch_sizes: list[int],
    num_microbatches: list[int],
    sequence_shapes: list[list[int] | tuple[int, ...]],
    constraints: ModelConstraints,
    repeats: int = 1,
    context_parallel_sizes: list[int] | None = None,
    schedules: list[str] | None = None,
    recompute_modes: list[str] | None = None,
) -> list[TrainingCandidate]:
    """Enumerate only candidates that satisfy V1 topology and batch constraints."""
    if repeats <= 0:
        raise ProfileValidationError("repeats must be positive.")
    if any(value <= 0 for value in num_microbatches):
        raise ProfileValidationError("num_microbatches values must be positive.")
    cp_values = context_parallel_sizes or [1]
    schedule_values = schedules or ["1f1b"]
    recompute_values = recompute_modes or ["full"]
    candidates: list[TrainingCandidate] = []

    dimensions = product(
        sorted(set(world_sizes)),
        sorted(set(tensor_parallel_sizes)),
        sorted(set(pipeline_parallel_sizes)),
        sorted(set(cp_values)),
        sorted(set(micro_batch_sizes)),
        sorted(set(num_microbatches)),
        sequence_shapes,
        schedule_values,
        recompute_values,
    )
    for world_size, tp, pp, cp, mbs, nmb, raw_shape, schedule, recompute in dimensions:
        if cp != 1 or schedule.lower() != "1f1b":
            continue
        model_parallel = tp * pp * cp
        if world_size % model_parallel != 0:
            continue
        if not tensor_parallel_is_valid(tp, constraints):
            continue
        if constraints.num_layers % pp != 0:
            continue
        dp = world_size // model_parallel
        shape = tuple(int(item) for item in raw_shape)
        if len(shape) != mbs or any(length <= 0 for length in shape):
            continue

        parallelism = Parallelism(tp=tp, pp=pp, dp=dp, cp=cp, world_size=world_size)
        for repeat in range(repeats):
            candidates.append(
                TrainingCandidate(
                    parallelism=parallelism,
                    micro_batch_size=mbs,
                    global_batch_size=mbs * dp * nmb,
                    num_microbatches=nmb,
                    sequence_shape=shape,
                    schedule=schedule,
                    recompute=recompute,
                    repeat=repeat,
                )
            )
    return candidates
