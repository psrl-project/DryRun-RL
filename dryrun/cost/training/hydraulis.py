"""Hydraulis Appendix C inspired V1 training latency and memory model."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from dryrun.cost.artifact import ArtifactEntry, CostArtifact, FitMetrics
from dryrun.profiler.io import profile_digest
from dryrun.profiler.schema import Parallelism, ProfileRecord, TrainingPayload

from .base import (
    TrainingCostModel,
    TrainingLatencyModel,
    TrainingMemoryModel,
    TrainingOutOfDomainError,
    TrainingParallelism,
    TrainingWorkload,
)
from .inventory import parameter_inventory


@dataclass(frozen=True)
class _CellKey:
    tp: int
    pp: int
    cp: int
    schedule: str
    recompute: str
    optimizer: str
    parameter_dtype: str
    gradient_dtype: str
    optimizer_state_dtype: str
    distributed_optimizer: bool
    parameter_count: int

    @property
    def parallelism(self) -> Parallelism:
        return Parallelism(tp=self.tp, pp=self.pp, dp=1, cp=self.cp)

    def selector(self, component: str) -> dict[str, Any]:
        return {
            "component": component,
            "schedule": self.schedule,
            "recompute": self.recompute,
            "optimizer": self.optimizer,
            "parameter_dtype": self.parameter_dtype,
            "gradient_dtype": self.gradient_dtype,
            "optimizer_state_dtype": self.optimizer_state_dtype,
            "distributed_optimizer": self.distributed_optimizer,
        }


def _training_records(records: list[ProfileRecord]) -> tuple[dict[_CellKey, list[ProfileRecord]], list[ProfileRecord]]:
    materialized = [record for record in records if record.record_type == "training"]
    if not materialized:
        raise ValueError("No training profiling records are available.")
    groups: dict[_CellKey, list[ProfileRecord]] = defaultdict(list)
    for record in materialized:
        payload = record.payload
        if not isinstance(payload, TrainingPayload):
            continue
        if record.parallelism.cp != 1 or payload.schedule.lower() != "1f1b":
            continue
        parameter_count = payload.parameter_count or record.model.parameter_count
        if parameter_count is None:
            raise ValueError("Hydraulis memory fitting requires parameter_count.")
        key = _CellKey(
            tp=record.parallelism.tp,
            pp=record.parallelism.pp,
            cp=record.parallelism.cp,
            schedule=payload.schedule.lower(),
            recompute=payload.recompute,
            optimizer=payload.optimizer,
            parameter_dtype=payload.parameter_dtype,
            gradient_dtype=payload.gradient_dtype,
            optimizer_state_dtype=payload.optimizer_state_dtype,
            distributed_optimizer=payload.distributed_optimizer,
            parameter_count=int(parameter_count),
        )
        groups[key].append(record)
    if not groups:
        raise ValueError("No CP=1, 1F1B training records are available.")
    return dict(groups), materialized


def _split_records(
    records: list[ProfileRecord],
    *,
    validation_fraction: float,
) -> tuple[list[ProfileRecord], list[ProfileRecord]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    run_ids = sorted({record.run.run_id for record in records})
    if validation_fraction == 0 or len(run_ids) < 2:
        return records, records
    ranked = sorted(
        run_ids,
        key=lambda run_id: hashlib.sha256(run_id.encode("utf-8")).digest(),
    )
    count = max(1, round(len(ranked) * validation_fraction))
    validation_ids = set(ranked[:count])
    train = [record for record in records if record.run.run_id not in validation_ids]
    validation = [record for record in records if record.run.run_id in validation_ids]
    return (train, validation) if train else (records, records)


def _latency_features(payload: TrainingPayload, pp: int) -> tuple[float, float, float]:
    q_values = [microbatch.sum_squared_tokens for microbatch in payload.microbatches]
    t_values = [microbatch.sum_tokens for microbatch in payload.microbatches]
    q_effective = sum(q_values) + (pp - 1) * max(q_values)
    t_effective = sum(t_values) + (pp - 1) * max(t_values)
    pipeline_slots = payload.num_microbatches + pp - 1
    return float(q_effective), float(t_effective), float(pipeline_slots)


def _activation_feature(payload: TrainingPayload, pp: int) -> float:
    max_tokens = max(microbatch.sum_tokens for microbatch in payload.microbatches)
    in_flight_microbatches = min(payload.num_microbatches, pp)
    return float(max_tokens * in_flight_microbatches)


def _scaled_nonnegative_fit(
    design: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    scales = np.maximum(np.max(np.abs(design), axis=0), 1.0)
    scaled = design / scales
    initial = np.maximum(np.linalg.lstsq(scaled, target, rcond=None)[0], 0)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return scaled @ parameters - target

    first = least_squares(
        residual,
        x0=np.maximum(initial, np.finfo(np.float64).eps),
        bounds=(0, np.inf),
        loss="soft_l1",
        max_nfev=50_000,
    )
    raw_residual = residual(first.x)
    median = float(np.median(raw_residual))
    mad = float(np.median(np.abs(raw_residual - median)))
    if mad <= np.finfo(np.float64).eps:
        inliers = np.ones(len(target), dtype=bool)
    else:
        inliers = np.abs(raw_residual - median) <= 4.5 * 1.4826 * mad
    if int(np.sum(inliers)) > design.shape[1]:
        second = least_squares(
            lambda parameters: residual(parameters)[inliers],
            x0=first.x,
            bounds=(0, np.inf),
            loss="soft_l1",
            max_nfev=50_000,
        )
        fitted = second.x
    else:
        fitted = first.x
        inliers = np.ones(len(target), dtype=bool)
    return fitted / scales, inliers, int(np.linalg.matrix_rank(design))


def _latency_arrays(
    records: list[ProfileRecord],
    pp: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float, float]] = []
    targets: list[float] = []
    for record in records:
        payload = record.payload
        if record.run.status != "success" or not isinstance(payload, TrainingPayload):
            continue
        q_effective, t_effective, pipeline_slots = _latency_features(payload, pp)
        for step_time in payload.step_times_s:
            rows.append((q_effective, t_effective, pipeline_slots, 1.0))
            targets.append(step_time)
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _fit_latency_entry(
    key: _CellKey,
    records: list[ProfileRecord],
    *,
    validation_fraction: float,
    require_full_rank: bool,
) -> ArtifactEntry | None:
    successful = [
        record
        for record in records
        if record.run.status == "success"
        and isinstance(record.payload, TrainingPayload)
        and record.payload.step_times_s
    ]
    if not successful:
        return None
    train_records, validation_records = _split_records(
        successful,
        validation_fraction=validation_fraction,
    )
    train_x, train_y = _latency_arrays(train_records, key.pp)
    validation_x, validation_y = _latency_arrays(validation_records, key.pp)
    if len(train_y) < 4:
        raise ValueError(f"Hydraulis latency needs at least four samples for {key!r}.")
    parameters, inliers, rank = _scaled_nonnegative_fit(train_x, train_y)
    if require_full_rank and rank < 4:
        raise ValueError(
            f"Hydraulis latency design matrix is rank {rank}, expected 4 for {key!r}. "
            "Add sequence-shape and pipeline-microbatch coverage."
        )
    train_predicted = train_x @ parameters
    validation_predicted = validation_x @ parameters
    all_features = [_latency_features(record.payload, key.pp) for record in successful]
    feature_ranges = {
        "q_effective": (
            min(item[0] for item in all_features),
            max(item[0] for item in all_features),
        ),
        "t_effective": (
            min(item[1] for item in all_features),
            max(item[1] for item in all_features),
        ),
        "pipeline_slots": (
            min(item[2] for item in all_features),
            max(item[2] for item in all_features),
        ),
        "micro_batch_size": (
            float(min(record.payload.micro_batch_size for record in successful)),
            float(max(record.payload.micro_batch_size for record in successful)),
        ),
    }
    return ArtifactEntry(
        parallelism=key.parallelism,
        selector=key.selector("latency"),
        coefficients={
            "a_q": float(parameters[0]),
            "b_t": float(parameters[1]),
            "c_pipeline": float(parameters[2]),
            "update_overhead": float(parameters[3]),
        },
        train_metrics=FitMetrics.from_predictions(
            train_y,
            train_predicted,
            n_outliers=int(np.sum(~inliers)),
            design_rank=rank,
        ),
        validation_metrics=FitMetrics.from_predictions(
            validation_y,
            validation_predicted,
            design_rank=rank,
        ),
        feature_ranges=feature_ranges,
        diagnostics={
            "formula": "a_q*Q_eff+b_t*T_eff+c_pipeline*(num_microbatches+PP-1)+update_overhead",
            "dp_latency": "ideal-local-workload-scaling",
        },
    )


def _record_inventory(
    key: _CellKey,
    record: ProfileRecord,
) -> float:
    parallelism = TrainingParallelism.from_profile_parallelism(record.parallelism)
    return parameter_inventory(
        parameter_count=key.parameter_count,
        parallelism=parallelism,
        parameter_dtype=key.parameter_dtype,
        gradient_dtype=key.gradient_dtype,
        optimizer_state_dtype=key.optimizer_state_dtype,
        distributed_optimizer=key.distributed_optimizer,
    ).total_bytes


def _memory_arrays(
    key: _CellKey,
    records: list[ProfileRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[float] = []
    targets: list[float] = []
    static_bytes: list[float] = []
    for record in records:
        payload = record.payload
        if record.run.status != "success" or not isinstance(payload, TrainingPayload):
            continue
        peak = payload.peak_reserved_bytes
        if peak is None:
            continue
        static = _record_inventory(key, record)
        features.append(_activation_feature(payload, key.pp))
        targets.append(float(peak))
        static_bytes.append(static)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(static_bytes, dtype=np.float64),
    )


def _fit_memory_entry(
    key: _CellKey,
    records: list[ProfileRecord],
    *,
    validation_fraction: float,
    require_full_rank: bool,
) -> ArtifactEntry | None:
    successful = [
        record
        for record in records
        if record.run.status == "success"
        and isinstance(record.payload, TrainingPayload)
        and record.payload.peak_reserved_bytes is not None
    ]
    oom_records = [record for record in records if record.run.status == "oom"]
    if not successful and not oom_records:
        return None
    if successful:
        train_records, validation_records = _split_records(
            successful,
            validation_fraction=validation_fraction,
        )
        train_feature, train_target, train_static = _memory_arrays(key, train_records)
        validation_feature, validation_target, validation_static = _memory_arrays(
            key,
            validation_records,
        )
        train_residual = np.maximum(train_target - train_static, 0)
        design = np.column_stack((train_feature, np.ones(len(train_feature))))
        parameters, inliers, rank = _scaled_nonnegative_fit(design, train_residual)
        if require_full_rank and rank < 2:
            raise ValueError(
                f"Hydraulis memory design matrix is rank {rank}, expected 2 for {key!r}. "
                "Add at least two activation-load points."
            )
    else:
        train_feature = np.asarray([], dtype=np.float64)
        train_target = np.asarray([], dtype=np.float64)
        train_static = np.asarray([], dtype=np.float64)
        validation_feature = train_feature
        validation_target = train_target
        validation_static = train_static
        parameters = np.asarray([0.0, 0.0])
        inliers = np.asarray([], dtype=bool)
        rank = 0

    constrained_oom = 0
    intercept_increase = 0.0
    for record in oom_records:
        payload = record.payload
        if not isinstance(payload, TrainingPayload) or record.hardware.gpu_memory_bytes is None:
            continue
        feature = _activation_feature(payload, key.pp)
        static = _record_inventory(key, record)
        required_residual = record.hardware.gpu_memory_bytes + 1 - static
        deficit = required_residual - (parameters[0] * feature + parameters[1])
        if deficit > intercept_increase:
            intercept_increase = deficit
        constrained_oom += 1
    parameters[1] += max(intercept_increase, 0.0)

    train_predicted = train_static + parameters[0] * train_feature + parameters[1]
    validation_predicted = (
        validation_static + parameters[0] * validation_feature + parameters[1]
    )
    all_payloads = [
        record.payload
        for record in successful + oom_records
        if isinstance(record.payload, TrainingPayload)
    ]
    activation_features = [_activation_feature(payload, key.pp) for payload in all_payloads]
    dp_values = [float(record.parallelism.dp) for record in successful + oom_records]
    return ArtifactEntry(
        parallelism=key.parallelism,
        selector=key.selector("memory"),
        coefficients={
            "activation_slope": float(parameters[0]),
            "residual_constant": float(parameters[1]),
            "parameter_count": float(key.parameter_count),
        },
        train_metrics=FitMetrics.from_predictions(
            train_target,
            train_predicted,
            n_outliers=int(np.sum(~inliers)) if inliers.size else 0,
            design_rank=rank,
        ),
        validation_metrics=FitMetrics.from_predictions(
            validation_target,
            validation_predicted,
            design_rank=rank,
        ),
        feature_ranges={
            "activation_feature": (
                min(activation_features),
                max(activation_features),
            ),
            "profiled_dp": (min(dp_values), max(dp_values)),
        },
        diagnostics={
            "memory_metric": "peak_reserved_bytes",
            "static_inventory": "uniform-layer-parameter-gradient-master-adam",
            "optimizer_sharding": "distributed-optimizer-dp",
            "oom_constraints": constrained_oom,
            "oom_intercept_increase": max(intercept_increase, 0.0),
        },
    )


def _selector(
    *,
    component: str,
    workload: TrainingWorkload,
    parameter_dtype: str,
    gradient_dtype: str,
    optimizer_state_dtype: str,
    distributed_optimizer: bool,
) -> dict[str, Any]:
    return {
        "component": component,
        "schedule": workload.schedule.lower(),
        "recompute": workload.recompute,
        "optimizer": workload.optimizer,
        "parameter_dtype": parameter_dtype,
        "gradient_dtype": gradient_dtype,
        "optimizer_state_dtype": optimizer_state_dtype,
        "distributed_optimizer": distributed_optimizer,
    }


def _check_feature(
    entry: ArtifactEntry,
    name: str,
    value: float,
    *,
    allow_extrapolation: bool,
) -> None:
    lower, upper = entry.feature_ranges[name]
    if not allow_extrapolation and not lower <= value <= upper:
        raise TrainingOutOfDomainError(
            f"Feature {name!r}={value} is outside the profiled range [{lower}, {upper}]."
        )


class HydraulisLatencyModel(TrainingLatencyModel):
    """Evaluate fitted Hydraulis latency cells with explicit OOD checks."""

    def __init__(
        self,
        artifact: CostArtifact,
        *,
        parameter_dtype: str,
        gradient_dtype: str,
        optimizer_state_dtype: str,
        distributed_optimizer: bool,
        allow_extrapolation: bool = False,
    ):
        self.artifact = artifact
        self.parameter_dtype = parameter_dtype
        self.gradient_dtype = gradient_dtype
        self.optimizer_state_dtype = optimizer_state_dtype
        self.distributed_optimizer = distributed_optimizer
        self.allow_extrapolation = allow_extrapolation

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> float:
        canonical = parallelism.to_profile_parallelism(canonical_dp=True)
        entry = self.artifact.entry_for(
            canonical,
            selector=_selector(
                component="latency",
                workload=workload,
                parameter_dtype=self.parameter_dtype,
                gradient_dtype=self.gradient_dtype,
                optimizer_state_dtype=self.optimizer_state_dtype,
                distributed_optimizer=self.distributed_optimizer,
            ),
        )
        microbatches = workload.critical_rank_microbatches(parallelism)
        payload = TrainingPayload(
            micro_batch_size=workload.micro_batch_size,
            global_batch_size=workload.global_batch_size,
            num_microbatches=len(microbatches),
            schedule=workload.schedule,
            recompute=workload.recompute,
            optimizer=workload.optimizer,
            microbatches=microbatches,
        )
        q_effective, t_effective, pipeline_slots = _latency_features(payload, parallelism.pp)
        for name, value in (
            ("q_effective", q_effective),
            ("t_effective", t_effective),
            ("pipeline_slots", pipeline_slots),
            ("micro_batch_size", float(workload.micro_batch_size)),
        ):
            _check_feature(
                entry,
                name,
                value,
                allow_extrapolation=self.allow_extrapolation,
            )
        coefficients = entry.coefficients
        latency = (
            coefficients["a_q"] * q_effective
            + coefficients["b_t"] * t_effective
            + coefficients["c_pipeline"] * pipeline_slots
            + coefficients["update_overhead"]
        )
        if latency <= 0:
            raise ValueError("Hydraulis latency prediction must be positive.")
        return latency


class HydraulisMemoryModel(TrainingMemoryModel):
    """Combine static parameter inventory with a fitted activation residual."""

    def __init__(
        self,
        artifact: CostArtifact,
        *,
        parameter_dtype: str,
        gradient_dtype: str,
        optimizer_state_dtype: str,
        distributed_optimizer: bool,
        allow_extrapolation: bool = False,
    ):
        self.artifact = artifact
        self.parameter_dtype = parameter_dtype
        self.gradient_dtype = gradient_dtype
        self.optimizer_state_dtype = optimizer_state_dtype
        self.distributed_optimizer = distributed_optimizer
        self.allow_extrapolation = allow_extrapolation

    def predict(
        self,
        workload: TrainingWorkload,
        parallelism: TrainingParallelism,
    ) -> int:
        canonical = parallelism.to_profile_parallelism(canonical_dp=True)
        entry = self.artifact.entry_for(
            canonical,
            selector=_selector(
                component="memory",
                workload=workload,
                parameter_dtype=self.parameter_dtype,
                gradient_dtype=self.gradient_dtype,
                optimizer_state_dtype=self.optimizer_state_dtype,
                distributed_optimizer=self.distributed_optimizer,
            ),
        )
        microbatches = workload.critical_rank_microbatches(parallelism)
        payload = TrainingPayload(
            micro_batch_size=workload.micro_batch_size,
            global_batch_size=workload.global_batch_size,
            num_microbatches=len(microbatches),
            schedule=workload.schedule,
            recompute=workload.recompute,
            optimizer=workload.optimizer,
            microbatches=microbatches,
        )
        feature = _activation_feature(payload, parallelism.pp)
        _check_feature(
            entry,
            "activation_feature",
            feature,
            allow_extrapolation=self.allow_extrapolation,
        )
        parameter_count = int(entry.coefficients["parameter_count"])
        static = parameter_inventory(
            parameter_count=parameter_count,
            parallelism=parallelism,
            parameter_dtype=self.parameter_dtype,
            gradient_dtype=self.gradient_dtype,
            optimizer_state_dtype=self.optimizer_state_dtype,
            distributed_optimizer=self.distributed_optimizer,
        ).total_bytes
        predicted = (
            static
            + entry.coefficients["activation_slope"] * feature
            + entry.coefficients["residual_constant"]
        )
        return int(math.ceil(predicted))


class HydraulisTrainingCost(TrainingCostModel):
    """Compose Hydraulis V1 latency and memory from one artifact."""

    model_type = "hydraulis"

    @classmethod
    def fit(
        cls,
        records: list[ProfileRecord],
        *,
        validation_fraction: float = 0.2,
        require_full_rank: bool = True,
    ) -> CostArtifact:
        """Fit latency and memory entries for every supported discrete cell."""
        groups, materialized = _training_records(records)
        model_names = {record.model.name for record in materialized}
        gpu_names = {record.hardware.gpu_name for record in materialized}
        if len(model_names) != 1 or len(gpu_names) != 1:
            raise ValueError("A training artifact must target one model and one GPU type.")
        first = materialized[0]
        model = asdict(first.model)
        hardware = asdict(first.hardware)
        hardware["gpu_count"] = sorted({record.hardware.gpu_count for record in materialized})
        entries: list[ArtifactEntry] = []
        for key, cell_records in groups.items():
            latency_entry = _fit_latency_entry(
                key,
                cell_records,
                validation_fraction=validation_fraction,
                require_full_rank=require_full_rank,
            )
            memory_entry = _fit_memory_entry(
                key,
                cell_records,
                validation_fraction=validation_fraction,
                require_full_rank=require_full_rank,
            )
            if latency_entry is not None:
                entries.append(latency_entry)
            if memory_entry is not None:
                entries.append(memory_entry)
        if not entries:
            raise ValueError("No Hydraulis entries could be fitted.")
        return CostArtifact(
            kind="training",
            model_type=cls.model_type,
            model=model,
            hardware=hardware,
            profile_digest=profile_digest(materialized),
            entries=tuple(entries),
            diagnostics={
                "supported_cp": [1],
                "supported_schedule": ["1f1b"],
                "layer_partition": "uniform",
                "dp_communication": "ignored",
                "activation_checkpoint_recompute": "discrete-profiled-selector",
            },
        )

    @classmethod
    def from_artifact(
        cls,
        artifact_or_path: CostArtifact | str | Path,
        *,
        parameter_dtype: str | None = None,
        gradient_dtype: str | None = None,
        optimizer_state_dtype: str | None = None,
        distributed_optimizer: bool | None = None,
        allow_extrapolation: bool = False,
    ) -> HydraulisTrainingCost:
        """Load composable latency and memory models from one artifact."""
        artifact = (
            artifact_or_path
            if isinstance(artifact_or_path, CostArtifact)
            else CostArtifact.load(artifact_or_path)
        )
        if artifact.kind != "training" or artifact.model_type != cls.model_type:
            raise ValueError("Artifact does not contain a Hydraulis training model.")
        selectors = [entry.selector for entry in artifact.entries]

        def infer(name: str, explicit: Any) -> Any:
            if explicit is not None:
                return explicit
            values = {selector[name] for selector in selectors}
            if len(values) != 1:
                raise ValueError(f"Artifact has multiple {name} values; select one explicitly.")
            return values.pop()

        selected_parameter_dtype = str(infer("parameter_dtype", parameter_dtype))
        selected_gradient_dtype = str(infer("gradient_dtype", gradient_dtype))
        selected_optimizer_state_dtype = str(
            infer("optimizer_state_dtype", optimizer_state_dtype)
        )
        selected_distributed_optimizer = bool(
            infer("distributed_optimizer", distributed_optimizer)
        )
        common = {
            "artifact": artifact,
            "parameter_dtype": selected_parameter_dtype,
            "gradient_dtype": selected_gradient_dtype,
            "optimizer_state_dtype": selected_optimizer_state_dtype,
            "distributed_optimizer": selected_distributed_optimizer,
            "allow_extrapolation": allow_extrapolation,
        }
        return cls(
            latency_model=HydraulisLatencyModel(**common),
            memory_model=HydraulisMemoryModel(**common),
        )
