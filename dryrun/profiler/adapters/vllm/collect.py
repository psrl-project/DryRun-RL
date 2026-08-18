"""Normalize paired vLLM timing/trace sweep shards into `ProfileRecord` observations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from ...schema import (
    Parallelism,
    ProfileRecord,
    ProfileValidationError,
    RunDescriptor,
    SoftwareDescriptor,
)
from .config import VLLMSweepConfig
from .parse import (
    _TRACE_RANK_PATTERN,
    _IterationDetail,
    aggregate_rank_traces,
    apply_iteration_timings,
    parse_sweep_log,
    worker_trace_files,
)


def _value_from_keys(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return default


def _parallelism_from_result(value: dict[str, Any], config: VLLMSweepConfig) -> Parallelism:
    tp = int(
        _value_from_keys(
            value,
            "tensor-parallel-size",
            "tensor_parallel_size",
            default=config.serve_args.get("tensor_parallel_size", config.hardware.gpu_count),
        )
    )
    pp = int(
        _value_from_keys(
            value,
            "pipeline-parallel-size",
            "pipeline_parallel_size",
            default=config.serve_args.get("pipeline_parallel_size", 1),
        )
    )
    dp = int(
        _value_from_keys(
            value,
            "data-parallel-size",
            "data_parallel_size",
            default=config.serve_args.get("data_parallel_size", 1),
        )
    )
    return Parallelism(tp=tp, pp=pp, dp=dp, cp=1)


def _manifest_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_manifest_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _experiment_dirs(
    config: VLLMSweepConfig,
    phase: str,
    measurement: str,
) -> dict[str, Path]:
    legacy = config.raw_experiment_dir(phase, measurement)
    shard_pattern = re.compile(rf"{re.escape(legacy.name)}-(?P<shard>s\d+)")
    shards = {
        match.group("shard"): path
        for path in sorted(legacy.parent.glob(f"{legacy.name}-s*"))
        if path.is_dir()
        and (match := shard_pattern.fullmatch(path.name)) is not None
    }
    if shards:
        return shards
    return {"": legacy} if legacy.is_dir() else {}


def _group_worker_traces(
    trace_results: Sequence[tuple[Path, int]],
    trace_paths: Sequence[Path],
) -> list[list[Path]]:
    expected_rank_traces = sum(world_size for _, world_size in trace_results)
    rank_zero_traces = [
        path
        for path in trace_paths
        if (match := _TRACE_RANK_PATTERN.search(path.name)) is not None
        and int(match.group("rank")) == 0
    ]
    if len(trace_paths) == expected_rank_traces:
        grouped_traces: list[list[Path]] = []
        cursor = 0
        for _, world_size in trace_results:
            grouped_traces.append(list(trace_paths[cursor : cursor + world_size]))
            cursor += world_size
        return grouped_traces
    if len(rank_zero_traces) == len(trace_results):
        return [[path] for path in rank_zero_traces]
    if len(trace_paths) == len(trace_results):
        return [[path] for path in trace_paths]
    raise ProfileValidationError(
        "Cannot pair vLLM runs with worker traces: "
        f"{len(trace_results)} runs, expected {expected_rank_traces} rank traces, "
        f"found {len(trace_paths)} worker traces."
    )


def _index_vllm_shard(
    config: VLLMSweepConfig,
    phase: str,
    shard_id: str,
    timing_root: Path,
    trace_root: Path,
) -> list[tuple[str, dict[str, Any]]]:
    timing_by_relative_path = {
        path.relative_to(timing_root): path
        for path in timing_root.rglob("run=*.json")
    }
    trace_result_paths = sorted(
        trace_root.rglob("run=*.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    if not timing_by_relative_path:
        raise ProfileValidationError(
            f"No vLLM timing run JSON files were found under {timing_root}."
        )
    trace_relative_paths = {
        path.relative_to(trace_root) for path in trace_result_paths
    }
    if set(timing_by_relative_path) != trace_relative_paths:
        raise ProfileValidationError(
            "Matched vLLM sweeps produced different parameter/run paths in "
            f"shard {shard_id or 'legacy'}."
        )
    timing_result_paths = [
        timing_by_relative_path[path.relative_to(trace_root)]
        for path in trace_result_paths
    ]
    trace_dir = config.trace_dir(phase, shard_id or None)
    trace_paths = worker_trace_files(trace_dir)
    if not trace_paths:
        raise ProfileValidationError(
            f"No worker profiler traces were found under {trace_dir}."
        )
    timing_iterations = parse_sweep_log(
        config.sweep_log_path(phase, "timing", shard_id or None)
    )
    trace_iterations = parse_sweep_log(
        config.sweep_log_path(phase, "trace", shard_id or None)
    )

    trace_results: list[tuple[Path, int]] = []
    for result_path in trace_result_paths:
        with result_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        trace_results.append(
            (
                result_path,
                int(_parallelism_from_result(value, config).world_size),
            )
        )
    grouped_traces = _group_worker_traces(trace_results, trace_paths)

    entries: list[tuple[str, dict[str, Any]]] = []
    for timing_result, (trace_result, trace_world_size), run_traces in zip(
        timing_result_paths,
        trace_results,
        grouped_traces,
        strict=True,
    ):
        combo_path = str(trace_result.relative_to(trace_root))
        timing_details = timing_iterations.get(timing_result.resolve(), [])
        trace_details = trace_iterations.get(trace_result.resolve(), [])
        if not timing_details or not trace_details:
            raise ProfileValidationError(
                "Missing scheduler iteration timings for matched vLLM runs: "
                f"timing={timing_result}, trace={trace_result}. "
                "vLLM only emits Iteration(...) logs at INFO; "
                "VLLM_LOGGING_LEVEL=WARN disables LoggingStatLogger."
            )
        with timing_result.open(encoding="utf-8") as handle:
            timing_value = json.load(handle)
        timing_world_size = int(
            _parallelism_from_result(timing_value, config).world_size
        )
        if timing_world_size != trace_world_size:
            raise ProfileValidationError(
                "Matched vLLM timing and trace runs use different parallelism."
            )
        entries.append(
            (
                combo_path,
                {
                    "combo_path": combo_path,
                    "timing_result": _manifest_path(
                        timing_result,
                        config.output_dir,
                    ),
                    "trace_result": _manifest_path(
                        trace_result,
                        config.output_dir,
                    ),
                    "traces": [
                        _manifest_path(trace_path, config.output_dir)
                        for trace_path in run_traces
                    ],
                    "timing_iterations": [
                        detail.to_dict() for detail in timing_details
                    ],
                    "trace_iterations": [
                        detail.to_dict() for detail in trace_details
                    ],
                },
            )
        )
    return entries


def index_vllm_traces(
    config: VLLMSweepConfig,
    phase: str,
) -> Path:
    """Pair matched timing/trace runs within shards, then merge by combo path."""
    timing_dirs = _experiment_dirs(config, phase, "timing")
    trace_dirs = _experiment_dirs(config, phase, "trace")
    if not timing_dirs or not trace_dirs:
        raise FileNotFoundError(
            "Matched vLLM experiment directories do not exist for "
            f"phase {phase!r}."
        )
    if set(timing_dirs) != set(trace_dirs):
        raise ProfileValidationError(
            "Matched vLLM timing and trace sweeps produced different shard sets: "
            f"timing={sorted(timing_dirs)}, trace={sorted(trace_dirs)}."
        )

    entries_by_combo: dict[str, dict[str, Any]] = {}
    for shard_id in sorted(trace_dirs):
        for combo_path, entry in _index_vllm_shard(
            config,
            phase,
            shard_id,
            timing_dirs[shard_id],
            trace_dirs[shard_id],
        ):
            if combo_path in entries_by_combo:
                raise ProfileValidationError(
                    f"Duplicate vLLM combo path {combo_path!r} across sweep shards."
                )
            entries_by_combo[combo_path] = entry
    manifest = {
        "schema_version": 1,
        "measurement_source": "vllm-unprofiled-timing-plus-detailed-trace",
        "phase": phase,
        "entries": [
            entries_by_combo[key]
            for key in sorted(entries_by_combo)
        ],
    }
    path = config.trace_manifest_path(phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def collect_vllm_records(
    config: VLLMSweepConfig,
    phase: str,
) -> list[ProfileRecord]:
    """Normalize official per-scheduler-step trace annotations into records."""
    timing_dirs = _experiment_dirs(config, phase, "timing")
    trace_dirs = _experiment_dirs(config, phase, "trace")
    if not timing_dirs or not trace_dirs:
        raise FileNotFoundError(
            f"Matched vLLM experiment directories do not exist for phase {phase!r}."
        )
    manifest_path = config.trace_manifest_path(phase)
    if not manifest_path.exists():
        index_vllm_traces(config, phase)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    records: list[ProfileRecord] = []
    for entry in manifest["entries"]:
        timing_result_path = _resolve_manifest_path(
            entry["timing_result"],
            config.output_dir,
        )
        trace_result_path = _resolve_manifest_path(
            entry["trace_result"],
            config.output_dir,
        )
        trace_paths = [
            _resolve_manifest_path(item, config.output_dir)
            for item in entry["traces"]
        ]
        with timing_result_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ProfileValidationError(
                f"Expected a JSON object in {timing_result_path}."
            )
        parallelism = _parallelism_from_result(value, config)
        hardware = replace(config.hardware, gpu_count=int(parallelism.world_size))
        timing_relative_path = _manifest_path(
            timing_result_path,
            config.output_dir,
        )
        trace_result_relative_path = _manifest_path(
            trace_result_path,
            config.output_dir,
        )
        trace_relative_paths = [
            _manifest_path(trace_path, config.output_dir)
            for trace_path in trace_paths
        ]
        combo_identity = entry.get("combo_path", timing_relative_path)
        identity = hashlib.sha256(
            f"{phase}:{combo_identity}".encode("utf-8")
        ).hexdigest()[:20]
        run_number = int(
            value.get(
                "run_number",
                timing_result_path.stem.removeprefix("run="),
            )
        )
        run = RunDescriptor(
            run_id=f"vllm-{identity}",
            repeat=run_number,
            status="success" if int(value.get("completed", 0)) > 0 else "failed",
            duration_s=float(value["duration"]) if value.get("duration") is not None else None,
            raw_artifact=trace_relative_paths[0],
            error=None if int(value.get("completed", 0)) > 0 else "No requests completed.",
        )
        software = SoftwareDescriptor(
            backend="vllm",
            backend_version=config.backend_version,
            source_commit=config.source_commit,
            command=(config.executable, "bench", "sweep", "serve"),
        )
        trace_scheduler_details = [
            _IterationDetail.from_dict(item)
            for item in entry.get("trace_iterations", [])
        ]
        timing_scheduler_details = [
            _IterationDetail.from_dict(item)
            for item in entry.get("timing_iterations", [])
        ]
        traced_steps = apply_iteration_timings(
            aggregate_rank_traces(trace_paths),
            trace_scheduler_details,
            iteration_tag="trace_scheduler_iteration",
        )
        steps = apply_iteration_timings(
            traced_steps,
            timing_scheduler_details,
            iteration_tag="timing_scheduler_iteration",
        )
        for step in steps:
            if phase == "prefill" and step.payload.phase == "decode":
                continue
            if phase == "decode" and step.payload.phase == "prefill":
                continue
            records.append(
                ProfileRecord(
                    record_type="rollout",
                    run=run,
                    model=config.model,
                    hardware=hardware,
                    software=software,
                    parallelism=parallelism,
                    payload=step.payload,
                    tags={
                        "phase": phase,
                        "observed_phase": step.payload.phase,
                        "source": "vllm-unprofiled-timing-plus-detailed-trace",
                        "latency_source": "unprofiled-scheduler-iteration-elapsed-ms",
                        "timing_benchmark_artifact": timing_relative_path,
                        "trace_benchmark_artifact": trace_result_relative_path,
                        "trace_artifacts": trace_relative_paths,
                        "trace_metrics": step.metrics,
                    },
                )
            )
    return records
