"""Adapter for pinned Megatron-Bridge performance recipes."""

from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..hf import model_descriptor_from_yaml
from ..schema import (
    HardwareDescriptor,
    ModelDescriptor,
    ProfileRecord,
    ProfileValidationError,
    RankMemory,
    RunDescriptor,
    SequenceStats,
    SoftwareDescriptor,
    TrainingPayload,
)
from ..search_space import TrainingCandidate

PINNED_BRIDGE_VERSION = "0.4.2"
PINNED_BRIDGE_COMMIT = "c810129341a84e58f4cbed3093f70668a088c028"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_BRIDGE_DIR = _REPO_ROOT / "third_party" / "Megatron-Bridge"


def infer_gpu_type(gpu_name: str) -> str:
    """Map a hardware GPU name onto a Megatron-Bridge `--gpu` token."""
    normalized = gpu_name.lower().replace(" ", "")
    for token in ("b200", "b100", "h200", "h100", "a800", "a100", "l40s", "l40"):
        if token in normalized:
            return token
    raise ProfileValidationError(
        f"Cannot infer Megatron-Bridge gpu type from gpu_name {gpu_name!r}."
    )


def infer_bridge_recipe(model: ModelDescriptor) -> tuple[str, str]:
    """Infer the official performance recipe from a HuggingFace model descriptor."""
    model_type = str(model.extra.get("model_type") or "").lower()
    architecture = str(model.architecture or "").lower()
    identity = f"{model_type} {architecture} {model.name.lower()}"
    params = model.parameter_count or 0
    size_b = max(1, int(round(params / 1e9))) if params else None
    if "qwen3" in identity:
        if size_b is None:
            raise ProfileValidationError("Qwen3 recipe inference requires parameter_count.")
        return "qwen", f"qwen3_{size_b}b"
    raise ProfileValidationError(
        f"Cannot infer Megatron-Bridge recipe from model {model.name!r}."
    )


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


@dataclass(frozen=True)
class MegatronBridgeConfig:
    """Configuration for candidate execution and result normalization."""

    bridge_dir: Path
    output_dir: Path
    model: ModelDescriptor
    hardware: HardwareDescriptor
    experiment_name: str = ""
    model_family_name: str = "qwen"
    model_recipe_name: str = "qwen3_8b"
    gpu_type: str = "h100"
    compute_dtype: str = "bf16"
    optimizer: str = "adam"
    parameter_dtype: str = "bf16"
    gradient_dtype: str = "fp32"
    optimizer_state_dtype: str = "fp32"
    distributed_optimizer: bool = True
    max_steps: int = 4
    warmup_steps: int = 1
    profile_start_step: int = 1
    profile_end_step: int = 4
    extra_args: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    slurm_args: dict[str, Any] = field(default_factory=dict)
    timer_tag: str = "iteration-time"
    timer_unit: str = "ms"
    backend_version: str = PINNED_BRIDGE_VERSION
    source_commit: str = PINNED_BRIDGE_COMMIT
    allow_incomplete: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.warmup_steps < 0:
            raise ProfileValidationError("max_steps must be positive and warmup_steps must be non-negative.")
        if self.profile_start_step < 0 or self.profile_end_step <= self.profile_start_step:
            raise ProfileValidationError("Profiler step bounds are invalid.")
        if self.timer_unit not in {"s", "ms"}:
            raise ProfileValidationError("timer_unit must be s or ms.")

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, base_dir: Path | None = None) -> MegatronBridgeConfig:
        root = base_dir or Path.cwd()

        def resolve(path_value: str) -> Path:
            path = Path(path_value)
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        profiling = value.get("profiling", {})
        training = value.get("training", {})
        model = model_descriptor_from_yaml(value["model"], base_dir=root)
        hardware = HardwareDescriptor.from_dict(value["hardware"])
        family, recipe = infer_bridge_recipe(model)
        return cls(
            bridge_dir=resolve(value["bridge_dir"]) if value.get("bridge_dir") else _DEFAULT_BRIDGE_DIR,
            output_dir=resolve(value.get("output_dir", "profile_output")),
            model=model,
            hardware=hardware,
            experiment_name=str(value.get("experiment_name", "")),
            model_family_name=str(value.get("model_family_name", family)),
            model_recipe_name=str(value.get("model_recipe_name", recipe)),
            gpu_type=str(value.get("gpu_type", infer_gpu_type(hardware.gpu_name))),
            compute_dtype=str(training.get("compute_dtype", "bf16")),
            optimizer=str(training.get("optimizer", "adam")),
            parameter_dtype=str(training.get("parameter_dtype", "bf16")),
            gradient_dtype=str(training.get("gradient_dtype", "fp32")),
            optimizer_state_dtype=str(training.get("optimizer_state_dtype", "fp32")),
            distributed_optimizer=bool(training.get("distributed_optimizer", True)),
            max_steps=int(training.get("max_steps", 4)),
            warmup_steps=int(training.get("warmup_steps", 1)),
            profile_start_step=int(profiling.get("start_step", 1)),
            profile_end_step=int(profiling.get("end_step", 4)),
            extra_args=tuple(str(item) for item in value.get("extra_args", [])),
            environment={str(key): str(item) for key, item in value.get("environment", {}).items()},
            slurm_args=dict(value.get("slurm", {})),
            timer_tag=str(profiling.get("timer_tag", "iteration-time")),
            timer_unit=str(profiling.get("timer_unit", "ms")),
            allow_incomplete=bool(profiling.get("allow_incomplete", False)),
        )

    @property
    def raw_root(self) -> Path:
        if self.experiment_name:
            return self.output_dir / "raw" / "megatron_bridge" / self.experiment_name
        return self.output_dir / "raw" / "megatron_bridge"


def write_candidate_manifest(
    path: str | Path,
    candidates: list[TrainingCandidate],
) -> Path:
    """Write constrained candidates in a stable JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def read_candidate_manifest(path: str | Path) -> list[TrainingCandidate]:
    """Read candidates generated by DryRun-RL."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != 1:
        raise ProfileValidationError("Unsupported candidate manifest schema.")
    return [TrainingCandidate.from_dict(item) for item in value["candidates"]]


def _effective_sequence_shape(candidate: TrainingCandidate) -> tuple[int, ...]:
    padded_length = max(candidate.sequence_shape)
    return (padded_length,) * candidate.micro_batch_size


def _recompute_cli_args(candidate: TrainingCandidate) -> list[str]:
    if candidate.recompute in {"", "none", "off"}:
        return []
    num_layers = "1" if candidate.recompute == "full" else str(candidate.recompute)
    return ["--recompute_num_layers", num_layers]


def _recompute_hydra_overrides(candidate: TrainingCandidate) -> list[str]:
    if candidate.recompute != "full":
        return []
    return [
        "model.recompute_granularity=full",
        "model.recompute_method=uniform",
        "model.recompute_num_layers=1",
    ]


def _common_recipe_args(
    config: MegatronBridgeConfig,
    candidate: TrainingCandidate,
    resolved_config_path: Path,
) -> list[str]:
    parallelism = candidate.parallelism
    sequence_length = max(candidate.sequence_shape)
    args = [
        "--model_family_name",
        config.model_family_name,
        "--model_recipe_name",
        config.model_recipe_name,
        "--num_gpus",
        str(parallelism.world_size),
        "--gpu",
        config.gpu_type,
        "--compute_dtype",
        config.compute_dtype,
        "--task",
        "pretrain",
        "--data",
        "mock",
        "--max_steps",
        str(config.max_steps),
        "--seq_length",
        str(sequence_length),
        "--tensor_model_parallel_size",
        str(parallelism.tp),
        "--pipeline_model_parallel_size",
        str(parallelism.pp),
        "--context_parallel_size",
        str(parallelism.cp),
        "--micro_batch_size",
        str(candidate.micro_batch_size),
        "--global_batch_size",
        str(candidate.global_batch_size),
        "--save_config_filepath",
        str(resolved_config_path),
    ]
    args.extend(_recompute_cli_args(candidate))
    vocab_size = config.model.extra.get("vocab_size")
    if vocab_size is not None and "--vocab_size" not in config.extra_args:
        args.extend(("--vocab_size", str(vocab_size)))
    args.extend(config.extra_args)
    return args


def prepare_bridge_candidate(
    config: MegatronBridgeConfig,
    candidate: TrainingCandidate,
    *,
    backend: str = "local",
) -> tuple[list[str], Path]:
    """Prepare one local or Slurm command and its immutable manifest."""
    if backend not in {"local", "slurm"}:
        raise ProfileValidationError(f"Unsupported Megatron-Bridge backend {backend!r}.")
    run_dir = config.raw_root / candidate.candidate_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_dir / "ConfigContainer.yaml"
    common_args = _common_recipe_args(config, candidate, resolved_config_path)

    if backend == "local":
        recipe_script = config.bridge_dir / "scripts" / "performance" / "run_recipe.py"
        command = [
            "torchrun",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc-per-node",
            str(candidate.parallelism.world_size),
            str(recipe_script),
            *common_args,
            *_recompute_hydra_overrides(candidate),
            "profiling.use_pytorch_profiler=true",
            f"profiling.profile_step_start={config.profile_start_step}",
            f"profiling.profile_step_end={config.profile_end_step}",
            "profiling.record_memory_history=true",
            f"profiling.memory_snapshot_path={run_dir / 'snapshot.pickle'}",
            "logger.log_timers_to_tensorboard=true",
            "logger.log_interval=1",
        ]
    else:
        setup_script = config.bridge_dir / "scripts" / "performance" / "setup_experiment.py"
        command = [
            sys.executable,
            str(setup_script),
            *common_args,
            "--pytorch_profiler",
            "true",
            "--record_memory_history",
            "true",
            "--profiling_start_step",
            str(config.profile_start_step),
            "--profiling_stop_step",
            str(config.profile_end_step),
            "--nemo_home",
            str(run_dir / "nemo_run"),
            "--wandb_experiment_name",
            candidate.candidate_id,
        ]
        for key, raw_value in config.slurm_args.items():
            option = f"--{key.replace('_', '-')}"
            if isinstance(raw_value, bool):
                if raw_value:
                    command.append(option)
            elif isinstance(raw_value, list):
                command.append(option)
                command.extend(str(item) for item in raw_value)
            elif raw_value is not None:
                command.extend((option, str(raw_value)))

    manifest = {
        "adapter": "megatron_bridge",
        "backend": backend,
        "candidate": candidate.to_dict(),
        "requested_sequence_shape": list(candidate.sequence_shape),
        "effective_sequence_shape": list(_effective_sequence_shape(candidate)),
        "command": command,
        "resolved_config": str(resolved_config_path),
        "backend_version": config.backend_version,
        "source_commit": config.source_commit,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return command, run_dir


def run_bridge_candidate(
    config: MegatronBridgeConfig,
    candidate: TrainingCandidate,
    *,
    backend: str = "local",
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    """Execute one candidate in an independent process or submit it to Slurm."""
    command, run_dir = prepare_bridge_candidate(config, candidate, backend=backend)
    status_path = run_dir / "status.json"
    if resume and status_path.exists():
        with status_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("status") == "success":
            return run_dir
    if dry_run:
        if backend == "slurm":
            command.append("--dryrun")
        return run_dir

    environment = os.environ.copy()
    environment.update(config.environment)
    started = time.time()
    if backend == "slurm":
        result = subprocess.run(command, check=False, env=environment)
        status = "submitted" if result.returncode == 0 else "failed"
    else:
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                check=False,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        status = "success" if result.returncode == 0 else "failed"
    status_payload = {
        "status": status,
        "returncode": result.returncode,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "duration_s": time.time() - started,
        "command": command,
    }
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return run_dir


def _load_sidecar_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "dryrun_metrics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ProfileValidationError(f"Expected a JSON object in {path}.")
    return value


def _validate_resolved_config(
    run_dir: Path,
    manifest: dict[str, Any],
    candidate: TrainingCandidate,
) -> Path | None:
    configured = Path(manifest.get("resolved_config", run_dir / "ConfigContainer.yaml"))
    path = configured if configured.exists() else run_dir / "ConfigContainer.yaml"
    if not path.exists():
        return None
    from omegaconf import OmegaConf

    resolved = OmegaConf.load(path)
    expected = {
        "model.tensor_model_parallel_size": candidate.parallelism.tp,
        "model.pipeline_model_parallel_size": candidate.parallelism.pp,
        "model.context_parallel_size": candidate.parallelism.cp,
        "train.micro_batch_size": candidate.micro_batch_size,
        "train.global_batch_size": candidate.global_batch_size,
    }
    for key, expected_value in expected.items():
        actual = OmegaConf.select(resolved, key)
        if actual is not None and int(actual) != expected_value:
            raise ProfileValidationError(
                f"Resolved ConfigContainer value {key}={actual!r} does not match "
                f"candidate value {expected_value!r}."
            )
    return path


def _load_tensorboard_times(
    run_dir: Path,
    *,
    timer_tag: str,
    timer_unit: str,
) -> tuple[float, ...]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return ()
    samples: list[float] = []
    for event_path in sorted(run_dir.rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        matching = [tag for tag in scalar_tags if timer_tag.lower() in tag.lower()]
        for tag in matching:
            samples.extend(float(event.value) for event in accumulator.Scalars(tag))
    scale = 0.001 if timer_unit == "ms" else 1.0
    return tuple(value * scale for value in samples)


def _rank_from_path(path: Path) -> int:
    match = re.search(r"(?:rank|r)[=_-]?(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _snapshot_peak(snapshot: dict[str, Any]) -> tuple[int, int]:
    allocated = 0
    reserved = 0
    peak_allocated = 0
    peak_reserved = 0
    traces = snapshot.get("device_traces", [])
    events = [event for device_events in traces for event in device_events]
    for event in events:
        action = event.get("action")
        size = int(event.get("size", 0))
        if action == "alloc":
            allocated += size
        elif action in {"free_completed", "free"}:
            allocated = max(0, allocated - size)
        elif action == "segment_alloc":
            reserved += size
        elif action == "segment_free":
            reserved = max(0, reserved - size)
        peak_allocated = max(peak_allocated, allocated)
        peak_reserved = max(peak_reserved, reserved)
    if peak_reserved == 0:
        segments = snapshot.get("segments", [])
        peak_reserved = sum(int(segment.get("total_size", 0)) for segment in segments)
        peak_allocated = sum(int(segment.get("allocated_size", 0)) for segment in segments)
    return peak_allocated, max(peak_allocated, peak_reserved)


def _load_memory_snapshots(run_dir: Path) -> tuple[RankMemory, ...]:
    rank_memory: list[RankMemory] = []
    for path in sorted(run_dir.rglob("snapshot*.pickle")):
        with path.open("rb") as handle:
            snapshot = pickle.load(handle)
        if not isinstance(snapshot, dict):
            continue
        allocated, reserved = _snapshot_peak(snapshot)
        rank_memory.append(
            RankMemory(
                rank=_rank_from_path(path),
                allocated_bytes=allocated,
                reserved_bytes=reserved,
            )
        )
    return tuple(rank_memory)


def _is_oom(run_dir: Path) -> bool:
    log_path = run_dir / "run.log"
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    return "cuda out of memory" in text or "torch.outofmemoryerror" in text


def _status(run_dir: Path) -> tuple[str, float | None, str | None]:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return "failed", None, "status.json is missing."
    with status_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    status = value.get("status")
    if status == "success":
        return "success", value.get("duration_s"), None
    if _is_oom(run_dir):
        return "oom", value.get("duration_s"), "CUDA out of memory."
    return "failed", value.get("duration_s"), f"Process exited with status {status!r}."


def collect_bridge_record(
    config: MegatronBridgeConfig,
    candidate: TrainingCandidate,
    *,
    run_dir: str | Path | None = None,
) -> ProfileRecord:
    """Normalize one resolved candidate, timers, memory history, and status."""
    source_dir = Path(run_dir) if run_dir is not None else config.raw_root / candidate.candidate_id
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Megatron-Bridge manifest does not exist: {manifest_path}.")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    resolved_config_path = _validate_resolved_config(source_dir, manifest, candidate)
    status, duration_s, error = _status(source_dir)
    metrics = _load_sidecar_metrics(source_dir) or {}
    step_times = tuple(float(item) for item in metrics.get("step_times_s", []))
    if not step_times and status == "success":
        step_times = _load_tensorboard_times(
            source_dir,
            timer_tag=config.timer_tag,
            timer_unit=config.timer_unit,
        )
    if config.warmup_steps and len(step_times) > config.warmup_steps:
        step_times = step_times[config.warmup_steps :]
    if status == "success" and not step_times and not config.allow_incomplete:
        raise ProfileValidationError(
            f"No training step timer was found for successful candidate {candidate.candidate_id!r}."
        )

    if "rank_memory" in metrics:
        rank_memory = tuple(RankMemory.from_dict(item) for item in metrics["rank_memory"])
    else:
        rank_memory = _load_memory_snapshots(source_dir)
    if status == "success" and not rank_memory and not config.allow_incomplete:
        raise ProfileValidationError(
            f"No rank memory peak was found for successful candidate {candidate.candidate_id!r}."
        )

    effective_shape = tuple(int(item) for item in manifest["effective_sequence_shape"])
    if "microbatch_sequence_lengths" in metrics:
        shapes = [tuple(int(item) for item in shape) for shape in metrics["microbatch_sequence_lengths"]]
    else:
        shapes = [effective_shape] * candidate.num_microbatches
    microbatches = tuple(SequenceStats.from_lengths(shape) for shape in shapes)
    if len(microbatches) != candidate.num_microbatches:
        raise ProfileValidationError("Collected microbatch sequence shapes do not match num_microbatches.")

    peak_allocated = max((item.allocated_bytes for item in rank_memory), default=None)
    peak_reserved = max((item.reserved_bytes for item in rank_memory), default=None)
    timers = {
        str(name): tuple(float(item) for item in values)
        for name, values in metrics.get("timers_s", {}).items()
    }
    hardware = replace(config.hardware, gpu_count=int(candidate.parallelism.world_size))
    raw_artifact = str(source_dir.relative_to(config.output_dir))
    run = RunDescriptor(
        run_id=f"megatron-bridge-{candidate.candidate_id}",
        repeat=candidate.repeat,
        status=status,
        duration_s=duration_s,
        raw_artifact=raw_artifact,
        error=error,
    )
    payload = TrainingPayload(
        micro_batch_size=candidate.micro_batch_size,
        global_batch_size=candidate.global_batch_size,
        num_microbatches=candidate.num_microbatches,
        schedule=candidate.schedule,
        recompute=candidate.recompute,
        optimizer=config.optimizer,
        microbatches=microbatches,
        step_times_s=step_times,
        timers_s=timers,
        rank_memory=rank_memory,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        parameter_dtype=config.parameter_dtype,
        gradient_dtype=config.gradient_dtype,
        optimizer_state_dtype=config.optimizer_state_dtype,
        distributed_optimizer=config.distributed_optimizer,
        parameter_count=int(metrics.get("parameter_count", config.model.parameter_count or 0)) or None,
        warmup_steps=config.warmup_steps,
    )
    software = SoftwareDescriptor(
        backend="megatron_bridge",
        backend_version=config.backend_version,
        source_commit=config.source_commit,
        command=tuple(str(item) for item in manifest["command"]),
    )
    return ProfileRecord(
        record_type="training",
        run=run,
        model=config.model,
        hardware=hardware,
        software=software,
        parallelism=candidate.parallelism,
        payload=payload,
        tags={
            "source": "megatron-bridge-performance-recipe",
            "resolved_config": (
                str(resolved_config_path.relative_to(config.output_dir))
                if resolved_config_path is not None
                and resolved_config_path.is_relative_to(config.output_dir)
                else ""
            ),
        },
    )


def collect_bridge_records(
    config: MegatronBridgeConfig,
    candidates: list[TrainingCandidate],
) -> list[ProfileRecord]:
    """Collect all candidate directories that have adapter manifests."""
    records: list[ProfileRecord] = []
    for candidate in candidates:
        run_dir = config.raw_root / candidate.candidate_id
        if not (run_dir / "manifest.json").exists():
            if config.allow_incomplete:
                continue
            raise FileNotFoundError(f"Candidate output is missing: {run_dir}.")
        records.append(collect_bridge_record(config, candidate, run_dir=run_dir))
    return records
