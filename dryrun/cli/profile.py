"""YAML-driven official vLLM and Megatron-Bridge profiling workflows."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .._logging import dryrun_logger, setup_logging
from ..profiler.adapters import (
    MegatronBridgeConfig,
    VLLMSweepConfig,
    collect_bridge_records,
    collect_vllm_records,
    format_vllm_job_command,
    prepare_bridge_candidate,
    prepare_vllm_jobs,
    read_candidate_manifest,
    run_bridge_candidate,
    run_vllm_profiles,
    write_candidate_manifest,
)
from ..profiler.io import write_records
from ..profiler.search_space import (
    ModelConstraints,
    TrainingCandidate,
    enumerate_training_candidates,
)


def _load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    value = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {config_path}.")
    return value, config_path.parent


def _phases(config: VLLMSweepConfig, requested: str) -> list[str]:
    if requested == "all":
        return list(config.phase_parameters)
    if requested not in config.phase_parameters:
        raise ValueError(f"Phase {requested!r} is not configured.")
    return [requested]


def _vllm_config(path: str | Path) -> VLLMSweepConfig:
    value, base_dir = _load_config(path)
    return VLLMSweepConfig.from_dict(value, base_dir=base_dir)


def _max_parallel_arg(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max parallel must be 'auto' or a positive integer"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max parallel must be positive")
    return parsed


def _bridge_config(path: str | Path) -> tuple[MegatronBridgeConfig, dict[str, Any]]:
    value, base_dir = _load_config(path)
    return MegatronBridgeConfig.from_dict(value, base_dir=base_dir), value


def _candidate_path(config: MegatronBridgeConfig, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return config.output_dir / "control" / "megatron_bridge" / "candidates.json"


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def _generate_bridge_candidates(
    config: MegatronBridgeConfig,
    raw_config: dict[str, Any],
) -> list[TrainingCandidate]:
    search = raw_config.get("search_space", {})
    if config.model.num_layers is None:
        raise ValueError("HuggingFace config must provide num_layers for candidate generation.")
    if config.model.num_attention_heads is None:
        raise ValueError("HuggingFace config must provide num_attention_heads for candidate generation.")
    if config.model.num_query_groups is None:
        raise ValueError("HuggingFace config must provide num_query_groups for candidate generation.")
    constraints = ModelConstraints(
        num_layers=int(config.model.num_layers),
        num_attention_heads=int(config.model.num_attention_heads),
        num_query_groups=int(config.model.num_query_groups),
    )
    world_size = int(search.get("world_size", config.hardware.gpu_count))
    micro_batch_size = int(search.get("micro_batch_size", 1))
    pipeline_parallel_sizes = _as_int_list(search["pipeline_parallel_sizes"])
    if "num_microbatches" in search:
        num_microbatches = _as_int_list(search["num_microbatches"])
    else:
        num_microbatches = sorted(set([1, *pipeline_parallel_sizes]))
    if "sequence_lengths" in search:
        sequence_shapes = [
            [int(length)] * micro_batch_size
            for length in search["sequence_lengths"]
        ]
    elif "sequence_shapes" in search:
        sequence_shapes = [
            [int(length) for length in shape]
            for shape in search["sequence_shapes"]
        ]
    else:
        raise ValueError("search_space.sequence_lengths is required.")
    recompute = search.get("recompute", search.get("recompute_modes", "full"))
    recompute_modes = [str(recompute)] if isinstance(recompute, str) else [str(item) for item in recompute]
    return enumerate_training_candidates(
        world_sizes=[world_size],
        tensor_parallel_sizes=_as_int_list(search["tensor_parallel_sizes"]),
        pipeline_parallel_sizes=pipeline_parallel_sizes,
        context_parallel_sizes=_as_int_list(search.get("context_parallel_sizes", [1])),
        micro_batch_sizes=[micro_batch_size],
        num_microbatches=num_microbatches,
        sequence_shapes=sequence_shapes,
        schedules=[str(item) for item in search.get("schedules", ["1f1b"])],
        recompute_modes=recompute_modes,
        constraints=constraints,
        repeats=1,
    )


def _select_candidates(
    candidates: list[TrainingCandidate],
    *,
    candidate_id: str | None,
    index: int | None,
) -> list[TrainingCandidate]:
    if candidate_id is not None:
        selected = [candidate for candidate in candidates if candidate.candidate_id == candidate_id]
        if not selected:
            raise ValueError(f"Unknown candidate_id {candidate_id!r}.")
        return selected
    if index is not None:
        try:
            return [candidates[index]]
        except IndexError as exc:
            raise ValueError(f"Candidate index {index} is out of range.") from exc
    return candidates


def cmd_vllm_generate(args: argparse.Namespace) -> None:
    config = _vllm_config(args.config)
    jobs = prepare_vllm_jobs(
        config,
        _phases(config, args.phase),
    )
    for job in jobs:
        dryrun_logger.info(
            f"Prepared vLLM {job.phase!r} {job.measurement} sweep: "
            f"{format_vllm_job_command(job)}."
        )


def cmd_vllm_run(args: argparse.Namespace) -> None:
    config = _vllm_config(args.config)
    max_parallel = (
        config.max_parallel
        if args.max_parallel is None
        else args.max_parallel
    )
    run_vllm_profiles(
        config,
        _phases(config, args.phase),
        max_parallel=max_parallel,
        dry_run=args.dry_run,
        resume=args.resume,
    )


def cmd_vllm_collect(args: argparse.Namespace) -> None:
    config = _vllm_config(args.config)
    records = [
        record
        for phase in _phases(config, args.phase)
        for record in collect_vllm_records(config, phase)
    ]
    output = Path(args.output) if args.output else config.output_dir / "profiles" / "vllm.jsonl"
    write_records(output, records)
    dryrun_logger.info(f"Wrote {len(records)} normalized vLLM observations to {output}.")


def cmd_megatron_generate(args: argparse.Namespace) -> None:
    config, raw_config = _bridge_config(args.config)
    candidates = _generate_bridge_candidates(config, raw_config)
    output = _candidate_path(config, args.candidates)
    write_candidate_manifest(output, candidates)
    dryrun_logger.info(f"Wrote {len(candidates)} constrained candidates to {output}.")


def cmd_megatron_run(args: argparse.Namespace) -> None:
    config, _ = _bridge_config(args.config)
    path = _candidate_path(config, args.candidates)
    candidates = _select_candidates(
        read_candidate_manifest(path),
        candidate_id=args.candidate_id,
        index=args.index,
    )
    failures = 0
    for candidate in candidates:
        if args.dry_run:
            command, _ = prepare_bridge_candidate(config, candidate, backend=args.backend)
            dryrun_logger.info(
                f"Prepared Megatron-Bridge candidate {candidate.candidate_id!r}: "
                f"{shlex.join(command)}."
            )
            continue
        try:
            run_bridge_candidate(
                config,
                candidate,
                backend=args.backend,
                resume=args.resume,
            )
        except subprocess.CalledProcessError as exc:
            failures += 1
            dryrun_logger.warning(
                f"Candidate {candidate.candidate_id!r} exited with code {exc.returncode}; "
                "its status and raw log were preserved."
            )
    if failures:
        dryrun_logger.warning(f"{failures} Megatron-Bridge candidates failed or ran out of memory.")


def cmd_megatron_collect(args: argparse.Namespace) -> None:
    config, _ = _bridge_config(args.config)
    path = _candidate_path(config, args.candidates)
    candidates = _select_candidates(
        read_candidate_manifest(path),
        candidate_id=args.candidate_id,
        index=args.index,
    )
    records = collect_bridge_records(config, candidates)
    output = (
        Path(args.output)
        if args.output
        else config.output_dir / "profiles" / "megatron_bridge.jsonl"
    )
    write_records(output, records)
    dryrun_logger.info(
        f"Wrote {len(records)} normalized Megatron-Bridge observations to {output}."
    )


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML workflow configuration.")


def _add_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidates", help="Candidate manifest path.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--candidate-id", help="Run or collect one candidate ID.")
    selection.add_argument("--index", type=int, help="Run or collect one candidate index.")


def build_parser() -> argparse.ArgumentParser:
    """Build the profiling command tree for tests and console entrypoints."""
    parser = argparse.ArgumentParser(
        prog="dryrun-profile",
        description="Run official profilers and normalize their raw artifacts.",
    )
    backends = parser.add_subparsers(dest="backend_name", required=True)

    vllm = backends.add_parser("vllm", help="Use vLLM bench sweep serve.")
    vllm_workflows = vllm.add_subparsers(dest="workflow", required=True)
    for name, handler in (
        ("generate", cmd_vllm_generate),
        ("run", cmd_vllm_run),
        ("collect", cmd_vllm_collect),
    ):
        workflow = vllm_workflows.add_parser(name)
        _add_config_argument(workflow)
        workflow.add_argument("--phase", choices=["all", "prefill", "decode"], default="all")
        if name == "run":
            workflow.add_argument("--dry-run", action="store_true")
            workflow.add_argument("--resume", action="store_true")
            workflow.add_argument(
                "--max-parallel",
                type=_max_parallel_arg,
                default=None,
                metavar="auto|N",
                help=(
                    "Maximum concurrent vLLM servers. Defaults to run.max_parallel "
                    "or all available GPU slots."
                ),
            )
        if name == "collect":
            workflow.add_argument("--output")
        workflow.set_defaults(func=handler)

    megatron = backends.add_parser("megatron", help="Use Megatron-Bridge recipes.")
    megatron_workflows = megatron.add_subparsers(dest="workflow", required=True)
    generate = megatron_workflows.add_parser("generate")
    _add_config_argument(generate)
    generate.add_argument("--candidates")
    generate.set_defaults(func=cmd_megatron_generate)

    run = megatron_workflows.add_parser("run")
    _add_config_argument(run)
    _add_candidate_arguments(run)
    run.add_argument("--backend", choices=["local", "slurm"], default="local")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=cmd_megatron_run)

    collect = megatron_workflows.add_parser("collect")
    _add_config_argument(collect)
    _add_candidate_arguments(collect)
    collect.add_argument("--output")
    collect.set_defaults(func=cmd_megatron_collect)
    return parser


def main() -> None:
    """Run a profiling workflow."""
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
