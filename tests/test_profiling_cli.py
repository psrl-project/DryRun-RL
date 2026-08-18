"""CLI command-tree and dry-run command construction tests."""

from __future__ import annotations

from pathlib import Path

from dryrun.cli.profile import build_parser as build_profile_parser
from dryrun.cli.regress import build_parser as build_regress_parser
from dryrun.profiler.adapters.megatron_bridge import (
    MegatronBridgeConfig,
    prepare_bridge_candidate,
)
from dryrun.profiler.adapters.vllm import (
    VLLMSweepConfig,
    format_vllm_job_command,
    prepare_vllm_jobs,
    prepare_vllm_sweeps,
)
from dryrun.profiler.schema import HardwareDescriptor, ModelDescriptor, Parallelism
from dryrun.profiler.search_space import TrainingCandidate


def test_profile_cli_exposes_generate_run_collect():
    parser = build_profile_parser()
    for backend in ("vllm", "megatron"):
        for workflow in ("generate", "run", "collect"):
            args = [backend, workflow, "--config", "config.yaml"]
            if backend == "megatron" and workflow == "run":
                args.extend(("--dry-run", "--index", "0"))
            parsed = parser.parse_args(args)
            assert parsed.backend_name == backend
            assert parsed.workflow == workflow


def test_regress_cli_exposes_model_owned_fitters():
    parser = build_regress_parser()
    rollout = parser.parse_args(
        [
            "rollout",
            "--model",
            "distserve",
            "--input",
            "profile.jsonl",
            "--output",
            "artifact.json",
        ]
    )
    training = parser.parse_args(
        [
            "training",
            "--input",
            "profile.jsonl",
            "--output",
            "artifact.json",
        ]
    )
    assert rollout.model == "distserve"
    assert training.latency_model == "hydraulis"
    assert training.memory_model == "hydraulis"


def test_vllm_generated_command_uses_official_detailed_trace(tmp_path: Path):
    config = VLLMSweepConfig(
        model=ModelDescriptor(name="Qwen3-8B"),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="test",
        phase_parameters={"prefill": ({"random-output-len": 1},)},
    )
    sweeps = prepare_vllm_sweeps(config, "prefill")
    timing_command, _ = sweeps["timing"]
    trace_command, _ = sweeps["trace"]
    assert timing_command[:4] == ["vllm", "bench", "sweep", "serve"]
    timing_bench = timing_command[timing_command.index("--bench-cmd") + 1]
    timing_serve = timing_command[timing_command.index("--serve-cmd") + 1]
    trace_bench = trace_command[trace_command.index("--bench-cmd") + 1]
    trace_serve = trace_command[trace_command.index("--serve-cmd") + 1]
    assert "--save-detailed" in timing_bench
    assert "--profile" not in timing_bench
    assert "--enable-logging-iteration-details" in timing_serve
    assert "--profile" in trace_bench
    assert '"detailed_trace_annotation":true' in trace_serve
    assert '"profiler":"torch"' in trace_serve


def test_vllm_parallel_jobs_shard_points_across_eight_gpus(tmp_path: Path):
    config = VLLMSweepConfig(
        model=ModelDescriptor(name="Qwen3-8B"),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="test",
        phase_parameters={
            "prefill": tuple(
                {
                    "random-input-len": 128 * (index + 1),
                    "random-output-len": 1,
                }
                for index in range(8)
            )
        },
    )
    jobs = prepare_vllm_jobs(
        config,
        ["prefill"],
        visible_devices=[str(index) for index in range(8)],
    )

    assert len(jobs) == 16
    assert {job.shard_id for job in jobs} == {
        f"s{index:03d}" for index in range(8)
    }
    for wave in {job.wave for job in jobs}:
        wave_jobs = [job for job in jobs if job.wave == wave]
        assert len(wave_jobs) == 8
        assert len({job.gpu_ids for job in wave_jobs}) == 8
        assert len({job.port for job in wave_jobs}) == 8
    assert all(len(job.gpu_ids) == 1 for job in jobs)
    assert all(job.shard_id in job.experiment_dir.name for job in jobs)
    first_serve = jobs[0].command[jobs[0].command.index("--serve-cmd") + 1]
    first_bench = jobs[0].command[jobs[0].command.index("--bench-cmd") + 1]
    assert f"--port {jobs[0].port}" in first_serve
    assert f"--base-url http://127.0.0.1:{jobs[0].port}" in first_bench
    trace_jobs = [job for job in jobs if job.measurement == "trace"]
    assert len({job.trace_dir for job in trace_jobs}) == 8
    assert len({job.log_path for job in jobs}) == 16
    assert "--dry-run" in format_vllm_job_command(jobs[0], dry_run=True)
    assert "CUDA_VISIBLE_DEVICES=" in format_vllm_job_command(jobs[0])


def test_vllm_parallel_jobs_pack_tp2_into_four_slots(tmp_path: Path):
    config = VLLMSweepConfig(
        model=ModelDescriptor(name="Qwen3-8B"),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=2),
        output_dir=tmp_path,
        experiment_name="test",
        serve_parameters=(
            {
                "tensor-parallel-size": 2,
                "pipeline-parallel-size": 1,
            },
        ),
        phase_parameters={
            "prefill": tuple(
                {
                    "random-input-len": 128 * (index + 1),
                    "random-output-len": 1,
                }
                for index in range(8)
            )
        },
    )
    jobs = prepare_vllm_jobs(
        config,
        ["prefill"],
        visible_devices=[str(index) for index in range(8)],
    )

    assert len(jobs) == 8
    assert all(job.world_size == 2 for job in jobs)
    assert all(len(job.gpu_ids) == 2 for job in jobs)
    for wave in {job.wave for job in jobs}:
        wave_jobs = [job for job in jobs if job.wave == wave]
        assert len(wave_jobs) == 4
        assert len({gpu for job in wave_jobs for gpu in job.gpu_ids}) == 8


def test_vllm_max_parallel_one_preserves_legacy_sweeps(tmp_path: Path):
    config = VLLMSweepConfig(
        model=ModelDescriptor(name="Qwen3-8B"),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="test",
        phase_parameters={
            "prefill": (
                {"random-input-len": 128, "random-output-len": 1},
                {"random-input-len": 256, "random-output-len": 1},
            )
        },
    )
    jobs = prepare_vllm_jobs(
        config,
        ["prefill"],
        max_parallel=1,
        visible_devices=[str(index) for index in range(8)],
    )

    assert len(jobs) == 2
    assert all(job.shard_id is None for job in jobs)
    assert [job.wave for job in jobs] == [0, 1]
    assert all(job.port is None for job in jobs)
    assert all("-s000" not in job.experiment_dir.name for job in jobs)


def test_vllm_run_cli_accepts_auto_or_bounded_parallelism():
    parser = build_profile_parser()
    automatic = parser.parse_args(
        [
            "vllm",
            "run",
            "--config",
            "config.yaml",
            "--max-parallel",
            "auto",
        ]
    )
    bounded = parser.parse_args(
        [
            "vllm",
            "run",
            "--config",
            "config.yaml",
            "--max-parallel",
            "4",
        ]
    )
    assert automatic.max_parallel == "auto"
    assert bounded.max_parallel == 4


def test_megatron_generated_command_uses_official_recipe(tmp_path: Path):
    bridge_dir = tmp_path / "Megatron-Bridge"
    config = MegatronBridgeConfig(
        bridge_dir=bridge_dir,
        output_dir=tmp_path,
        experiment_name="test",
        model=ModelDescriptor(name="Qwen3-8B", parameter_count=8_000_000_000),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
    )
    candidate = TrainingCandidate(
        parallelism=Parallelism(),
        micro_batch_size=1,
        global_batch_size=1,
        num_microbatches=1,
        sequence_shape=(512,),
    )
    command, _ = prepare_bridge_candidate(config, candidate)
    assert command[0] == "torchrun"
    assert str(bridge_dir / "scripts" / "performance" / "run_recipe.py") in command
    assert command[command.index("--model_family_name") + 1] == "qwen"
    assert "profiling.use_pytorch_profiler=true" in command
    assert "profiling.record_memory_history=true" in command
