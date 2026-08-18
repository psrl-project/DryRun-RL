"""Fixture tests for official profiler normalization."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dryrun.profiler.adapters.megatron_bridge import (
    MegatronBridgeConfig,
    collect_bridge_record,
    prepare_bridge_candidate,
)
from dryrun.profiler.adapters.vllm import VLLMSweepConfig, collect_vllm_records
from dryrun.profiler.adapters.vllm.execute import (
    _prefix_console_bytes,
    _run_vllm_job_pool,
    _sweep_runtime_environment,
)
from dryrun.profiler.adapters.vllm.prepare import VLLMSweepJob
from dryrun.profiler.adapters.vllm.parse import parse_sweep_log
from dryrun.profiler.io import profile_digest, read_records, write_records
from dryrun.profiler.schema import (
    HardwareDescriptor,
    ModelDescriptor,
    Parallelism,
    ProfileValidationError,
    TrainingPayload,
)
from dryrun.profiler.search_space import (
    ModelConstraints,
    TrainingCandidate,
    enumerate_training_candidates,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _model() -> ModelDescriptor:
    return ModelDescriptor(
        name="Qwen3-8B",
        parameter_count=10_000_000,
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_query_groups=8,
        extra={"intermediate_size": 12288},
    )


def _vllm_config(tmp_path: Path) -> VLLMSweepConfig:
    return VLLMSweepConfig(
        model=_model(),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="fixture",
        phase_parameters={
            "prefill": ({"random-output-len": 1},),
            "decode": ({"random-output-len": 3},),
        },
    )


@pytest.mark.parametrize(
    ("phase", "result_fixture", "trace_fixture", "expected_records"),
    [
        ("prefill", "prefill_run.json", "prefill_trace.json", 1),
        ("decode", "decode_run.json", "decode_trace.json", 2),
    ],
)
def test_vllm_detailed_trace_normalization(
    tmp_path: Path,
    phase: str,
    result_fixture: str,
    trace_fixture: str,
    expected_records: int,
):
    config = _vllm_config(tmp_path)
    result_paths = {}
    for measurement in ("timing", "trace"):
        run_dir = config.raw_experiment_dir(phase, measurement) / "point"
        run_dir.mkdir(parents=True)
        result_paths[measurement] = run_dir / "run=0.json"
        shutil.copy(
            FIXTURES / "vllm" / result_fixture,
            result_paths[measurement],
        )
    trace_dir = config.trace_dir(phase)
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "dp0_pp0_tp0_dcp0_ep0_rank0.123.pt.trace.json"
    if phase == "decode":
        trace_path = trace_path.with_suffix(f"{trace_path.suffix}.gz")
        with (
            (FIXTURES / "vllm" / trace_fixture).open("rb") as source,
            gzip.open(trace_path, "wb") as destination,
        ):
            shutil.copyfileobj(source, destination)
    else:
        shutil.copy(FIXTURES / "vllm" / trace_fixture, trace_path)
    for measurement in ("timing", "trace"):
        prefill_elapsed = 10.0 if measurement == "timing" else 11.0
        decode_elapsed = (
            (4.2, 4.4) if measurement == "timing" else (5.2, 5.4)
        )
        iteration_lines = {
            "prefill": [
                "Iteration(0): 2 context requests, 256 context tokens, "
                "0 generation requests, 0 generation tokens, "
                f"iteration elapsed time: {prefill_elapsed:.2f} ms, "
                "GPU KV cache usage: 1.0%",
            ],
            "decode": [
                "Iteration(0): 2 context requests, 384 context tokens, "
                "0 generation requests, 0 generation tokens, "
                "iteration elapsed time: 12.00 ms, GPU KV cache usage: 1.0%",
                "Iteration(1): 0 context requests, 0 context tokens, "
                "2 generation requests, 2 generation tokens, "
                f"iteration elapsed time: {decode_elapsed[0]:.2f} ms, "
                "GPU KV cache usage: 1.0%",
                "Iteration(2): 0 context requests, 0 context tokens, "
                "2 generation requests, 2 generation tokens, "
                f"iteration elapsed time: {decode_elapsed[1]:.2f} ms, "
                "GPU KV cache usage: 1.0%",
            ],
        }
        config.sweep_log_path(phase, measurement).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        profile_markers = (
            ["Traffic request rate: inf"]
            if measurement == "timing"
            else ["Profiler started"]
        )
        config.sweep_log_path(phase, measurement).write_text(
            "\n".join(
                [
                    f"Output file: {result_paths[measurement]}",
                    *profile_markers,
                    *iteration_lines[phase],
                    "Stopping profiler..."
                    if measurement == "trace"
                    else "[END BENCHMARK]",
                ]
            ),
            encoding="utf-8",
        )

    records = collect_vllm_records(config, phase)
    assert len(records) == expected_records
    assert all(record.record_type == "rollout" for record in records)
    assert all(record.payload.phase == phase for record in records)
    assert all(record.payload.request_count == 2 for record in records)
    if phase == "prefill":
        assert records[0].payload.sum_l2 == 2 * 128**2
        assert records[0].payload.latency_s == 0.01
    else:
        assert records[0].payload.ctxsum == 129 + 257
        assert records[0].payload.latency_s == pytest.approx(0.0042)
        assert records[0].tags["trace_metrics"]["gen_sqsk"] == 386

    output = tmp_path / f"{phase}.jsonl"
    write_records(output, records)
    assert read_records(output) == records
    assert profile_digest(output) == profile_digest(records)


def test_vllm_collect_merges_parallel_shards_by_combo_path(tmp_path: Path):
    config = VLLMSweepConfig(
        model=_model(),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="fixture",
        phase_parameters={
            "prefill": (
                {"random-input-len": 128, "random-output-len": 1},
                {"random-input-len": 256, "random-output-len": 1},
            )
        },
    )
    for shard_index in range(2):
        shard_id = f"s{shard_index:03d}"
        combo = f"BENCH--random-input-len={128 * (shard_index + 1)}"
        for measurement in ("timing", "trace"):
            run_dir = (
                config.raw_experiment_dir(
                    "prefill",
                    measurement,
                    shard_id,
                )
                / combo
            )
            run_dir.mkdir(parents=True)
            result_path = run_dir / "run=0.json"
            shutil.copy(
                FIXTURES / "vllm" / "prefill_run.json",
                result_path,
            )
            log_path = config.sweep_log_path(
                "prefill",
                measurement,
                shard_id,
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "\n".join(
                    [
                        f"Output file: {result_path}",
                        (
                            "Traffic request rate: inf"
                            if measurement == "timing"
                            else "Profiler started"
                        ),
                        "Iteration(0): 2 context requests, 256 context tokens, "
                        "0 generation requests, 0 generation tokens, "
                        "iteration elapsed time: 10.00 ms, "
                        "GPU KV cache usage: 1.0%",
                        (
                            "[END BENCHMARK]"
                            if measurement == "timing"
                            else "Stopping profiler..."
                        ),
                    ]
                ),
                encoding="utf-8",
            )
        trace_dir = config.trace_dir("prefill", shard_id)
        trace_dir.mkdir(parents=True)
        shutil.copy(
            FIXTURES / "vllm" / "prefill_trace.json",
            trace_dir
            / f"dp0_pp0_tp0_dcp0_ep0_rank0.{shard_index}.pt.trace.json",
        )

    records = collect_vllm_records(config, "prefill")
    manifest = json.loads(
        config.trace_manifest_path("prefill").read_text(encoding="utf-8")
    )

    assert len(records) == 2
    assert [entry["combo_path"] for entry in manifest["entries"]] == [
        "BENCH--random-input-len=128/run=0.json",
        "BENCH--random-input-len=256/run=0.json",
    ]
    assert all("-s00" in entry["timing_result"] for entry in manifest["entries"])
    assert all("/s00" in entry["traces"][0] for entry in manifest["entries"])


def _bridge_config(tmp_path: Path) -> MegatronBridgeConfig:
    return MegatronBridgeConfig(
        bridge_dir=tmp_path / "Megatron-Bridge",
        output_dir=tmp_path,
        experiment_name="fixture",
        model=_model(),
        hardware=HardwareDescriptor(
            gpu_name="H100",
            gpu_count=1,
            gpu_memory_mib=81920,
        ),
        warmup_steps=0,
    )


def _candidate(repeat: int = 0) -> TrainingCandidate:
    return TrainingCandidate(
        parallelism=Parallelism(tp=1, pp=2, dp=1, cp=1),
        micro_batch_size=1,
        global_batch_size=4,
        num_microbatches=4,
        sequence_shape=(512,),
        repeat=repeat,
    )


def _prepare_bridge_fixture(
    tmp_path: Path,
    *,
    repeat: int,
    success: bool,
) -> tuple[MegatronBridgeConfig, TrainingCandidate, Path]:
    config = _bridge_config(tmp_path)
    candidate = _candidate(repeat)
    _, run_dir = prepare_bridge_candidate(config, candidate)
    fixture_dir = FIXTURES / "megatron_bridge"
    shutil.copy(fixture_dir / "ConfigContainer.yaml", run_dir / "ConfigContainer.yaml")
    if success:
        shutil.copy(fixture_dir / "status_success.json", run_dir / "status.json")
        shutil.copy(fixture_dir / "dryrun_metrics.json", run_dir / "dryrun_metrics.json")
    else:
        shutil.copy(fixture_dir / "status_failed.json", run_dir / "status.json")
        shutil.copy(fixture_dir / "oom_output.txt", run_dir / "run.log")
    return config, candidate, run_dir


def test_megatron_bridge_timer_memory_and_resolved_config(tmp_path: Path):
    config, candidate, run_dir = _prepare_bridge_fixture(
        tmp_path,
        repeat=0,
        success=True,
    )
    record = collect_bridge_record(config, candidate, run_dir=run_dir)
    assert record.run.status == "success"
    assert isinstance(record.payload, TrainingPayload)
    assert record.payload.step_times_s == (0.101, 0.099, 0.1)
    assert record.payload.peak_allocated_bytes == 1_050_000_000
    assert record.payload.peak_reserved_bytes == 1_150_000_000
    assert record.payload.timers_s["optimizer"] == (0.01, 0.01, 0.01)
    assert record.tags["resolved_config"].endswith("ConfigContainer.yaml")


def test_megatron_bridge_oom_is_preserved(tmp_path: Path):
    config, candidate, run_dir = _prepare_bridge_fixture(
        tmp_path,
        repeat=1,
        success=False,
    )
    record = collect_bridge_record(config, candidate, run_dir=run_dir)
    assert record.run.status == "oom"
    assert isinstance(record.payload, TrainingPayload)
    assert record.payload.step_times_s == ()
    assert record.run.error == "CUDA out of memory."


def test_candidate_generator_enforces_parallel_constraints():
    candidates = enumerate_training_candidates(
        world_sizes=[4],
        tensor_parallel_sizes=[1, 3, 4],
        pipeline_parallel_sizes=[1, 5],
        micro_batch_sizes=[1],
        num_microbatches=[1, 2, 4],
        sequence_shapes=[[512]],
        constraints=ModelConstraints(
            num_layers=36,
            num_attention_heads=32,
            num_query_groups=8,
        ),
    )
    assert candidates
    assert {candidate.parallelism.tp for candidate in candidates} == {1, 4}
    assert {candidate.parallelism.pp for candidate in candidates} == {1}
    assert all(
        candidate.global_batch_size
        == candidate.micro_batch_size
        * candidate.parallelism.dp
        * candidate.num_microbatches
        for candidate in candidates
    )


def test_profile_schema_rejects_inconsistent_world_size():
    with pytest.raises(ProfileValidationError, match="world_size"):
        Parallelism(tp=2, pp=2, dp=1, cp=1, world_size=2)


def test_prefix_console_bytes_labels_newlines_and_carriage_returns():
    first, at_start = _prefix_console_bytes(
        b"load 10%\rload 20%\nready\n",
        b"[s000] ",
        at_line_start=True,
    )
    assert first == b"[s000] load 10%\r[s000] load 20%\n[s000] ready\n"
    assert at_start is True
    second, at_start = _prefix_console_bytes(
        b"partial",
        b"[s000] ",
        at_line_start=True,
    )
    assert second == b"[s000] partial"
    assert at_start is False


def test_sweep_runtime_environment_upgrades_warn_and_unbuffers():
    environment = _sweep_runtime_environment(
        {"VLLM_LOGGING_LEVEL": "WARN", "CUDA_VISIBLE_DEVICES": "0"}
    )
    assert environment["VLLM_LOGGING_LEVEL"] == "INFO"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_sweep_runtime_environment_preserves_debug():
    environment = _sweep_runtime_environment({"VLLM_LOGGING_LEVEL": "DEBUG"})
    assert environment["VLLM_LOGGING_LEVEL"] == "DEBUG"


def _fake_sweep_script(tmp_path: Path) -> Path:
    """A stand-in sweep that records its startup window and its ready marker."""
    path = tmp_path / "fake_sweep.py"
    path.write_text(
        "\n".join(
            [
                "import pathlib, sys, time",
                "events, label, exit_code = sys.argv[1], sys.argv[2], int(sys.argv[3])",
                "log = pathlib.Path(events)",
                "log.open('a').write(f'enter {label}\\n')",
                # Signs of life long before the server is ready, as the real sweep does.
                "print('[BEGIN SERVER]', flush=True)",
                "time.sleep(0.4)",
                "log.open('a').write(f'ready {label}\\n')",
                "print('[BEGIN BENCHMARK]', flush=True)",
                "time.sleep(0.2)",
                "sys.exit(exit_code)",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _fake_jobs(
    tmp_path: Path,
    script: Path,
    events: Path,
    *,
    count: int,
    failing: frozenset[str] = frozenset(),
) -> list[VLLMSweepJob]:
    return [
        VLLMSweepJob(
            phase="prefill",
            measurement="timing",
            shard_id=f"s{index:03d}",
            wave=0,
            gpu_ids=(str(index),),
            port=8000 + index,
            command=(
                sys.executable,
                str(script),
                str(events),
                f"s{index:03d}",
                "1" if f"s{index:03d}" in failing else "0",
            ),
            experiment_dir=tmp_path / "raw" / f"s{index:03d}",
            log_path=tmp_path / "control" / f"s{index:03d}.log",
            trace_dir=None,
            world_size=1,
            expected_runs=1,
        )
        for index in range(count)
    ]


def _max_concurrent_startups(events: Path) -> int:
    active = 0
    peak = 0
    for line in events.read_text(encoding="utf-8").splitlines():
        if line.startswith("enter "):
            active += 1
            peak = max(peak, active)
        elif line.startswith("ready "):
            active -= 1
    return peak


def test_vllm_pool_admits_one_startup_at_a_time(tmp_path: Path):
    events = tmp_path / "events.log"
    events.touch()
    config = VLLMSweepConfig(
        model=_model(),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="test",
        phase_parameters={"prefill": ({"random-output-len": 1},)},
    )
    jobs = _fake_jobs(tmp_path, _fake_sweep_script(tmp_path), events, count=4)

    _run_vllm_job_pool(config, jobs, resume=False)

    assert _max_concurrent_startups(events) == 1
    assert len(events.read_text(encoding="utf-8").splitlines()) == 8


def test_vllm_pool_runs_every_shard_before_reporting_failures(tmp_path: Path):
    events = tmp_path / "events.log"
    events.touch()
    config = VLLMSweepConfig(
        model=_model(),
        hardware=HardwareDescriptor(gpu_name="H100", gpu_count=1),
        output_dir=tmp_path,
        experiment_name="test",
        phase_parameters={"prefill": ({"random-output-len": 1},)},
        max_concurrent_startups=2,
    )
    jobs = _fake_jobs(
        tmp_path,
        _fake_sweep_script(tmp_path),
        events,
        count=4,
        failing=frozenset({"s001"}),
    )

    with pytest.raises(RuntimeError) as failure:
        _run_vllm_job_pool(config, jobs, resume=False)

    assert "prefill/timing/s001" in str(failure.value)
    assert "1 of 4" in str(failure.value)
    assert isinstance(failure.value.__cause__, subprocess.CalledProcessError)
    started = {
        line.removeprefix("enter ")
        for line in events.read_text(encoding="utf-8").splitlines()
        if line.startswith("enter ")
    }
    assert started == {f"s{index:03d}" for index in range(4)}


def test_parse_sweep_log_binds_namespace_and_skips_warmup(tmp_path: Path):
    result = tmp_path / "SERVE--tp=1" / "run=0.json"
    result.parent.mkdir()
    result.write_text("{}", encoding="utf-8")
    log_path = tmp_path / "timing_sweep.log"
    log_path.write_text(
        "\n".join(
            [
                "Starting initial single prompt test run...",
                "Iteration(9): 1 context requests, 8 context tokens, "
                "0 generation requests, 0 generation tokens, "
                "iteration elapsed time: 99.00 ms, GPU KV cache usage: 1.0%",
                f"Namespace(result_dir='{result.parent}', result_filename='run=0.json')",
                "Starting initial single prompt test run...",
                "Iteration(0): 1 context requests, 8 context tokens, "
                "0 generation requests, 0 generation tokens, "
                "iteration elapsed time: 1.00 ms, GPU KV cache usage: 1.0%",
                "Starting main benchmark run...",
                "Starting profiler...",
                "Traffic request rate: inf",
                "(EngineCore pid=1) INFO 08-17 12:00:00 [loggers.py:182] "
                "Engine 000: Iteration(1): 2 context requests, 256 context tokens, "
                "0 generation requests, 0 generation tokens, "
                "iteration elapsed time: 10.50 ms, GPU KV cache usage: 1.0%",
                "Stopping profiler...",
                "============ Serving Benchmark Result ============",
            ]
        ),
        encoding="utf-8",
    )
    observations = parse_sweep_log(log_path)
    details = observations[result.resolve()]
    assert len(details) == 1
    assert details[0].index == 1
    assert details[0].ctx_tokens == 256
    assert details[0].elapsed_ms == 10.5


def test_parse_sweep_log_extracts_concatenated_output_file(tmp_path: Path):
    result = tmp_path / "run=0.json"
    result.write_text("{}", encoding="utf-8")
    log_path = tmp_path / "timing_sweep.log"
    log_path.write_text(
        "\n".join(
            [
                f"Output file: {result}(APIServer pid=1) INFO: POST /tokenize 200 OK",
                "Traffic request rate: inf",
                "Iteration(0): 2 context requests, 256 context tokens, "
                "0 generation requests, 0 generation tokens, "
                "iteration elapsed time: 4.00 ms, GPU KV cache usage: 1.0%",
                "[END BENCHMARK]",
            ]
        ),
        encoding="utf-8",
    )
    observations = parse_sweep_log(log_path)
    details = observations[result.resolve()]
    assert len(details) == 1
    assert details[0].elapsed_ms == 4.0
