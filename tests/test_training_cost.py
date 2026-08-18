"""Synthetic Hydraulis V1 latency, memory, OOD, and simulator tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from dryrun.cost.artifact import CostArtifact
from dryrun.cost.rollout import UnifiedRoofline
from dryrun.cost.training import (
    AnalyticalTrainingCost,
    HydraulisTrainingCost,
    LinearTrainingCost,
    TrainingOutOfDomainError,
    TrainingParallelism,
    TrainingWorkload,
    parameter_inventory,
)
from dryrun.telemetry.store import JsonlStore
from dryrun.policy.psrl import PSRLPolicy
from dryrun.profiler.schema import (
    BYTES_PER_MIB,
    HardwareDescriptor,
    ModelDescriptor,
    Parallelism,
    ProfileRecord,
    RankMemory,
    RunDescriptor,
    SequenceStats,
    SoftwareDescriptor,
    TrainingPayload,
)
from dryrun.simulator import SimConfig, Simulator
from dryrun.workload.distributions import uniform

PARAMETER_COUNT = 10_000_000
LATENCY_COEFFICIENTS = (2e-10, 1e-7, 0.002, 0.01)
ACTIVATION_SLOPE = 2048.0
MEMORY_RESIDUAL = 5_000_000.0


def _training_records() -> tuple[list[ProfileRecord], ProfileRecord]:
    records: list[ProfileRecord] = []
    model = ModelDescriptor(
        name="synthetic-qwen",
        parameter_count=PARAMETER_COUNT,
        num_layers=8,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=4,
    )
    software = SoftwareDescriptor(
        backend="megatron_bridge",
        backend_version="0.4.2",
    )
    for dp in (1, 2):
        parallelism = Parallelism(tp=1, pp=2, dp=dp, cp=1)
        hardware = HardwareDescriptor(
            gpu_name="H100",
            gpu_count=int(parallelism.world_size),
            gpu_memory_mib=81920,
        )
        for micro_batch_size in (1, 2):
            for num_microbatches in (2, 4, 8):
                for sequence_length in (256, 512, 1024):
                    microbatches = tuple(
                        SequenceStats.from_lengths(
                            [sequence_length] * micro_batch_size
                        )
                        for _ in range(num_microbatches)
                    )
                    q_values = [item.sum_squared_tokens for item in microbatches]
                    t_values = [item.sum_tokens for item in microbatches]
                    q_effective = sum(q_values) + max(q_values)
                    t_effective = sum(t_values) + max(t_values)
                    pipeline_slots = num_microbatches + 1
                    a_q, b_t, c_pipeline, update = LATENCY_COEFFICIENTS
                    latency = (
                        a_q * q_effective
                        + b_t * t_effective
                        + c_pipeline * pipeline_slots
                        + update
                    )
                    training_parallelism = TrainingParallelism(
                        tp=1,
                        pp=2,
                        dp=dp,
                        cp=1,
                    )
                    static = parameter_inventory(
                        parameter_count=PARAMETER_COUNT,
                        parallelism=training_parallelism,
                        parameter_dtype="bf16",
                        gradient_dtype="fp32",
                        optimizer_state_dtype="fp32",
                        distributed_optimizer=True,
                    ).total_bytes
                    activation_feature = max(t_values) * min(num_microbatches, 2)
                    peak = int(
                        static
                        + ACTIVATION_SLOPE * activation_feature
                        + MEMORY_RESIDUAL
                    )
                    payload = TrainingPayload(
                        micro_batch_size=micro_batch_size,
                        global_batch_size=micro_batch_size * dp * num_microbatches,
                        num_microbatches=num_microbatches,
                        schedule="1f1b",
                        recompute="none",
                        optimizer="adam",
                        microbatches=microbatches,
                        step_times_s=(latency, latency * 1.001),
                        rank_memory=(
                            RankMemory(
                                rank=0,
                                allocated_bytes=peak - 1024,
                                reserved_bytes=peak,
                            ),
                        ),
                        peak_allocated_bytes=peak - 1024,
                        peak_reserved_bytes=peak,
                        parameter_count=PARAMETER_COUNT,
                    )
                    run_id = (
                        f"dp{dp}-mbs{micro_batch_size}-nmb{num_microbatches}"
                        f"-seq{sequence_length}"
                    )
                    records.append(
                        ProfileRecord(
                            record_type="training",
                            run=RunDescriptor(run_id=run_id),
                            model=model,
                            hardware=hardware,
                            software=software,
                            parallelism=parallelism,
                            payload=payload,
                        )
                    )

    oom_parallelism = Parallelism(tp=1, pp=2, dp=1, cp=1)
    oom_microbatches = tuple(
        SequenceStats.from_lengths([1024])
        for _ in range(2)
    )
    oom_static = parameter_inventory(
        parameter_count=PARAMETER_COUNT,
        parallelism=TrainingParallelism(tp=1, pp=2, dp=1, cp=1),
        parameter_dtype="bf16",
        gradient_dtype="fp32",
        optimizer_state_dtype="fp32",
        distributed_optimizer=True,
    ).total_bytes
    oom_activation = 1024 * 2
    unconstrained_peak = int(
        oom_static + ACTIVATION_SLOPE * oom_activation + MEMORY_RESIDUAL
    )
    oom_hardware = HardwareDescriptor(
        gpu_name="H100",
        gpu_count=int(oom_parallelism.world_size),
        gpu_memory_mib=(unconstrained_peak + 2_000_000 + BYTES_PER_MIB - 1) // BYTES_PER_MIB,
    )
    oom_payload = TrainingPayload(
        micro_batch_size=1,
        global_batch_size=2,
        num_microbatches=2,
        schedule="1f1b",
        recompute="none",
        optimizer="adam",
        microbatches=oom_microbatches,
        parameter_count=PARAMETER_COUNT,
    )
    oom_record = ProfileRecord(
        record_type="training",
        run=RunDescriptor(run_id="oom", status="oom", error="CUDA out of memory."),
        model=model,
        hardware=oom_hardware,
        software=software,
        parallelism=oom_parallelism,
        payload=oom_payload,
    )
    records.append(oom_record)
    return records, oom_record


def test_hydraulis_fit_roundtrip_and_known_latency(tmp_path: Path):
    records, _ = _training_records()
    artifact = HydraulisTrainingCost.fit(records)
    path = artifact.save(tmp_path / "hydraulis.json")
    loaded_artifact = CostArtifact.load(path)
    model = HydraulisTrainingCost.from_artifact(loaded_artifact)
    workload = TrainingWorkload(
        sequence_lengths=(512,) * 8,
        micro_batch_size=2,
    )
    parallelism = TrainingParallelism(tp=1, pp=2, dp=1, cp=1)
    estimate = model.estimate(workload, parallelism)

    q_effective = 5 * (2 * 512**2)
    t_effective = 5 * (2 * 512)
    expected = (
        LATENCY_COEFFICIENTS[0] * q_effective
        + LATENCY_COEFFICIENTS[1] * t_effective
        + LATENCY_COEFFICIENTS[2] * 5
        + LATENCY_COEFFICIENTS[3]
    )
    assert estimate.latency_s == pytest.approx(expected * 1.0005, rel=2e-3)
    assert estimate.peak_memory_bytes > 0
    assert {entry.selector["component"] for entry in artifact.entries} == {
        "latency",
        "memory",
    }
    assert all(entry.train_metrics.design_rank in {2, 4} for entry in artifact.entries)


def test_hydraulis_uses_oom_as_memory_inequality():
    records, oom_record = _training_records()
    model = HydraulisTrainingCost.from_artifact(
        HydraulisTrainingCost.fit(records)
    )
    workload = TrainingWorkload(
        sequence_lengths=(1024, 1024),
        micro_batch_size=1,
    )
    predicted = model.memory_model.predict(
        workload,
        TrainingParallelism(tp=1, pp=2, dp=1, cp=1),
    )
    assert oom_record.hardware.gpu_memory_bytes is not None
    assert predicted > oom_record.hardware.gpu_memory_bytes


def test_hydraulis_dp_optimizer_sharding_reduces_memory():
    records, _ = _training_records()
    model = HydraulisTrainingCost.from_artifact(
        HydraulisTrainingCost.fit(records)
    )
    workload = TrainingWorkload(
        sequence_lengths=(512,) * 8,
        micro_batch_size=1,
    )
    dp1 = model.memory_model.predict(
        workload,
        TrainingParallelism(tp=1, pp=2, dp=1, cp=1),
    )
    dp2 = model.memory_model.predict(
        workload,
        TrainingParallelism(tp=1, pp=2, dp=2, cp=1),
    )
    assert dp2 < dp1


def test_hydraulis_rejects_numerical_extrapolation():
    records, _ = _training_records()
    model = HydraulisTrainingCost.from_artifact(
        HydraulisTrainingCost.fit(records)
    )
    workload = TrainingWorkload(
        sequence_lengths=(8192,) * 8,
        micro_batch_size=2,
    )
    with pytest.raises(TrainingOutOfDomainError, match="outside the profiled range"):
        model.estimate(
            workload,
            TrainingParallelism(tp=1, pp=2, dp=1, cp=1),
        )


def test_hydraulis_rejects_rank_deficient_design():
    records, _ = _training_records()
    restricted = [
        record
        for record in records
        if isinstance(record.payload, TrainingPayload)
        and record.run.status == "success"
        and record.parallelism.dp == 1
        and record.payload.micro_batch_size == 1
        and record.payload.num_microbatches == 2
    ]
    with pytest.raises(ValueError, match="design matrix is rank"):
        HydraulisTrainingCost.fit(restricted)


def test_fitted_training_cost_integrates_with_simulator(tmp_path: Path):
    records, _ = _training_records()
    training_cost = HydraulisTrainingCost.from_artifact(
        HydraulisTrainingCost.fit(records)
    )
    store = JsonlStore(tmp_path / "logs")
    simulator = Simulator(
        rollout_cost=UnifiedRoofline(
            F=0.005,
            W=0.001,
            G=0.0001,
            A_p=0,
            A_d=1e-7,
        ),
        training_cost=training_cost,
        policy=PSRLPolicy(),
        lengths=uniform(32, 50, 100, seed=1),
        cfg=SimConfig(
            batch_size=8,
            n_versions=1,
            max_staleness=2,
            training_pp=2,
            training_micro_batch_size=2,
        ),
        store=store,
    )
    result = simulator.run()
    rows = list(store.iter("train"))
    assert result.versions_done == 1
    assert result.train_peak_memory_bytes[0] > 0
    assert rows[0]["peak_memory_bytes"] == result.train_peak_memory_bytes[0]
    assert rows[0]["training_pp"] == 2


def test_linear_latency_scales_with_tokens_and_memory_with_gpus():
    model = LinearTrainingCost(time_per_token=1e-4, base_memory_bytes=80_000_000_000)
    short = TrainingWorkload(sequence_lengths=(256,) * 4, micro_batch_size=1)
    long = TrainingWorkload(sequence_lengths=(512,) * 4, micro_batch_size=1)
    one_gpu = TrainingParallelism()
    four_gpus = TrainingParallelism(tp=2, pp=1, dp=2, cp=1)

    short_estimate = model.estimate(short, one_gpu)
    long_estimate = model.estimate(long, one_gpu)
    sharded = model.estimate(long, four_gpus)

    assert short_estimate.latency_s == pytest.approx(0.1024)
    assert long_estimate.latency_s == pytest.approx(2 * short_estimate.latency_s)
    assert long_estimate.peak_memory_bytes == 80_000_000_000
    assert sharded.peak_memory_bytes == 20_000_000_000
    assert sharded.latency_s == long_estimate.latency_s


def test_analytical_roofline_uses_compute_and_shards_memory():
    model = AnalyticalTrainingCost(
        model_params=8_000_000_000,
        peak_flops=312e12,
        memory_bandwidth=2e12,
        overhead_factor=1.0,
        bytes_per_parameter=16.0,
    )
    workload = TrainingWorkload(sequence_lengths=(1000,) * 8, micro_batch_size=1)
    one_gpu = model.estimate(workload, TrainingParallelism())
    two_gpu = model.estimate(workload, TrainingParallelism(tp=2, pp=1, dp=1, cp=1))
    expected_compute = 6 * 8_000_000_000 * 8000 / 312e12
    assert one_gpu.latency_s == pytest.approx(expected_compute)
    assert two_gpu.latency_s == pytest.approx(expected_compute / 2)
    assert two_gpu.peak_memory_bytes == one_gpu.peak_memory_bytes // 2
