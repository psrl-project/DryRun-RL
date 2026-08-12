"""Megatron training step profiling.

Launches Megatron training runs via subprocess, collects step-time
statistics at varying batch sizes and sequence lengths. Outputs a CSV
for training cost model fitting.

Megatron is NOT imported at module level. The profiler launches training
as a subprocess and parses stdout logs.
"""

from __future__ import annotations

import csv
import logging
import re
import shlex
from pathlib import Path

from .runner import ProfileRunner
from .schemas import MegatronProfileConfig

dryrun_logger = logging.getLogger("dryrun")

_STEP_TIME_PATTERN = re.compile(
    r"elapsed time per iteration \(ms\):\s*([\d.]+)"
)


def _parse_step_times(output: str, warmup: int, measure: int) -> list[float]:
    """Extract step times from Megatron stdout."""
    all_times = []
    for match in _STEP_TIME_PATTERN.finditer(output):
        all_times.append(float(match.group(1)))

    if len(all_times) <= warmup:
        dryrun_logger.warning(
            f"Found only {len(all_times)} step times, needed {warmup} warmup + "
            f"{measure} measured."
        )
        return all_times[warmup:] if len(all_times) > warmup else []

    return all_times[warmup:warmup + measure]


def _build_train_cmd(
    base_cmd: str,
    batch_size: int,
    seq_len: int,
    pp: int,
    tp: int,
    dp: int,
    total_steps: int,
    extra_args: list[str],
) -> list[str]:
    """Build the training launch command."""
    cmd = shlex.split(base_cmd)
    cmd.extend([
        "--micro-batch-size", str(batch_size),
        "--seq-length", str(seq_len),
        "--pipeline-model-parallel-size", str(pp),
        "--tensor-model-parallel-size", str(tp),
        "--train-iters", str(total_steps),
    ])
    cmd.extend(extra_args)
    return cmd


def run_megatron_profile(cfg: MegatronProfileConfig, output_path: str | Path) -> Path:
    """
    Run Megatron training profiling and write results to CSV.

    Args:
        cfg: Profiling configuration.
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV.
    """
    assert cfg.launch_cmd, "megatron profiler requires launch_cmd."

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total_steps = cfg.warmup_steps + cfg.measure_steps
    total_sweeps = len(cfg.batch_sizes) * len(cfg.seq_lengths)

    for idx, (bs, seq_len) in enumerate(
        [(b, s) for b in cfg.batch_sizes for s in cfg.seq_lengths]
    ):
        dryrun_logger.info(
            f"Sweep point {idx + 1}/{total_sweeps}: "
            f"batch_size={bs}, seq_len={seq_len}."
        )

        cmd = _build_train_cmd(
            cfg.launch_cmd, bs, seq_len,
            cfg.pp, cfg.tp, cfg.dp,
            total_steps, cfg.extra_args,
        )

        runner = ProfileRunner(cmd, health_url=None, timeout=3600)
        try:
            runner.launch()
            assert runner.proc is not None, "Failed to launch training process."
            runner.proc.wait(timeout=3600)
            output = runner.read_output()
        except Exception as exc:
            dryrun_logger.warning(
                f"Sweep point (bs={bs}, seq={seq_len}) failed: {exc}."
            )
            runner.shutdown()
            continue

        step_times = _parse_step_times(output, cfg.warmup_steps, cfg.measure_steps)
        if not step_times:
            dryrun_logger.warning(
                f"No step times extracted for bs={bs}, seq_len={seq_len}."
            )
            continue

        import numpy as np  # noqa: PLC0415

        times_arr = np.array(step_times)
        rows.append({
            "batch_size": bs,
            "seq_len": seq_len,
            "pp": cfg.pp,
            "tp": cfg.tp,
            "dp": cfg.dp,
            "step_time_mean_ms": float(np.mean(times_arr)),
            "step_time_std_ms": float(np.std(times_arr)),
            "step_time_p50_ms": float(np.median(times_arr)),
            "step_time_p99_ms": float(np.percentile(times_arr, 99)),
            "n_samples": len(step_times),
        })

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "batch_size", "seq_len", "pp", "tp", "dp",
            "step_time_mean_ms", "step_time_std_ms",
            "step_time_p50_ms", "step_time_p99_ms", "n_samples",
        ])
        writer.writeheader()
        writer.writerows(rows)

    dryrun_logger.info(
        f"Megatron profiling complete: {len(rows)} sweep points → {output_path}."
    )
    return output_path
