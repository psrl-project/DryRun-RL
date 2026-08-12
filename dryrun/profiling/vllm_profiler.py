"""vLLM decode/prefill profiling via server + HTTP requests.

Launches a vLLM OpenAI-compatible server, sends synthetic requests at
controlled concurrency and context lengths, and measures inter-token
decode latency. Outputs a CSV suitable for `fit.py`.

vLLM is NOT imported at module level. The profiler validates that vLLM is
installed before launching, and reports a clear error if not.
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import logging
import time
from pathlib import Path

from .runner import ProfileRunner
from .schemas import SweepPoint, VLLMProfileConfig

dryrun_logger = logging.getLogger("dryrun")


def _check_vllm_available() -> None:
    """Raise ImportError if vLLM is not installed."""
    if importlib.util.find_spec("vllm") is None:
        raise ImportError(
            "vLLM is not installed. Install it with: pip install vllm. "
            "The vLLM profiler requires a working vLLM installation."
        )


def _build_server_cmd(cfg: VLLMProfileConfig) -> list[str]:
    """Build the vLLM server launch command."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg.model,
        "--tensor-parallel-size", str(cfg.tp),
        "--pipeline-parallel-size", str(cfg.pp),
        "--port", str(cfg.port),
        "--disable-log-requests",
    ]
    cmd.extend(cfg.extra_args)
    return cmd


async def _send_requests(
    n_concurrent: int,
    prompt_len: int,
    output_len: int,
    model: str,
    port: int,
    warmup: int,
    measure: int,
) -> list[SweepPoint]:
    """
    Send concurrent requests and measure inter-token latency.

    Returns a list of `SweepPoint` measurements.
    """
    try:
        import aiohttp  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "aiohttp is required for vLLM profiling. Install with: pip install aiohttp"
        )

    url = f"http://localhost:{port}/v1/completions"
    prompt = "x " * prompt_len
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
    }

    async def _single_request(session: aiohttp.ClientSession) -> dict:
        """Send one streaming request and measure inter-token latency."""
        token_times = []
        async with session.post(url, json=payload) as resp:
            assert resp.status == 200, f"Request failed with status {resp.status}."
            async for line in resp.content:
                text = line.decode("utf-8").strip()
                if not text.startswith("data: "):
                    continue
                text = text[6:]
                if text == "[DONE]":
                    break
                token_times.append(time.monotonic())

        if len(token_times) < 2:
            return {"n_tokens": 0, "avg_itl": 0.0}

        itls = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
        return {
            "n_tokens": len(token_times),
            "avg_itl": sum(itls) / len(itls),
        }

    results = []
    total_rounds = warmup + measure

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        for round_idx in range(total_rounds):
            tasks = [_single_request(session) for _ in range(n_concurrent)]
            round_results = await asyncio.gather(*tasks, return_exceptions=True)

            if round_idx < warmup:
                continue

            valid = [r for r in round_results if isinstance(r, dict) and r["avg_itl"] > 0]
            if not valid:
                continue

            avg_itl = sum(r["avg_itl"] for r in valid) / len(valid)
            total_tokens = sum(r["n_tokens"] for r in valid)
            token_num = n_concurrent * (prompt_len + output_len // 2)

            results.append(SweepPoint(
                request_num=n_concurrent,
                token_num=token_num,
                throughput=n_concurrent / avg_itl if avg_itl > 0 else 0.0,
                latency_ms=avg_itl * 1000,
                context_length=prompt_len,
            ))

    return results


def run_vllm_profile(cfg: VLLMProfileConfig, output_path: str | Path) -> Path:
    """
    Run a full vLLM profiling sweep and write results to CSV.

    Args:
        cfg: Profiling configuration.
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV.
    """
    _check_vllm_available()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_server_cmd(cfg)
    health_url = f"http://localhost:{cfg.port}/health"

    all_points: list[SweepPoint] = []

    with ProfileRunner(cmd, health_url=health_url, timeout=cfg.timeout) as runner:
        runner.wait_healthy()

        total = len(cfg.batch_sizes) * len(cfg.context_lengths)
        for idx, (bs, ctx) in enumerate(
            [(b, c) for b in cfg.batch_sizes for c in cfg.context_lengths]
        ):
            dryrun_logger.info(
                f"Sweep point {idx + 1}/{total}: "
                f"batch_size={bs}, context_length={ctx}."
            )

            points = asyncio.run(_send_requests(
                n_concurrent=bs,
                prompt_len=ctx,
                output_len=cfg.output_len,
                model=cfg.model,
                port=cfg.port,
                warmup=cfg.warmup_requests,
                measure=cfg.measure_requests,
            ))
            all_points.extend(points)

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["request_num", "token_num", "throughput", "latency_ms", "context_length"])
        for p in all_points:
            writer.writerow([p.request_num, p.token_num, p.throughput, p.latency_ms, p.context_length])

    dryrun_logger.info(f"vLLM profiling complete: {len(all_points)} data points → {output_path}.")
    return output_path
