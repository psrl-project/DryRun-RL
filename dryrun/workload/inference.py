"""Fit output-length distributions by running live inference against a model.

Launches a `vllm serve` process for the given model, sends sampled dataset
prompts to its OpenAI-compatible completions endpoint, and records the
completion token count of each response as an output length. The server is
always torn down when collection finishes, even on error.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .._logging import dryrun_logger
from .fitting import load_prompt_messages, render_prompt
from .trace import write_length_trace

_POLL_INTERVAL_S = 2.0
_TERMINATE_TIMEOUT_S = 30.0


class _VLLMServer:
    """Manages the lifecycle of one `vllm serve` subprocess."""

    def __init__(
        self,
        model_path: str,
        port: int,
        tp: int,
        extra_serve_args: dict[str, Any],
        ready_timeout: int,
        log_path: Path,
    ):
        self.model_path = model_path
        self.port = port
        self.tp = tp
        self.extra_serve_args = extra_serve_args
        self.ready_timeout = ready_timeout
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _command(self) -> list[str]:
        command = [
            "vllm",
            "serve",
            self.model_path,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tp),
        ]
        for key, value in self.extra_serve_args.items():
            option = f"--{str(key).replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    command.append(option)
                continue
            command.extend((option, str(value)))
        return command

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout
        health_url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            assert self.process is not None, "vLLM server process was not started."
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"vLLM server exited early with code {return_code}. See log at {self.log_path}."
                )
            try:
                with urllib.request.urlopen(health_url, timeout=5):  # noqa: S310
                    dryrun_logger.info(f"vLLM server is ready at {self.base_url!r}.")
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(_POLL_INTERVAL_S)
        raise TimeoutError(f"vLLM server did not become ready within {self.ready_timeout}s.")

    def __enter__(self) -> "_VLLMServer":
        command = self._command()
        dryrun_logger.info(f"Launching vLLM server: {' '.join(command)}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        self._log_handle = log_handle
        self._wait_until_ready()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.process is None:
            return
        dryrun_logger.info("Shutting down vLLM server...")
        self.process.terminate()
        try:
            self.process.wait(timeout=_TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            dryrun_logger.warning("vLLM server did not exit in time, killing it.")
            self.process.kill()
            self.process.wait()
        self._log_handle.close()


async def _send_requests(
    base_url: str,
    served_model_name: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
) -> list[int]:
    """Send prompts with bounded concurrency and return completion token counts."""
    import aiohttp  # noqa: PLC0415

    semaphore = asyncio.Semaphore(concurrency)
    url = f"{base_url}/v1/completions"

    async def _one_request(session: "aiohttp.ClientSession", prompt: str) -> int:
        payload = {
            "model": served_model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        async with semaphore, session.post(url, json=payload) as response:
            data = await response.json()
            return int(data["usage"]["completion_tokens"])

    async with aiohttp.ClientSession() as session:
        return list(await asyncio.gather(*(_one_request(session, prompt) for prompt in prompts)))


def collect_output_lengths(
    dataset_path: str | Path,
    tokenizer_path: str,
    model_path: str,
    output_path: str | Path,
    tp: int = 1,
    sample_size: int = 200,
    max_tokens: int = 4096,
    seed: int = 42,
    concurrency: int = 16,
    port: int = 8000,
    server_ready_timeout: int = 900,
    extra_serve_args: dict[str, Any] | None = None,
    log_path: str | Path | None = None,
) -> Path:
    """
    Launch a vLLM server, run sampled dataset prompts through it, and write
    the resulting output-token-count lengths as a trace artifact.

    Args:
        dataset_path (str | Path): Parquet dataset with a `prompt` column of
            `list[{content, role}]` chat messages.
        tokenizer_path (str): HuggingFace tokenizer name or local path, used
            to render the chat template for each sampled prompt.
        model_path (str): Model passed to `vllm serve` and to the
            completions request's `model` field.
        output_path (str | Path): Destination for the JSONL trace artifact.
        tp (int): Tensor-parallel size for the served model.
        sample_size (int): Number of prompts to sample from the dataset.
        max_tokens (int): Maximum output tokens requested per completion.
        seed (int): RNG seed for prompt subsampling.
        concurrency (int): Maximum number of in-flight requests.
        port (int): Local port for the vLLM server.
        server_ready_timeout (int): Seconds to wait for `/health` before giving up.
        extra_serve_args (dict[str, Any] | None): Additional `vllm serve` CLI
            flags, e.g. `{"gpu_memory_utilization": 0.9}`.
        log_path (str | Path | None): Where to write the vLLM server log.
            Defaults to `<output_path>.server.log`.

    Returns:
        Path: Path to the written trace artifact.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    output = Path(output_path)
    server_log = Path(log_path) if log_path is not None else output.with_suffix(".server.log")

    dryrun_logger.info(f"Loading tokenizer from {tokenizer_path!r}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    dryrun_logger.info(f"Reading dataset from {dataset_path!r}...")
    messages_list = load_prompt_messages(dataset_path, sample_size, seed)
    prompts = [render_prompt(tokenizer, messages) for messages in messages_list]

    with _VLLMServer(
        model_path=model_path,
        port=port,
        tp=tp,
        extra_serve_args=extra_serve_args or {},
        ready_timeout=server_ready_timeout,
        log_path=server_log,
    ) as server:
        dryrun_logger.info(
            f"Sending {len(prompts)} requests to {server.base_url!r} with max_tokens={max_tokens}..."
        )
        lengths = asyncio.run(
            _send_requests(server.base_url, model_path, prompts, max_tokens, concurrency)
        )

    written = write_length_trace(
        lengths,
        output,
        metadata={
            "kind": "output_length",
            "dataset": str(dataset_path),
            "tokenizer": str(tokenizer_path),
            "model": str(model_path),
            "tp": tp,
            "max_tokens": max_tokens,
            "seed": seed,
        },
    )
    dryrun_logger.info(f"Wrote {len(lengths)} output lengths to {written}.")
    return written
