"""Parsing of vLLM scheduler iteration logs and PyTorch profiler traces."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ...schema import ProfileValidationError, RolloutPayload

_TRACE_EVENT_PATTERN = re.compile(
    r"execute_(?P<total>\d+)_context_(?P<ctx_requests>\d+)"
    r"\(sq(?P<ctx_sq>\d+)sk(?P<ctx_sk>\d+)sqsq(?P<ctx_sqsq>\d+)sqsk(?P<ctx_sqsk>\d+)\)"
    r"_generation_(?P<gen_requests>\d+)"
    r"\(sq(?P<gen_sq>\d+)sk(?P<gen_sk>\d+)sqsq(?P<gen_sqsq>\d+)sqsk(?P<gen_sqsk>\d+)\)"
)
_TRACE_RANK_PATTERN = re.compile(r"(?:^|_)rank(?P<rank>\d+)(?:[._]|$)")
# Official sweep prints `Output file:` before the bench child starts. When stdout is a
# pipe those prints can flush late and concatenate with APIServer lines, so require the
# `run=N.json` suffix instead of consuming the rest of the line.
_OUTPUT_FILE_PATTERN = re.compile(r"Output file:\s*(?P<path>.+?/run=\d+\.json)")
_RESULT_DIR_PATTERN = re.compile(r"result_dir=['\"](?P<path>[^'\"]+)['\"]")
_RESULT_FILENAME_PATTERN = re.compile(r"result_filename=['\"](?P<name>[^'\"]+)['\"]")
_ITERATION_PATTERN = re.compile(
    r"Iteration\((?P<index>\d+)\): "
    r"(?P<ctx_requests>\d+) context requests, "
    r"(?P<ctx_tokens>\d+) context tokens, "
    r"(?P<gen_requests>\d+) generation requests, "
    r"(?P<gen_tokens>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed_ms>[0-9.]+) ms"
)
_MEASUREMENT_START_MARKERS = (
    "Starting main benchmark run...",
    "Traffic request rate:",
    "Traffic ramp-up strategy:",
    "Starting profiler...",
    "Profiler started",
)
_MEASUREMENT_STOP_MARKERS = (
    "Stopping profiler",
    "Profiler stopped",
    "[END BENCHMARK]",
    "============ Serving Benchmark Result",
)


@dataclass(frozen=True)
class _IterationDetail:
    index: int
    ctx_requests: int
    ctx_tokens: int
    gen_requests: int
    gen_tokens: int
    elapsed_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "index": self.index,
            "ctx_requests": self.ctx_requests,
            "ctx_tokens": self.ctx_tokens,
            "gen_requests": self.gen_requests,
            "gen_tokens": self.gen_tokens,
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> _IterationDetail:
        return cls(
            index=int(value["index"]),
            ctx_requests=int(value["ctx_requests"]),
            ctx_tokens=int(value["ctx_tokens"]),
            gen_requests=int(value["gen_requests"]),
            gen_tokens=int(value["gen_tokens"]),
            elapsed_ms=float(value["elapsed_ms"]),
        )


@dataclass(frozen=True)
class _TraceStep:
    timestamp_us: float
    signature: tuple[int, ...]
    payload: RolloutPayload
    metrics: dict[str, int]


def _result_path_from_line(line: str) -> Path | None:
    output_match = _OUTPUT_FILE_PATTERN.search(line)
    if output_match is not None:
        return Path(output_match.group("path")).resolve()
    directory_match = _RESULT_DIR_PATTERN.search(line)
    filename_match = _RESULT_FILENAME_PATTERN.search(line)
    if directory_match is None or filename_match is None:
        return None
    return (Path(directory_match.group("path")) / filename_match.group("name")).resolve()


def parse_sweep_log(path: Path) -> dict[Path, list[_IterationDetail]]:
    """Parse a sweep log into per-result-file scheduler iteration timings."""
    if not path.exists():
        raise ProfileValidationError(
            f"Missing vLLM sweep log {path}; scheduler iteration timing is required."
        )
    observations: dict[Path, list[_IterationDetail]] = {}
    current_result: Path | None = None
    capturing = False
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            for line in raw_line.replace("\r", "\n").split("\n"):
                if not line:
                    continue
                result_path = _result_path_from_line(line)
                if result_path is not None:
                    current_result = result_path
                    observations.setdefault(current_result, [])
                    capturing = False
                    continue
                if any(marker in line for marker in _MEASUREMENT_START_MARKERS):
                    capturing = True
                if any(marker in line for marker in _MEASUREMENT_STOP_MARKERS):
                    capturing = False
                    continue
                iteration_match = _ITERATION_PATTERN.search(line)
                if (
                    capturing
                    and current_result is not None
                    and iteration_match is not None
                ):
                    values = iteration_match.groupdict()
                    observations[current_result].append(
                        _IterationDetail(
                            index=int(values["index"]),
                            ctx_requests=int(values["ctx_requests"]),
                            ctx_tokens=int(values["ctx_tokens"]),
                            gen_requests=int(values["gen_requests"]),
                            gen_tokens=int(values["gen_tokens"]),
                            elapsed_ms=float(values["elapsed_ms"]),
                        )
                    )
    return observations


def _open_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ProfileValidationError(f"Expected a trace object in {path}.")
    return value


def parse_trace(path: Path) -> list[_TraceStep]:
    """Extract per-step attention feature annotations from one detailed trace file."""
    value = _open_trace(path)
    steps: list[_TraceStep] = []
    for event in value.get("traceEvents", []):
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        match = _TRACE_EVENT_PATTERN.fullmatch(str(event.get("name", "")))
        if match is None:
            continue
        metrics = {key: int(item) for key, item in match.groupdict().items()}
        if metrics["total"] != metrics["ctx_sq"] + metrics["gen_sq"]:
            raise ProfileValidationError(
                f"Trace event token total is inconsistent in {path}: {event['name']!r}."
            )
        request_count = metrics["ctx_requests"] + metrics["gen_requests"]
        if metrics["total"] <= 0 or request_count <= 0:
            continue
        has_context = metrics["ctx_requests"] > 0
        has_generation = metrics["gen_requests"] > 0
        phase = "mixed" if has_context and has_generation else "prefill" if has_context else "decode"
        duration_s = float(event.get("dur", 0.0)) / 1_000_000.0
        payload = RolloutPayload(
            phase=phase,
            request_count=request_count,
            input_tokens=metrics["ctx_sq"],
            output_tokens=metrics["gen_sq"],
            n_tokens=metrics["total"],
            sum_l=float(metrics["ctx_sk"] + metrics["gen_sk"]),
            sum_l2=float(metrics["ctx_sqsk"]),
            ctxsum=float(metrics["gen_sqsk"]),
            latency_s=duration_s,
        )
        signature = tuple(metrics[key] for key in sorted(metrics))
        steps.append(
            _TraceStep(
                timestamp_us=float(event.get("ts", 0.0)),
                signature=signature,
                payload=payload,
                metrics=metrics,
            )
        )
    return sorted(steps, key=lambda item: item.timestamp_us)


def aggregate_rank_traces(paths: list[Path]) -> list[_TraceStep]:
    """Merge per-rank traces of one distributed run into a single step timeline."""
    per_rank = [parse_trace(path) for path in paths]
    if not per_rank or not per_rank[0]:
        raise ProfileValidationError(
            f"No detailed vLLM execute annotations were found in {[str(path) for path in paths]}."
        )
    reference = per_rank[0]
    for steps in per_rank[1:]:
        if [step.signature for step in steps] != [step.signature for step in reference]:
            raise ProfileValidationError(
                "Detailed vLLM traces disagree across ranks; refusing to align steps by client timing."
            )
    aggregated: list[_TraceStep] = []
    for index, step in enumerate(reference):
        latency_s = max(rank_steps[index].payload.latency_s for rank_steps in per_rank)
        aggregated.append(
            replace(
                step,
                payload=replace(
                    step.payload,
                    latency_s=latency_s,
                    batch_step_index=index,
                ),
            )
        )
    return aggregated


def apply_iteration_timings(
    steps: list[_TraceStep],
    details: list[_IterationDetail],
    *,
    iteration_tag: str,
) -> list[_TraceStep]:
    """Overlay scheduler iteration latencies and indices onto trace steps."""
    if len(steps) != len(details):
        raise ProfileValidationError(
            "vLLM trace and scheduler log contain different step counts: "
            f"{len(steps)} trace annotations versus {len(details)} iteration timings."
        )
    timed: list[_TraceStep] = []
    for step_index, (step, detail) in enumerate(
        zip(steps, details, strict=True)
    ):
        metrics = step.metrics
        observed = (
            metrics["ctx_requests"],
            metrics["ctx_sq"],
            metrics["gen_requests"],
            metrics["gen_sq"],
        )
        logged = (
            detail.ctx_requests,
            detail.ctx_tokens,
            detail.gen_requests,
            detail.gen_tokens,
        )
        if observed != logged:
            raise ProfileValidationError(
                "vLLM trace annotation and scheduler iteration log disagree at "
                f"step {step_index}: trace={observed}, scheduler={logged}."
            )
        timed.append(
            replace(
                step,
                payload=replace(
                    step.payload,
                    latency_s=detail.elapsed_ms / 1000.0,
                    batch_step_index=step_index,
                ),
                metrics={**metrics, iteration_tag: detail.index},
            )
        )
    return timed


def worker_trace_files(trace_dir: Path) -> list[Path]:
    """List a shard's worker trace files, ranked by rank when annotated."""
    candidates = sorted(
        (
            path
            for path in trace_dir.rglob("*")
            if path.is_file()
            and (
                path.name.endswith(".pt.trace.json")
                or path.name.endswith(".pt.trace.json.gz")
            )
        ),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    ranked = [path for path in candidates if _TRACE_RANK_PATTERN.search(path.name)]
    return ranked or candidates
