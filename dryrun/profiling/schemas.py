"""Data structures for profiling and cost model fitting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SweepPoint:
    """
    One measurement point from a profiling sweep.

    Captures the latency/throughput of the inference engine at a specific
    batch size and context length combination.
    """

    request_num: int
    token_num: int
    throughput: float
    latency_ms: float
    context_length: int = 0


@dataclass
class FitQuality:
    """Quality metrics for a cost model fit."""

    other_r_squared: float = 0.0
    attn_r_squared: float = 0.0
    mre_percent: float = 0.0
    rmse: float = 0.0
    n_samples: int = 0
    n_outliers_removed: int = 0


@dataclass
class FitResult:
    """
    Result of two-stage regression fitting.

    The five core coefficients match the `PSRLFitted` constructor and the
    JSON format consumed by `PSRLFitted.from_json()`.
    """

    attn_latency_b: float = 0.0
    attn_latency_k: float = 0.0
    other_threshold: float = 0.0
    other_latency_b: float = 0.0
    other_latency_k: float = 0.0
    fit_quality: FitQuality = field(default_factory=FitQuality)

    def to_dict(self) -> dict:
        """Convert to the JSON-serialisable dict format."""
        d = {
            "attn_latency_b": self.attn_latency_b,
            "attn_latency_k": self.attn_latency_k,
            "other_threshold": self.other_threshold,
            "other_latency_b": self.other_latency_b,
            "other_latency_k": self.other_latency_k,
            "fit_quality": {
                "other_r_squared": self.fit_quality.other_r_squared,
                "attn_r_squared": self.fit_quality.attn_r_squared,
                "mre_percent": self.fit_quality.mre_percent,
                "rmse": self.fit_quality.rmse,
                "n_samples": self.fit_quality.n_samples,
                "n_outliers_removed": self.fit_quality.n_outliers_removed,
            },
        }
        return d


@dataclass
class VLLMProfileConfig:
    """Configuration for a vLLM profiling run."""

    model: str = ""
    tp: int = 1
    pp: int = 1
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    context_lengths: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    output_len: int = 128
    port: int = 8000
    extra_args: list[str] = field(default_factory=list)
    warmup_requests: int = 2
    measure_requests: int = 10
    timeout: int = 300


@dataclass
class MegatronProfileConfig:
    """Configuration for a Megatron training profiling run."""

    launch_cmd: str = ""
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    seq_lengths: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    pp: int = 1
    tp: int = 1
    dp: int = 1
    warmup_steps: int = 5
    measure_steps: int = 20
    extra_args: list[str] = field(default_factory=list)
