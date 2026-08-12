"""Empirical cost models fitted from profiling data.

These models use coefficients obtained by running the inference engine on
real hardware and fitting a regression model to the observed latencies.
The fitting is done by `dryrun/profiling/fit.py` via two-stage regression:

1. Stage 1 fits the non-attention (other) component against request count.
2. Stage 2 fits the attention component against the residual (actual latency
   minus fitted other latency) as a function of total context tokens.

The resulting JSON is loadable by `PSRLFitted.from_json()`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import CostModel


class PSRLFitted(CostModel):
    """
    Empirical cost model fitted from profiling data.

    Formula:
        tau = attn_b + attn_k * ctxsum + max(other_threshold, other_b + other_k * n_reqs)

    This decomposes step latency into two additive parts:

    **Attention component** (scales with total KV cache size):
    - **attn_b** (seconds): Base attention overhead. Covers kernel launch and
      memory allocation for the attention operation, independent of context
      length. Typically ~1-5ms.
    - **attn_k** (seconds/token): Marginal attention cost per context token.
      Each decoding request reads its full KV cache. `ctxsum` is the sum of
      context lengths across all requests. Determines how quickly latency
      grows as generation progresses and KV caches lengthen.

    **Non-attention component** (scales with request count):
    - **other_threshold** (seconds): Minimum non-attention latency. When the
      GPU is under-utilised (few requests), GEMMs and sampling take at least
      this long due to kernel launch and memory latency. This is the
      roofline knee for the non-attention part.
    - **other_b** (seconds): Base non-attention latency in the linear regime.
      Captures per-step fixed costs (scheduler, sampling, detokenisation)
      when above the knee.
    - **other_k** (seconds/request): Marginal non-attention cost per request.
      Covers per-request GEMM work (QKV projection, MLP, output projection).
      `n_reqs` is the number of concurrent requests.

    The attention term sits OUTSIDE the max because within a decode segment
    `n_d` (number of decoding requests) is fixed, so the non-attention
    component is constant and no kink is crossed. This means F=0 in the
    `decode_coeffs` return and the segment math never needs to handle a
    piecewise break in the attention dimension.
    """

    def __init__(self, attn_b, attn_k, other_threshold, other_b, other_k):
        self.attn_b, self.attn_k = attn_b, attn_k
        self.oth, self.ob, self.ok = other_threshold, other_b, other_k

    @classmethod
    def from_json(cls, path: str | Path, key: str = "TP1_PP1") -> PSRLFitted:
        d = json.loads(Path(path).read_text())[key]
        return cls(
            attn_b=d["attn_latency_b"],
            attn_k=d["attn_latency_k"],
            other_threshold=d["other_threshold"],
            other_b=d["other_latency_b"],
            other_k=d["other_latency_k"],
        )

    def step_time(self, n_tokens, t2, ctxsum, n_reqs):
        other = self.ob + self.ok * n_reqs
        return self.attn_b + self.attn_k * ctxsum + (self.oth if other < self.oth else other)

    def saturated(self, n_tokens, t2, ctxsum, n_reqs) -> bool:
        return self.ob + self.ok * n_reqs >= self.oth

    def decode_coeffs(self, n_d: int, ctxsum):
        other = self.ob + self.ok * n_d
        alpha = self.attn_b + self.attn_k * ctxsum + (self.oth if other < self.oth else other)
        beta = self.attn_k * n_d
        return 0, alpha, beta


class DistServe(CostModel):
    """
    DistServe Appendix A analytical cost model.

    Derives GEMM and attention costs from model architecture parameters
    (hidden dim `h`, MLP intermediate dim `m`) and hardware-specific
    calibration constants (C1-C5).

    Formula:
        tau = max(F, W + G*n_tokens + A_p*t2/b + A_d*ctxsum)

    where the derived coefficients are:
        gemm = 4*h*h + 2*h*m  (total GEMM FLOPs per token per layer)
        W = C3 + C4 * gemm    (per-step overhead, scaled by model size)
        G = C1 * gemm         (per-token GEMM cost)
        A_p = C2 * 3*h        (prefill attention per token^2)
        A_d = C5 * 3*h        (decode attention per context token)

    Args:
        h: Hidden dimension of the model (e.g. 4096 for Llama-7B).
        m: MLP intermediate dimension (e.g. 11008 for Llama-7B).
        C1: GEMM time per FLOP (seconds/FLOP). Hardware-dependent.
        C2: Prefill attention time per head-dim-FLOP.
        C3: Fixed per-step overhead (seconds).
        C4: Per-GEMM-FLOP overhead contribution to W.
        C5: Decode attention time per head-dim-FLOP.
        b: Prefill block size for t2 normalisation.
        F: Floor latency (roofline knee).
    """

    def __init__(self, h, m, C1, C2, C3, C4, C5, b=16, F=0):
        self.h, self.m, self.b, self.F = h, m, b, F
        gemm = 4 * h * h + 2 * h * m
        self._W = C3 + C4 * gemm
        self._G = C1 * gemm
        self._A_p = C2 * 3 * h
        self._A_d = C5 * 3 * h

    def _raw(self, n_tokens, t2, ctxsum):
        return self._W + self._G * n_tokens + self._A_p * t2 / self.b + self._A_d * ctxsum

    def step_time(self, n_tokens, t2, ctxsum, n_reqs=0):
        raw = self._raw(n_tokens, t2, ctxsum)
        return self.F if raw < self.F else raw

    def saturated(self, n_tokens, t2, ctxsum, n_reqs=0) -> bool:
        return self._raw(n_tokens, t2, ctxsum) >= self.F

    def decode_coeffs(self, n_d: int, ctxsum):
        alpha = self._W + self._G * n_d + self._A_d * ctxsum
        beta = self._A_d * n_d
        return self.F, alpha, beta
