"""Analytical cost models: roofline-based latency estimation.

These models predict per-step inference latency from hardware specifications
alone, without profiling. They decompose latency into non-attention (GEMM)
and attention components, connected by a roofline knee that captures GPU
under-utilisation at low batch sizes.
"""

from __future__ import annotations

from fractions import Fraction

from .base import CostModel


class UnifiedRoofline(CostModel):
    """
    Roofline cost model for one inference engine step.

    Formula:
        tau = max(F, W + G*n_tokens + A_p*t2/b + A_d*ctxsum)

    Below the roofline knee the GPU is under-utilised: step latency is
    pinned at the floor F regardless of batch size (adding more work just
    fills idle compute). Above the knee, latency grows linearly with load.

    Coefficient meanings:

    - **F** (seconds): Floor latency. Minimum step time set by kernel launch
      overhead and scheduler round-trip. Typically 3-8ms for vLLM on A100.
      Determines the roofline knee: the batch size at which GPU utilisation
      saturates.

    - **W** (seconds): Workload-independent overhead per step. Covers
      sampling, detokenisation, and Python-side scheduling. Usually ~1ms.

    - **G** (seconds/token): Per-token GEMM cost. Covers QKV projection,
      MLP, and output projection. For decode steps, `n_tokens = n_requests`
      (one token per request). Derives from model FLOPs:
      `G ~ 2 * model_params / peak_flops`.

    - **A_p** (seconds/token^2): Prefill attention cost coefficient. Prefill
      attention is O(L^2) in chunk length L because each token attends to
      all prior tokens in the chunk. The `t2/b` term sums squared chunk
      lengths divided by block size.

    - **A_d** (seconds/token): Decode attention cost per context token.
      Each decoding request attends to its full KV cache (prompt + generated
      tokens). `ctxsum` is the sum of context lengths across all decoding
      requests. Derives from attention FLOPs:
      `A_d ~ 2 * 3 * n_heads * head_dim / peak_flops`.

    - **b** (int): Block size for chunked prefill grouping. Affects the
      `t2/b` term normalisation.
    """

    def __init__(self, F, W=0, G=0, A_p=0, A_d=0, b=16):
        self.F, self.W, self.G, self.A_p, self.A_d, self.b = F, W, G, A_p, A_d, b

    def _raw(self, n_tokens, t2, ctxsum):
        return self.W + self.G * n_tokens + self.A_p * t2 / self.b + self.A_d * ctxsum

    def step_time(self, n_tokens, t2, ctxsum, n_reqs=0):
        raw = self._raw(n_tokens, t2, ctxsum)
        return self.F if raw < self.F else raw

    def saturated(self, n_tokens, t2, ctxsum, n_reqs=0) -> bool:
        return self._raw(n_tokens, t2, ctxsum) >= self.F

    def decode_coeffs(self, n_d: int, ctxsum):
        alpha = self.W + self.G * n_d + self.A_d * ctxsum
        beta = self.A_d * n_d
        return self.F, alpha, beta


class LinearLPS(UnifiedRoofline):
    """
    Limited-Processor-Sharing server for validation against the closed-form
    LPS order-statistics formula.
    """

    def __init__(self, r1=Fraction(1), K: int = 1):
        r1 = Fraction(r1)
        self.r1, self.K = r1, K
        super().__init__(F=1 / r1, W=0, G=1 / (r1 * K), A_p=0, A_d=0, b=1)
