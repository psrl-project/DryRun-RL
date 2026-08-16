"""Cost model abstract interface.

A cost model predicts the wall-clock latency of a single inference engine
iteration (one "step") given the current batch state. The engine alternates
between prefill steps (processing prompt chunks) and decode steps (generating
one token per request). Both are GPU-bound; the cost model captures how
latency scales with batch composition.

Two families of cost model exist:

1. **Analytical (roofline)** — derived from hardware specs (FLOP/s, memory
   bandwidth). No profiling required. Good for order-of-magnitude estimates
   and for comparing architectures before deployment.

2. **Empirical (fitted)** — parameters extracted from real profiling data via
   two-stage regression. Accurate for the specific hardware + model
   combination that was profiled.

Both families share the same abstract interface so the simulator is agnostic
to how the parameters were obtained.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class CostModel(ABC):
    """
    Predict the latency of one inference engine step.

    A single step processes all requests currently in the batch. During
    prefill the engine processes a chunk of prompt tokens; during decode it
    generates one token per request. The cost model must capture both regimes.

    All arithmetic is type-agnostic: pass `fractions.Fraction` parameters and
    the whole pipeline stays exact (used by the closed-form segment math).

    The key insight behind all cost models is the decomposition of step
    latency into two additive components:

    - **Non-attention (other):** GEMM operations (QKV projection, MLP, output
      projection). Scales with `n_tokens` (total tokens processed) because
      GEMMs are batched across all tokens in a step. Below a threshold batch
      size the GPU is under-utilised and latency is constant (the "roofline
      knee").

    - **Attention:** Self-attention computation. Scales with `ctxsum` (sum of
      context lengths across all decoding requests) because each request
      attends to its entire KV cache. During prefill, scales quadratically
      with chunk length (`t2` = sum of squared chunk lengths).
    """

    @abstractmethod
    def step_time(self, n_tokens, t2, ctxsum, n_reqs):
        """
        Predict latency of a single engine step.

        Returns:
            Strictly positive wall-clock latency. A non-positive value is a
            cost-model bug, not an idle engine.

        Args:
            n_tokens: Total tokens processed in this step. For a pure decode
                step with `n_d` requests this equals `n_d` (one new token
                each). For a prefill step it equals the chunk size.
            t2: Sum of squared prefill chunk lengths. Captures the O(L^2)
                scaling of flash-attention during prefill. Zero during pure
                decode steps.
            ctxsum: Sum of context lengths over all decoding requests. Each
                decoding request attends to `prompt_len + generated_so_far`
                tokens in its KV cache, and the attention kernel processes
                all of them.
            n_reqs: Number of requests in this step. Used by empirical models
                where non-attention overhead scales per-request (kernel
                launch overhead, sampling, etc.).
        """

    @abstractmethod
    def decode_coeffs(self, n_d: int, ctxsum) -> tuple:
        """
        Return (F, alpha, beta) for a pure-decode segment.

        During decode, context grows by `n_d` tokens per step (one per
        request). The per-step latency therefore follows:

            tau(k) = max(F, alpha + beta * k)

        where k is the step index within the segment:
        - F: Floor latency. When the GPU is under-utilised, latency is
          pinned at this minimum regardless of batch size.
        - alpha: Latency at the start of the segment (k=0), capturing
          current batch composition.
        - beta: Marginal latency increase per step, equal to the attention
          cost of `n_d` additional context tokens.

        The closed-form advance (`engine/advance.py`) integrates this
        piecewise-linear function analytically, avoiding per-token simulation.

        Args:
            n_d: Number of concurrently decoding requests.
            ctxsum: Sum of their context lengths at segment start.
        """

    def saturated(self, n_tokens, t2, ctxsum, n_reqs) -> bool:
        """Whether the GPU is fully utilised (above the roofline knee)."""
        return False
