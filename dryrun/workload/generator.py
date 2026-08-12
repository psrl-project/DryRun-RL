"""Workload generator: produces request streams from distributions."""

from __future__ import annotations

from ..core.types import Request


class WorkloadGenerator:
    """
    Produces a stream of Requests with configurable prompt/output length
    distributions.
    """

    def __init__(
        self,
        output_lengths: list[int],
        prompt_lengths: list[int] | int = 512,
        seed: int = 0,
    ):
        self.output_lengths = output_lengths
        self.prompt_lengths = prompt_lengths if isinstance(prompt_lengths, list) else None
        self.fixed_prompt_len = prompt_lengths if isinstance(prompt_lengths, int) else None
        self._idx = 0

    def _get_prompt_len(self) -> int:
        if self.fixed_prompt_len is not None:
            return self.fixed_prompt_len
        return self.prompt_lengths[self._idx % len(self.prompt_lengths)]

    def _get_output_len(self) -> int:
        return self.output_lengths[self._idx % len(self.output_lengths)]

    def next_request(self, rid: int, version: int, t: float) -> Request:
        req = Request(
            rid=rid,
            prompt_len=self._get_prompt_len(),
            target_len=self._get_output_len(),
            v_traj=version,
            admit_time=t,
        )
        self._idx += 1
        return req

    def batch(self, n: int, start_rid: int, version: int, t: float) -> list[Request]:
        return [self.next_request(start_rid + i, version, t) for i in range(n)]
