"""Single rollout instance: event-driven with closed-form segment advancement.

Models one inference engine instance with continuous batching, KV cache
accounting, and preemption.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import ReqStatus, Request
from ..cost.base import CostModel
from . import segment as seg


def blocks_for(n_tokens: int, block_size: int) -> int:
    return (n_tokens + block_size - 1) // block_size


@dataclass
class StepStat:
    """Aggregated engine steps."""

    n_tokens: int
    n_decode: int
    n_prefill: int
    saturated: bool
    steps: int = 1
    saturated_steps: int = 0

    def __post_init__(self) -> None:
        if self.saturated and self.saturated_steps == 0:
            self.saturated_steps = self.steps


@dataclass
class InstanceLoad:
    """Snapshot of instance load for routing decisions."""

    instance_id: int
    n_waiting: int
    n_running: int
    kv_utilization: float
    total_ctx: int


class NativeInstance:
    """A single simulated engine instance."""

    def __init__(
        self,
        cost: CostModel,
        token_budget: int,
        kv_blocks: int,
        block_size: int = 16,
        max_running: int | None = None,
        chunk_cap: int | None = None,
        instance_id: int = 0,
    ):
        self.cost = cost
        self.M = token_budget
        self.kv_blocks = kv_blocks
        self.block_size = block_size
        self.max_running = max_running
        self.chunk_cap = chunk_cap
        self.instance_id = instance_id

        self.t_local: float = 0.0
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.step_stats: list[StepStat] = []
        self.n_preemptions: int = 0
        self.completed: list[Request] = []

    @property
    def n_active(self) -> int:
        return len(self.waiting) + len(self.running)

    def get_load(self) -> InstanceLoad:
        return InstanceLoad(
            instance_id=self.instance_id,
            n_waiting=len(self.waiting),
            n_running=len(self.running),
            kv_utilization=self.blocks_used() / max(1, self.kv_blocks),
            total_ctx=sum(r.ctx for r in self.running),
        )

    def blocks_used(self) -> int:
        return sum(blocks_for(r.prefilled, self.block_size) for r in self.waiting + self.running)

    def free_blocks(self) -> int:
        return self.kv_blocks - self.blocks_used()

    def add_request(self, req: Request, t: float) -> bool:
        if self.max_running is not None and self.n_active >= self.max_running:
            return False
        need = blocks_for(req.prompt_len + req.target_len, self.block_size)
        if need > self.kv_blocks:
            return False
        committed = sum(
            blocks_for(r.prompt_len + r.target_len, self.block_size)
            for r in self.waiting + self.running
        )
        if committed + need > self.kv_blocks:
            return False
        req.status = ReqStatus.WAITING
        req.dispatch_time = t
        req.instance_id = self.instance_id
        self.waiting.append(req)
        return True

    def _promote_ready(self) -> None:
        for req in list(self.waiting):
            if req.prompt_len + req.generated - req.prefilled <= 0:
                self.waiting.remove(req)
                req.status = ReqStatus.RUNNING
                self.running.append(req)

    def _prefill_budget_step(self, version: int):
        n_d = len(self.running)
        n_decode = min(n_d, self.M)
        decoding = self.running[:n_decode]
        budget = self.M - n_decode
        t2 = 0
        prefill_tokens = 0
        finished_prefill: list[Request] = []

        free = self.free_blocks()
        for req in sorted(self.waiting, key=lambda r: (r.dispatch_time, r.rid)):
            if budget <= 0:
                break
            need = req.prompt_len + req.generated - req.prefilled
            if need <= 0:
                finished_prefill.append(req)
                continue
            chunk = min(need, budget)
            if self.chunk_cap:
                chunk = min(chunk, self.chunk_cap)
            have = blocks_for(req.prefilled, self.block_size)
            chunk = min(chunk, (have + max(0, free)) * self.block_size - req.prefilled)
            if chunk <= 0:
                break
            free -= blocks_for(req.prefilled + chunk, self.block_size) - have
            req.prefilled += chunk
            prefill_tokens += chunk
            t2 += chunk * chunk
            budget -= chunk
            if req.prefilled >= req.prompt_len + req.generated:
                finished_prefill.append(req)

        ctxsum = sum(r.ctx for r in self.running)
        n_tokens = n_decode + prefill_tokens
        dt = self.cost.step_time(n_tokens, t2, ctxsum, n_d + len(self.waiting))
        self.step_stats.append(
            StepStat(
                n_tokens,
                n_decode,
                prefill_tokens,
                self.cost.saturated(n_tokens, t2, ctxsum, n_d + len(self.waiting)),
                steps=1,
            )
        )

        stepped = []
        oldest = min(self.running, key=lambda r: (r.dispatch_time, r.rid)) if self.running else None
        for r in decoding:
            grows = (r.prefilled % self.block_size) == 0
            if grows and self.free_blocks() <= 0 and r is not oldest:
                continue
            r.add_tokens(version, 1)
            r.prefilled += 1
            stepped.append(r)
        newly_done = [r for r in stepped if r.remaining <= 0]

        for req in finished_prefill:
            self.waiting.remove(req)
            req.status = ReqStatus.RUNNING
            self.running.append(req)
        return dt, newly_done

    def _decode_coeffs(self):
        n_d = len(self.running)
        ctxsum = sum(r.ctx for r in self.running)
        return self.cost.decode_coeffs(n_d, ctxsum)

    def _kv_exhaustion_steps(self) -> int | None:
        base = self.blocks_used()
        if base > self.kv_blocks:
            return 0

        def blocks_at(k: int) -> int:
            return sum(blocks_for(r.prefilled + k, self.block_size) for r in self.running) + sum(
                blocks_for(r.prefilled, self.block_size) for r in self.waiting
            )

        cap = max((r.remaining for r in self.running), default=0)
        if cap == 0 or blocks_at(cap) <= self.kv_blocks:
            return None
        lo, hi = 0, cap
        while lo < hi:
            mid = (lo + hi) // 2
            if blocks_at(mid) > self.kv_blocks:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def run_until(self, t_limit: float, version: int) -> list[Request]:
        done = []
        while self.t_local < t_limit:
            got, reached = self.advance_to(t_limit, version)
            done.extend(got)
            if reached:
                break
            if not self.running and not self.waiting:
                break
        return done

    def advance_to(
        self,
        t_limit: float,
        version: int,
        max_steps: int = 100_000,
    ) -> tuple[list[Request], bool]:
        """
        Advance this instance up to t_limit. Returns (completed, reached_limit).
        """
        done: list[Request] = []
        steps = 0
        while self.t_local < t_limit:
            self._promote_ready()
            if self.waiting:
                if steps >= max_steps:
                    return done, False
                steps += 1
                dt, newly = self._prefill_budget_step(version)
                if self.t_local + dt > t_limit and not newly:
                    self.t_local = t_limit
                    return done, True
                if dt <= 0:
                    self.t_local = t_limit
                    return done, True
                self.t_local += dt
                done.extend(self._retire(newly))
                if newly:
                    return done, False
                if self._deadlocked():
                    if not self._preempt_for_progress():
                        self.t_local = t_limit
                        return done, True
                continue

            if not self.running:
                self.t_local = t_limit
                return done, True

            F, alpha, beta = self._decode_coeffs()
            k_done, _ = seg.next_completion([r.remaining for r in self.running])
            k_kv = self._kv_exhaustion_steps()
            k_int = k_done if k_kv is None else min(k_done, k_kv)
            budget = t_limit - self.t_local
            k_budget = seg.steps_within(F, alpha, beta, budget)

            k = min(k_int, k_budget)
            if k > 0:
                n_d = len(self.running)
                for r in self.running:
                    r.add_tokens(version, k)
                    r.prefilled += k
                self.t_local += seg.elapsed(F, alpha, beta, k)
                k0 = seg.crossing_step(F, alpha, beta, k_cap=k)
                self.step_stats.append(
                    StepStat(
                        n_tokens=n_d * k,
                        n_decode=n_d * k,
                        n_prefill=0,
                        saturated=k > k0,
                        steps=k,
                        saturated_steps=k - k0,
                    )
                )

            if self.blocks_used() > self.kv_blocks:
                self._preempt_one()
                return done, False

            if k == k_budget and k < k_int:
                self.t_local = t_limit
                return done, True

            newly = [r for r in self.running if r.remaining <= 0]
            if newly:
                done.extend(self._retire(newly))
                return done, False
            if k_kv is not None and k >= k_kv:
                self._preempt_one()
                return done, False
            if k == 0:
                self.t_local = t_limit
                return done, True
        return done, True

    def _deadlocked(self) -> bool:
        if not self.running or self.free_blocks() > 0:
            return False
        decode_blocked = all((r.prefilled % self.block_size) == 0 for r in self.running)
        prefill_blocked = any(
            r.prompt_len + r.generated - r.prefilled > 0 for r in self.waiting
        )
        return decode_blocked and prefill_blocked

    def _preempt_for_progress(self) -> bool:
        if not self.running:
            return False
        oldest = min(self.running + self.waiting, key=lambda r: (r.dispatch_time, r.rid))
        victims = [r for r in self.running if r is not oldest]
        if not victims:
            return False
        victim = max(victims, key=lambda r: (r.dispatch_time, r.rid))
        self._preempt_one(victim)
        return True

    def _retire(self, reqs: list[Request]) -> list[Request]:
        out = []
        for r in reqs:
            if r in self.running:
                self.running.remove(r)
            r.status = ReqStatus.DONE
            r.complete_time = self.t_local
            self.completed.append(r)
            out.append(r)
        return out

    def _preempt_one(self, victim: Request | None = None) -> Request | None:
        if not self.running:
            return None
        if victim is None:
            victim = self.running[-1]
        self.running.remove(victim)
        victim.status = ReqStatus.PREEMPTED
        victim.num_preemptions += 1
        victim.reprefill_tokens += victim.ctx
        victim.prefilled = 0
        self.waiting.insert(0, victim)
        self.n_preemptions += 1
        return victim

    def _release(self, req: Request, keep_tokens: bool = True) -> None:
        req.num_aborts += 1
        if keep_tokens:
            req.reprefill_tokens += req.ctx
        req.prefilled = 0

    def abort_all(self) -> list[Request]:
        aborted = list(self.running) + list(self.waiting)
        for r in aborted:
            self._release(r, keep_tokens=True)
            r.status = ReqStatus.WAITING
        self.waiting = aborted[:]
        self.running = []
        return aborted

    def cancel(self, reqs: list[Request], keep_tokens: bool = True) -> list[Request]:
        canceled = []
        for r in reqs:
            if r in self.running:
                self.running.remove(r)
            elif r in self.waiting:
                self.waiting.remove(r)
            else:
                continue
            self._release(r, keep_tokens=keep_tokens)
            if keep_tokens:
                r.status = ReqStatus.WAITING
                self.waiting.append(r)
            canceled.append(r)
        return canceled
