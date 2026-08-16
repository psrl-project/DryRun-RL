# Codebase Map

## System Architecture

DryRun-RL is a discrete-event simulator for async disaggregate RL training.
The simulation loop: admit → generate (closed-form) → complete → select batch →
(recompute) → train → weight update → abort/re-prefill.

Code is organised around the RL loop components. Everything that models a stage
of the loop lives under `dryrun/component/` (rollout, training, sync, recompute),
while `dryrun/simulator/` drives them and `dryrun/policy/` decides staleness.

Configuration is Hydra-based: structured `@dataclass` configs in `schema.py`,
YAML defaults in `config/defaults/`, CLI overrides via `dryrun key=value`.

Profiling pipeline: vLLM/Megatron benchmarks → CSV → two-stage regression
fitter → JSON cost model → `PSRLFitted.from_json()`.

## Directory Tree

```
dryrun/
├── __init__.py
├── _logging.py              # dryrun_logger
├── component/               # One subpackage per RL loop stage
│   ├── rollout/
│   │   ├── engine/
│   │   │   ├── request.py       # ReqStatus, Segment, Request (shared vocabulary)
│   │   │   ├── advance.py       # Closed-form decode advance (elapsed, steps_within, next_completion)
│   │   │   ├── instance.py      # NativeInstance (continuous batching, KV, preemption)
│   │   │   ├── event.py         # EventKind, EngineEvent
│   │   │   ├── state.py         # StepStat, InstanceLoad
│   │   │   └── snapshot.py      # ReqSnapshot, InstanceSnapshot (event rollback)
│   │   └── router/              # (placeholder for routing strategies)
│   ├── training/
│   │   └── cost.py          # TrainCostModel, FixedTrainCost, AnalyticalTrainCost
│   ├── sync/
│   │   └── cost.py          # SyncCostModel, FixedSyncCost, BandwidthSyncCost
│   └── recompute/
│       └── cost.py          # RecomputeCostModel, FixedRecomputeCost, AnalyticalRecomputeCost
├── simulator/
│   ├── config.py            # SimConfig, SimResult
│   └── simulator.py         # Simulator (main driver)
├── policy/
│   ├── base.py              # StalenessPolicy ABC, SimState, CompleteAction
│   ├── psrl.py              # PSRLPolicy (Reserve/Occupy/Consume)
│   ├── areal.py             # ArealPolicy (token bucket)
│   ├── roll.py              # RollPolicy (admission + GC discard)
│   ├── slime.py             # SlimePolicy (loop barrier)
│   └── verl.py              # VerlPolicy (window + bounded queue)
├── cost/                    # Inference step time models (shared by all instances)
│   ├── base.py              # CostModel ABC
│   ├── analytical.py        # UnifiedRoofline, LinearLPS
│   └── empirical.py         # PSRLFitted, DistServe
├── config/
│   ├── __init__.py          # register_configs() for Hydra ConfigStore
│   ├── schema.py            # @dataclass structured configs (SimulateConfig, etc.)
│   └── defaults/
│       ├── simulate.yaml    # Top-level default config
│       ├── cost_model/      # roofline.yaml, psrl_fitted.yaml, distserve.yaml
│       ├── policy/          # psrl.yaml, areal.yaml, roll.yaml, slime.yaml, verl.yaml
│       ├── workload/        # uniform.yaml, bimodal.yaml, lognormal.yaml, powerlaw.yaml
│       └── train_cost/      # fixed.yaml, analytical.yaml
├── profiling/
│   ├── __init__.py          # Public exports
│   ├── schemas.py           # SweepPoint, FitResult, FitQuality, profile configs
│   ├── runner.py            # ProfileRunner (subprocess lifecycle)
│   ├── vllm_profiler.py     # run_vllm_profile (server + HTTP sweep)
│   ├── megatron_profiler.py # run_megatron_profile (training step profiling)
│   └── fit.py               # Two-stage regression fitter → JSON
├── workload/
│   ├── distributions.py     # bimodal, lognormal, powerlaw, uniform, from_trace
│   └── generator.py         # WorkloadGenerator
├── buffer/
│   └── queue.py             # CompletionBuffer, RecomputeQueue
├── viz/
│   └── plots.py             # SimPlotter (matplotlib offline)
└── cli/
    ├── main.py              # Hydra CLI entrypoint (@hydra.main)
    └── profile.py           # Profiling CLI (argparse subcommands)
tests/
├── conftest.py
├── test_advance.py          # Closed-form advance correctness
├── test_e2e.py              # Full simulation tests for all 5 policies
├── test_config.py           # Hydra config + factory function tests
└── test_fit.py              # Two-stage regression fitter tests
```

## Key Interfaces

### Request (dryrun/component/rollout/engine/request.py)
The unit of work every component handles. Import it from the engine package:
`from ..component.rollout.engine import Request`.
- `remaining`, `ctx`, `v_min`, `v_max` properties
- `add_tokens(version, n) -> None` appends to the current `Segment`
- `staleness_true()`, `staleness_reported()`

### CostModel (dryrun/cost/base.py)
- `step_time(n_tokens, t2, ctxsum, n_reqs) -> float`
- `decode_coeffs(n_d, ctxsum) -> (F, alpha, beta)`

### NativeInstance (dryrun/component/rollout/engine/instance.py)
- `add_request(req, t) -> bool`
- `advance_to(t_limit, version) -> (completed, reached_limit)`
- `advance_one_event(version) -> EngineEvent | None`, paired with `undo_last_event()`
- `abort_all() -> list[Request]`
- `cancel(reqs, keep_tokens) -> list[Request]`

### StalenessPolicy (dryrun/policy/base.py)
- `admit_quota(st: SimState) -> int`
- `on_admit(req, st) -> None`
- `on_complete(req, st) -> CompleteAction`
- `select_batch(st) -> list[Request] | None`
- `on_version_advance(version, st) -> None`
- `expire(st) -> list[Request]`
- `engine_version_after_train(st) -> int`

### Simulator (dryrun/simulator/simulator.py)
- `__init__(cost, policy, lengths, cfg, train_cost, sync_cost, recompute_cost)`
- `run() -> SimResult`

### CostModelFitter (dryrun/profiling/fit.py)
- `fit_cost_model(data_or_path, key) -> FitResult`
- `save_fit_result(result, path, key) -> Path`

### ProfileRunner (dryrun/profiling/runner.py)
- `launch() -> Popen`
- `wait_healthy() -> None`
- `shutdown() -> None`

## Import Dependency Order

1. `component.rollout.engine.request` (no internal deps)
2. `component.rollout.engine.advance` (no internal deps)
3. `component.{training,sync,recompute}.cost` (no internal deps)
4. `cost.base` (no internal deps)
5. `cost.analytical`, `cost.empirical` (depend on cost.base)
6. `component.rollout.engine.event`, `.state`, `.snapshot` (depend on engine.request)
7. `component.rollout.engine.instance` (depends on the other engine modules, cost.base)
8. `policy.base` (depends on engine.request)
9. `policy.*` (depend on policy.base, engine.request)
10. `workload.*` (depends on engine.request)
11. `buffer.queue` (depends on engine.request)
12. `simulator.config` (depends on engine.request)
13. `simulator.simulator` (depends on engine, policy, cost, component.*)
14. `config.schema` (no internal deps)
15. `config.__init__` (depends on config.schema)
16. `profiling.schemas`, `profiling.runner` (no internal deps)
17. `profiling.fit` (depends on profiling.schemas, numpy, scipy)
18. `profiling.vllm_profiler`, `profiling.megatron_profiler` (depend on profiling.runner, profiling.schemas)
19. `viz.plots` (depends on simulator)
20. `cli.main` (depends on everything except profiling)
21. `cli.profile` (depends on profiling)

## Naming Notes

Two unrelated concepts both use the word "segment":

- `Segment` in `engine/request.py` is a stretch of generated tokens produced
  under one weight version. It is what the staleness metrics read.
- A decode run in `engine/advance.py` is a stretch of steps over a fixed
  batch with no prefill mixed in. It is a unit of time integration, not data.

The closed-form helpers live in `advance.py` so the word `Segment` belongs to
the request data model alone.
