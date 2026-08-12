# Codebase Map

## System Architecture

DryRun-RL is a discrete-event simulator for async disaggregate RL training.
The simulation loop: admit → generate (closed-form) → complete → select batch →
(recompute) → train → weight update → abort/re-prefill.

Configuration is Hydra-based: structured `@dataclass` configs in `schema.py`,
YAML defaults in `config/defaults/`, CLI overrides via `dryrun key=value`.

Profiling pipeline: vLLM/Megatron benchmarks → CSV → two-stage regression
fitter → JSON cost model → `PSRLFitted.from_json()`.

## Directory Tree

```
dryrun/
├── __init__.py
├── _logging.py              # dryrun_logger
├── sim.py                   # Simulator class (main driver)
├── core/
│   ├── types.py             # Request, Segment, ReqStatus
│   ├── events.py            # EventType, Event, EventQueue
│   ├── component.py         # SimComponent ABC
│   └── coordinator.py       # EventCoordinator, SimResult
├── engine/
│   ├── segment.py           # Closed-form math (elapsed, steps_within, next_completion)
│   └── instance.py          # NativeInstance (continuous batching, KV, preemption)
├── policy/
│   ├── base.py              # StalenessPolicy ABC, SimState, CompleteAction
│   ├── psrl.py              # PSRLPolicy (Reserve/Occupy/Consume)
│   ├── areal.py             # ArealPolicy (token bucket)
│   ├── roll.py              # RollPolicy (admission + GC discard)
│   ├── slime.py             # SlimePolicy (loop barrier)
│   └── verl.py              # VerlPolicy (window + bounded queue)
├── cost/
│   ├── base.py              # CostModel, TrainCostModel, SyncCostModel, RecomputeCostModel
│   ├── analytical.py        # UnifiedRoofline, LinearLPS
│   ├── empirical.py         # PSRLFitted, DistServe
│   └── train_cost.py        # FixedTrainCost, AnalyticalTrainCost, sync/recompute impls
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
│   ├── schemas.py           # SweepPoint, FitResult, FitQuality, ProfileConfigs
│   ├── runner.py            # ProfileRunner (subprocess lifecycle)
│   ├── vllm_profiler.py     # VLLMProfiler (server + HTTP sweep)
│   ├── megatron_profiler.py # MegatronProfiler (training step profiling)
│   └── fit.py               # Two-stage regression fitter → JSON
├── workload/
│   ├── distributions.py     # bimodal, lognormal, powerlaw, uniform, from_trace
│   └── generator.py         # WorkloadGenerator
├── buffer/
│   └── queue.py             # CompletionBuffer, RecomputeQueue
├── viz/
│   └── plots.py             # SimPlotter (matplotlib offline)
├── cli/
│   ├── main.py              # Hydra CLI entrypoint (@hydra.main)
│   └── profile.py           # Profiling CLI (argparse subcommands)
├── router/                  # (placeholder for routing strategies)
├── trainer/                 # (placeholder for TrainerComponent)
├── sync/                    # (placeholder for WeightSyncComponent)
├── recompute/               # (placeholder for RecomputeComponent)
└── search/                  # (placeholder for auto-search)
tests/
├── conftest.py
├── test_segment_math.py     # Closed-form math correctness
├── test_e2e.py              # Full simulation tests for all 5 policies
├── test_config.py           # Hydra config + factory function tests
└── test_fit.py              # Two-stage regression fitter tests
```

## Key Interfaces

### CostModel (dryrun/cost/base.py)
- `step_time(n_tokens, t2, ctxsum, n_reqs) -> float`
- `decode_coeffs(n_d, ctxsum) -> (F, alpha, beta)`

### StalenessPolicy (dryrun/policy/base.py)
- `admit_quota(st: SimState) -> int`
- `on_admit(req, st) -> None`
- `on_complete(req, st) -> CompleteAction`
- `select_batch(st) -> list[Request] | None`
- `on_version_advance(version, st) -> None`
- `expire(st) -> list[Request]`
- `engine_version_after_train(st) -> int`

### NativeInstance (dryrun/engine/instance.py)
- `add_request(req, t) -> bool`
- `advance_to(t_limit, version) -> (completed, reached_limit)`
- `abort_all() -> list[Request]`
- `cancel(reqs, keep_tokens) -> list[Request]`

### Simulator (dryrun/sim.py)
- `__init__(cost, policy, lengths, cfg)`
- `run() -> SimResult`

### CostModelFitter (dryrun/profiling/fit.py)
- `fit_cost_model(data_or_path, key) -> FitResult`
- `save_fit_result(result, path, key) -> Path`

### ProfileRunner (dryrun/profiling/runner.py)
- `launch() -> Popen`
- `wait_healthy() -> None`
- `shutdown() -> None`

## Import Dependency Order

1. `core.types` (no internal deps)
2. `core.events` (no internal deps)
3. `cost.base` (no internal deps)
4. `cost.analytical`, `cost.empirical`, `cost.train_cost` (depend on cost.base)
5. `engine.segment` (no internal deps)
6. `engine.instance` (depends on core.types, cost.base, engine.segment)
7. `policy.base` (depends on core.types)
8. `policy.*` (depend on policy.base, core.types)
9. `workload.*` (depends on core.types)
10. `buffer.queue` (depends on core.types)
11. `config.schema` (no internal deps)
12. `config.__init__` (depends on config.schema)
13. `profiling.schemas` (no internal deps)
14. `profiling.runner` (no internal deps)
15. `profiling.fit` (depends on profiling.schemas, numpy, scipy)
16. `profiling.vllm_profiler` (depends on profiling.runner, profiling.schemas)
17. `profiling.megatron_profiler` (depends on profiling.runner, profiling.schemas)
18. `sim` (depends on core, engine, policy, cost, workload, buffer)
19. `viz.plots` (depends on sim)
20. `cli.main` (depends on everything except profiling)
21. `cli.profile` (depends on profiling)
