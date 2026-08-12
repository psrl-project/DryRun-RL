# DryRun-RL

Async disaggregate RL training simulator. Evaluates staleness control policies, GPU allocation strategies, and parallelism configurations without full GPU deployments.

## Quick Start

```bash
# Install the package
pip install -e .

# Run a simulation with PSRL policy (Hydra CLI)
dryrun policy=psrl n_versions=20 n_instances=4 batch_size=8

# Run with bimodal workload
dryrun policy=areal workload=bimodal n_versions=10

# Compare all policies
for p in psrl areal roll slime verl; do
    dryrun policy=$p n_versions=10 output_dir=outputs/$p
done

# Hydra multirun sweep
dryrun --multirun policy=psrl,areal,roll n_instances=1,2,4
```

## Simulation CLI (Hydra)

```bash
dryrun [key=value ...]

# Key overrides:
#   policy=psrl|areal|roll|slime|verl    Staleness control policy
#   cost_model=roofline|psrl_fitted      Inference cost model
#   workload=uniform|bimodal|lognormal|powerlaw
#   train_cost=fixed|analytical          Training cost model
#   n_versions=20                        Training steps to simulate
#   n_instances=1                        Number of rollout instances
#   batch_size=8                         Training batch size
#   max_staleness=2                      Maximum allowed staleness
#   output_dir=outputs                   Output directory for plots

# Use a fitted cost model from profiling
dryrun cost_model=psrl_fitted cost_model.path=cost_models/llama8b.json

# View the resolved config
dryrun --cfg job
```

## Profiling CLI

Profile real hardware to build accurate cost models:

```bash
# Profile vLLM decode performance
dryrun-profile vllm \
    --model meta-llama/Llama-3-8B \
    --tp 1 \
    --batch-sizes 1,2,4,8,16,32 \
    --ctx-lengths 128,256,512,1024,2048 \
    --output profiles/llama8b_tp1.csv

# Profile Megatron training step times
dryrun-profile megatron \
    --launch-cmd "torchrun --nproc_per_node=4 train.py" \
    --batch-sizes 1,2,4,8 \
    --seq-lengths 512,1024,2048 \
    --output profiles/llama8b_train.csv

# Fit cost model from profiling data
dryrun-profile fit \
    --input profiles/llama8b_tp1.csv \
    --key TP1_PP1 \
    --output cost_models/llama8b.json

# Use fitted model in simulation
dryrun cost_model=psrl_fitted cost_model.path=cost_models/llama8b.json
```

## Architecture

Layered event-driven simulation with closed-form segment advancement:

```
WorkloadGenerator → Router → RolloutEngine (N instances) → [Recompute] → Trainer → WeightSync
```

Each decode segment is integrated analytically: `tau(k) = max(F, alpha + beta*k)`. The event queue has O(n_instances) pending completions, not O(total_tokens).

## Cost Models

Two pluggable families with different tradeoffs:

### Analytical Roofline (no hardware needed)

```
tau = max(F, W + G*n_tokens + A_p*t2/batch + A_d*ctxsum)
```

| Coefficient | Physical meaning | Typical value (A100) |
|-------------|-----------------|---------------------|
| **F** | Floor latency (kernel launch + scheduler overhead) | 3-8 ms |
| **W** | Per-step fixed overhead (sampling, detokenisation) | ~1 ms |
| **G** | Per-token GEMM cost (QKV + MLP + output projection) | ~0.1 ms/token |
| **A_p** | Prefill attention cost (O(L²) in chunk length) | model-dependent |
| **A_d** | Decode attention cost per KV cache token | ~0.1 µs/token |
| **b** | Prefill block size for chunked prefill | 16 |

### Empirical Fitted (from profiling data)

```
tau = attn_b + attn_k*ctxsum + max(other_threshold, other_b + other_k*n_reqs)
```

| Coefficient | Physical meaning | How it's fitted |
|-------------|-----------------|----------------|
| **attn_b** | Base attention overhead (kernel launch, memory alloc) | Stage 2: linear intercept |
| **attn_k** | Marginal attention cost per context token | Stage 2: linear slope |
| **other_threshold** | Minimum non-attention latency (roofline knee) | Stage 1: threshold parameter |
| **other_b** | Base non-attention latency in linear regime | Stage 1: linear intercept |
| **other_k** | Per-request non-attention cost (GEMM work per request) | Stage 1: linear slope |

The two-stage fitting separates attention (scales with KV cache size) from non-attention (scales with request count) to capture the roofline knee where the GPU transitions from under-utilised to compute-bound.

## Staleness Policies

| Policy | Mechanism | Bound Holds? | Cost |
|--------|-----------|-------------|------|
| PSRL | Reserve/Occupy/Consume | Always | Training stall (blocking) |
| AReaL | Cumulative token bucket | Tail-driven failure | None (no enforcement) |
| ROLL | Admission + GC discard | Always | Silent data loss |
| Slime | Loop barrier | Always | Training stall (barrier) |
| verl | Window + bounded queue | Overflow-driven failure | Oldest dropped |

## Package Structure

```
dryrun/
    core/           Event coordinator, types, component interface
    engine/         Rollout instances, segment math
    policy/         5 staleness policies (psrl, areal, roll, slime, verl)
    cost/           Analytical + empirical cost models
    config/         Hydra structured configs + YAML defaults
    profiling/      vLLM/Megatron profiling + two-stage regression fitter
    workload/       Length distribution generators
    buffer/         Queue data structures
    viz/            Matplotlib offline plots
    cli/            Hydra simulation CLI + profiling CLI
    sim.py          Simulation driver
```

## Running Tests

```bash
source /apdcephfs_zwfy10/share_303541817/lhy/env/dryrun-rl.sh
python -m pytest tests/ -v
```

## Programmatic API

```python
from dryrun.cost.analytical import UnifiedRoofline
from dryrun.policy.psrl import PSRLPolicy
from dryrun.sim import SimConfig, Simulator
from dryrun.workload.distributions import bimodal

cost = UnifiedRoofline(F=0.005, W=0.001, G=0.0001, A_d=1e-7)
policy = PSRLPolicy()
lengths = bimodal(200, short_len=50, long_len=500, long_frac=0.25, seed=42)
cfg = SimConfig(batch_size=8, max_staleness=2, n_versions=20, n_instances=4)

sim = Simulator(cost=cost, policy=policy, lengths=lengths, cfg=cfg)
result = sim.run()

print(f"Throughput: {result.throughput:.1f} tokens/s")
print(f"Max staleness: {result.max_staleness_true}")
```

### Profiling API

```python
from dryrun.profiling.fit import fit_cost_model, save_fit_result
from dryrun.cost.empirical import PSRLFitted

# Fit from CSV
result = fit_cost_model("profiles/llama8b_tp1.csv")
save_fit_result(result, "cost_models/llama8b.json", key="TP1_PP1")

# Load and use
model = PSRLFitted.from_json("cost_models/llama8b.json", key="TP1_PP1")
latency = model.step_time(n_tokens=16, t2=0, ctxsum=8000, n_reqs=16)
```

## Roadmap

- [ ] Auto-search: binary search over GPU allocation
- [ ] Routing strategies: throughput_balance, cache_aware
- [ ] Prefix cache modeling
- [ ] Multiple rollout model versions (heterogeneous instances)
- [ ] Elastic scaling during simulation
- [ ] Online visualization (live dashboard)
- [ ] Multi-turn / agent environment calls
