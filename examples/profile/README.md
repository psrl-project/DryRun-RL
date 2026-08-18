# Qwen3-8B profiling

These workflows call pinned official benchmark entrypoints. DryRun-RL preserves their raw output and writes a
validated JSONL normalization as the only regression input.

Install pinned vLLM and Megatron-Bridge into the current conda environment first:

```bash
source /apdcephfs_zwfy10_303541817/share_303541817/lhy/env/dryrun-rl.sh
bash third_party/install.sh
pip install -e ".[dev]"
```

## vLLM rollout

`qwen3_8b_tp1.yaml` defines separate output-one prefill and long-output decode workloads.
For each phase the adapter launches matched official sweeps: an unprofiled timing sweep using scheduler iteration
logs, and a feature sweep using detailed profiler trace annotations. It validates the scheduler signatures before
joining low-overhead elapsed times with exact `sq`, `sk`, `sqsq`, and `sqsk` features. Client TTFT/ITL remain in
the untouched benchmark JSON for diagnostics but are not clustered into synthetic GPU steps.

Preview commands without starting a server:

```bash
bash examples/profile/vllm/profile_qwen3_8b_tp1.sh generate
bash examples/profile/vllm/profile_qwen3_8b_tp1.sh run --dry-run
```

Run and collect both phases:

```bash
bash examples/profile/vllm/profile_qwen3_8b_tp1.sh all
```

When multiple GPUs are visible, TP=1 benchmark points are sharded across them and
run concurrently. Set `CUDA_VISIBLE_DEVICES` to choose the GPU pool, or force the
legacy serial layout with `MAX_PARALLEL=1` (equivalent to
`dryrun-profile vllm run --max-parallel 1`). `--max-parallel auto` is the
default; a workflow can set the same policy with `run.max_parallel: auto|N`.
Each parallel worker gets an isolated server port, experiment directory,
scheduler log, and profiler trace directory.

Shards benchmark in parallel but boot one at a time, because concurrent imports of
torch, vLLM, and the checkpoint from a shared filesystem are slow enough to trip
`server_ready_timeout` and kill the shard before it benchmarks anything. Raise
`run.max_concurrent_startups` when the checkpoint and environment sit on a fast local
disk. A shard that still fails no longer cancels the others; rerunning the same command
retries only the unfinished shards.

The normalized output is `outputs/profile/qwen3_8b/vllm_tp1/profiles/vllm.jsonl`. Official benchmark JSON remains
under `raw/vllm_timing/` and `raw/vllm_trace/`, worker traces under `raw/vllm_traces/`, and scheduler logs plus the
pairing manifest under `control/vllm/`. Parallel experiment directories and
control files carry an `sNNN` shard suffix; collection merges them by their
original serve/bench combination path.

For a smoke test, copy the YAML and reduce each phase grid, `num_runs`, and `bench.args.num-prompts`. Keep at least
one output-one prefill point and one decode point with more than one output token.

Prefill must enumerate prompt length and concurrency because each request is a single scheduler step. Decode already
emits one record per generation step, so the decode grid only needs a few starting contexts, one medium output
length to walk `ctxsum`, and a concurrency sweep up to `max-num-seqs`.

## Megatron-Bridge training

The YAML only needs a HuggingFace checkpoint path. Architecture constraints, the Megatron-Bridge recipe, and
`--gpu` are inferred from `config.json` and `hardware.gpu_name`. `hardware.gpu_count` is the fixed world size;
remaining ranks after TP and PP become DP. Each candidate warms up for one step and records three step times, with
micro-batch size 1 and full activation recomputation. The same topology is not repeated.

`num_microbatches` is the local 1F1B depth on one DP rank. Global batch is `MBS * DP * num_microbatches`. A value
smaller than PP leaves the pipeline underfilled, so peak activation memory is lower than a full `PP` in-flight
window.

Generate candidates from architecture, topology, batch, and pipeline constraints:

```bash
bash examples/profile/megatron/profile_qwen3_8b.sh generate
```

The script defaults to candidate index zero so an accidental invocation cannot launch the entire sweep:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash examples/profile/megatron/profile_qwen3_8b.sh run
bash examples/profile/megatron/profile_qwen3_8b.sh collect
```

Set `CANDIDATE_INDEX=all` only for a provisioned sweep. A Slurm submission uses the same candidate manifest:

```bash
BACKEND=slurm CANDIDATE_INDEX=12 \
  bash examples/profile/megatron/profile_qwen3_8b.sh run
```

Each candidate uses the official Qwen3-8B recipe, mock data, training timers, and PyTorch memory history. The
adapter reads resolved configuration, TensorBoard timer scalars, memory snapshots, and OOM status. A launcher can
instead provide `dryrun_metrics.json` in a candidate directory with `step_times_s`, `rank_memory`, and optional
`microbatch_sequence_lengths`.

The seed search covers four sequence lengths at MBS=1 and local microbatch counts `{1, 4, 8, 16}`. That keeps
Hydraulis `pipeline_slots` inside range for simulated GBS up to 16 on DP=1. Qwen3-8B TP=1 training can OOM even on
an 80 GiB GPU. OOM candidates are expected data for memory regression and are preserved instead of being discarded.
