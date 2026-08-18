# Workload distribution fitting examples

Fit prompt and output length distributions from real datasets and models, then feed the resulting
trace artifacts into a simulation via `workload.prompt.name=from_trace` / `workload.output.name=from_trace`.

```bash
bash examples/workload/fit_prompt_distribution.sh
bash examples/workload/fit_output_distribution.sh
```

`fit_prompt_distribution.sh` tokenizes the `prompt` column of a parquet dataset (chat messages of the
form `list[{content, role}]`, as in DAPO/AIME) with a HuggingFace tokenizer and writes a prompt-length
trace artifact.

`fit_output_distribution.sh` launches a `vllm serve` process for the given model, sends sampled dataset
prompts through its completions endpoint, records each response's completion token count, and tears the
server down when done.

Both write the same trace format: a `_meta` header (provenance and summary stats) followed by one
`{"length": N}` line per sample. Use the resulting artifacts in a simulation, see
`examples/simulate/simulate_with_fitted_distributions.sh`.
