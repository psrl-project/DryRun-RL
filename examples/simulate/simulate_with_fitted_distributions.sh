#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROMPT_TRACE="${PROMPT_TRACE:-${ROOT_DIR}/outputs/workload/dapo_prompt_lengths.jsonl}"
OUTPUT_TRACE="${OUTPUT_TRACE:-${ROOT_DIR}/outputs/workload/qwen3_8b_aime_output_lengths.jsonl}"

if [[ ! -f "${PROMPT_TRACE}" || ! -f "${OUTPUT_TRACE}" ]]; then
    echo "Missing trace artifacts. Run examples/workload/fit_prompt_distribution.sh and" >&2
    echo "examples/workload/fit_output_distribution.sh first." >&2
    exit 1
fi

cd "${ROOT_DIR}"
dryrun \
    workload=from_trace \
    workload.prompt.trace_path="${PROMPT_TRACE}" \
    workload.output.trace_path="${OUTPUT_TRACE}" \
    job.n_versions=20 \
    job.batch_size=8 \
    output_dir=outputs/simulate/from_fitted_distributions
