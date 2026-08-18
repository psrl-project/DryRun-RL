#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${DATASET:-/jizhicfs/lhy/data/dapo/dapo-math-17k.parquet}"
TOKENIZER="${TOKENIZER:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3-8B}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/outputs/workload/dapo_prompt_lengths.jsonl}"

cd "${ROOT_DIR}"
dryrun-workload fit-prompt \
    --dataset "${DATASET}" \
    --tokenizer "${TOKENIZER}" \
    --output "${OUTPUT}" \
    --sample-size 5000 \
    --seed 42
