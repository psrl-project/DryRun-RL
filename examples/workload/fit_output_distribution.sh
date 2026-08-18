#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${DATASET:-/jizhicfs/lhy/data/dapo/aime-2024.parquet}"
TOKENIZER="${TOKENIZER:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3-8B}"
MODEL="${MODEL:-/apdcephfs_zwfy10_303541817/share_303541817/lhy/models/Qwen3-8B}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/outputs/workload/qwen3_8b_aime_output_lengths.jsonl}"

# Launches `vllm serve` itself and tears it down when collection finishes.
cd "${ROOT_DIR}"
dryrun-workload fit-output \
    --dataset "${DATASET}" \
    --tokenizer "${TOKENIZER}" \
    --model "${MODEL}" \
    --output "${OUTPUT}" \
    --tp 1 \
    --sample-size 200 \
    --max-tokens 4096 \
    --concurrency 16 \
    --seed 42
