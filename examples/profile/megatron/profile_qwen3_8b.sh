#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/examples/profile/megatron/qwen3_8b.yaml}"
CANDIDATES="${CANDIDATES:-${ROOT_DIR}/outputs/profile/qwen3_8b/megatron_bridge/control/megatron_bridge/candidates.json}"
ACTION="${1:-generate}"
BACKEND="${BACKEND:-local}"
CANDIDATE_INDEX="${CANDIDATE_INDEX:-0}"

cd "${ROOT_DIR}"

selection_args=()
if [[ "${CANDIDATE_INDEX}" != "all" ]]; then
    selection_args=(--index "${CANDIDATE_INDEX}")
fi

case "${ACTION}" in
    generate)
        dryrun-profile megatron generate \
            --config "${CONFIG}" \
            --candidates "${CANDIDATES}"
        ;;
    run)
        dryrun-profile megatron run \
            --config "${CONFIG}" \
            --candidates "${CANDIDATES}" \
            --backend "${BACKEND}" \
            "${selection_args[@]}" \
            "${@:2}"
        ;;
    collect)
        dryrun-profile megatron collect \
            --config "${CONFIG}" \
            --candidates "${CANDIDATES}" \
            "${selection_args[@]}"
        ;;
    *)
        echo "Usage: $0 [generate|run|collect]" >&2
        exit 2
        ;;
esac
