#!/bin/bash
set -euo pipefail

# Minimal launcher for direct p(theta | x_obs) SBI variants.
# Override variables via environment or edit below.

CACHE_PATH="${CACHE_PATH:-/path/to/build_arrays_cache.npz}"
EXCLUDE_INDICES="${EXCLUDE_INDICES:-/path/to/test_indices.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output_sbi_variants}"
RUN_NAME="${RUN_NAME:-sbi_fm_v1}"
METHOD="${METHOD:-flow_matching}"   # flow_matching | normalizing_flow
NF_BACKEND="${NF_BACKEND:-zuko}"    # zuko | nflows
NF_FAMILY="${NF_FAMILY:-nsf}"       # nsf | maf | nice
DEVICE="${DEVICE:-cuda}"
CONFIG_PATH="${CONFIG_PATH:-}"

mkdir -p "${OUTPUT_DIR}"

if [[ -n "${CONFIG_PATH}" ]]; then
  python sbi_variants/train_sbi_posterior.py \
    --config "${CONFIG_PATH}" \
    --device "${DEVICE}"
else
  python sbi_variants/train_sbi_posterior.py \
    --cache-path "${CACHE_PATH}" \
    --exclude-indices "${EXCLUDE_INDICES}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --method "${METHOD}" \
    --nf-backend "${NF_BACKEND}" \
    --nf-family "${NF_FAMILY}" \
    --batch-size 4096 \
    --epochs 300 \
    --lr 8e-4 \
    --lr-min 1e-5 \
    --val-split 0.1 \
    --time-prior-exponent 0.0 \
    --use-missingness-context \
    --amp \
    --device "${DEVICE}"
fi
