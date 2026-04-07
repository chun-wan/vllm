#!/bin/bash
# Gemma4-31B-IT on MI300X (8-GPU TP=8) — Optimized Launch Script
# Requires: rocm/atom-dev:vllm-latest or rocm/vllm Docker image

set -e

MODEL_PATH="${MODEL_PATH:-/SHARE/gemma-4-31B-it}"
TP_SIZE="${TP_SIZE:-8}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
ENABLE_SPEC="${ENABLE_SPEC:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNING_DIR="${SCRIPT_DIR}/../configs/gemma4_mi300x/tuning"

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_MHA=1

if [ "$TP_SIZE" -eq 8 ] && [ -f "${TUNING_DIR}/hipblaslt_tuning_override_gemma4_full.txt" ]; then
    export HIPBLASLT_TUNING_OVERRIDE_FILE="${TUNING_DIR}/hipblaslt_tuning_override_gemma4_full.txt"
elif [ "$TP_SIZE" -eq 4 ] && [ -f "${TUNING_DIR}/hipblaslt_tp4_override.txt" ]; then
    export HIPBLASLT_TUNING_OVERRIDE_FILE="${TUNING_DIR}/hipblaslt_tp4_override.txt"
fi

SPEC_ARGS=""
if [ "$ENABLE_SPEC" -eq 1 ]; then
    SPEC_ARGS='--speculative-config {"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":7}'
fi

echo "=== Gemma4-31B MI300X Launch ==="
echo "Model: ${MODEL_PATH}"
echo "TP: ${TP_SIZE}, Port: ${PORT}"
echo "AITER: enabled, hipBLASLt override: ${HIPBLASLT_TUNING_OVERRIDE_FILE:-none}"
echo "Speculative decode: ${ENABLE_SPEC}"
echo "================================"

exec vllm serve "${MODEL_PATH}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --attention-backend TRITON_ATTN \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    ${SPEC_ARGS}
