# Gemma4-31B MI300X Optimization Patches

## Overview
Patches and tuning configs for running Gemma4-31B-IT on AMD Instinct MI300X (gfx942) with vLLM + AITER.

## Patches

### fix_aiter_1702_causal.patch
Fixes ROCm/aiter#1702: Makes causal flag dynamic in `rocm_aiter_unified_attn.py`.
Original code hardcodes `causal=True`, blocking bidirectional attention needed by Gemma4 vision tokens.

Apply inside container:
```bash
cd /usr/local/lib/python3.12/dist-packages
patch -p1 < /path/to/fix_aiter_1702_causal.patch
```

## Tuning Configs

### hipblaslt_tuning_override_gemma4_full.txt
hipBLASLt Tensile Lite tuning override for Gemma4-31B TP=8 (27 shapes, BF16).

### hipblaslt_tp4_override.txt
hipBLASLt tuning for TP=4 deployment (30 shapes, BF16).

## Recommended Launch Commands

### TP=8 Best Throughput (with speculative decoding)
```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_RMSNORM=1 \
VLLM_ROCM_USE_AITER_LINEAR=1 \
VLLM_ROCM_USE_AITER_MHA=1 \
HIPBLASLT_TUNING_OVERRIDE_FILE=/path/to/hipblaslt_tuning_override_gemma4_full.txt \
vllm serve google/gemma-4-31B-it \
    --tensor-parallel-size 8 --trust-remote-code \
    --gpu-memory-utilization 0.9 --attention-backend TRITON_ATTN \
    --quantization fp8 --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":7}'
```

### TP=4 GPU-Efficient
```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_RMSNORM=1 \
VLLM_ROCM_USE_AITER_LINEAR=1 \
VLLM_ROCM_USE_AITER_MHA=1 \
vllm serve google/gemma-4-31B-it \
    --tensor-parallel-size 4 --trust-remote-code \
    --gpu-memory-utilization 0.9 --attention-backend TRITON_ATTN \
    --quantization fp8 --kv-cache-dtype fp8 \
    --max-model-len 32768
```

## Performance (MI300X 8-GPU, con=40, 10K→512 random)
| Config | Output TPS | TPOT (ms) | Gain |
|--------|-----------|-----------|------|
| Baseline | ~202 | ~172 | — |
| + AITER full | ~232 | ~151 | +15% |
| + Spec decode (5) | **288** | **110** | **+43%** |

## Versions
- vLLM: 0.18.2rc1 (ROCm 7.2.1)
- AITER: 0.1.10.post2
- PyTorch: 2.10.0 (ROCm)
- hipBLASLt: 1.2.2
