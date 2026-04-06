# Gemma 4 31B on MI250 (gfx90a) -- vLLM Configuration

## Requirements

- AMD Instinct MI250/MI250X (gfx90a, CDNA2)
- ROCm 7.0+ (tested with 7.2.1)
- AITER fork: https://github.com/chun-wan/aiter/tree/gemma4_mi250
- Model: google/gemma-4-31B-it (BF16, 59GB)

## vLLM Launch Command

```bash
# BF16 TP=2 (最低配置, 2 GCDs)
vllm serve google/gemma-4-31B-it \
  --attention-backend TRITON_ATTN \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --trust-remote-code

# BF16 TP=8 (全部 4 張 MI250 卡, 8 GCDs)
vllm serve google/gemma-4-31B-it \
  --attention-backend TRITON_ATTN \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --trust-remote-code
```

## Performance Baselines (MI250, BF16)

| Config | Output TPS (con=2) | Output TPS (con=40) | TPOT (con=2) |
|--------|-------------------|--------------------|--------------| 
| TP=2 torch.compile | 38.35 | 361.64 | 50.88 ms |
| TP=2 enforce-eager | 31.87 | 331.98 | 60.66 ms |
| TP=8 torch.compile | 66.35 | 319.14 | 27.82 ms |

## Accuracy Baseline

- GSM8K CoT 0-shot (TRITON_ATTN): **78.5%** (±2.91%, 200 samples)

## AITER Operators Verified on gfx90a

| Operator | Module | Status | Speedup |
|----------|--------|--------|---------|
| CK RMSNorm | module_rmsnorm | PASS | 1.42x |
| CK Flash Attention D=256 | mha_fwd | PASS | 1.75x (prefill) |
| GeGLU activation | module_activation | PASS | -- |
| SiLU*Mul | module_activation | PASS | -- |
| KV Cache | module_cache | PASS | -- |

## Kernel Math Library Map

| Operation | Library | Notes |
|-----------|---------|-------|
| GEMM | hipBLASLt | Tensile gfx90a MFMA |
| Attention (D=256, 50 layers) | Triton | Runtime-compiled for gfx90a |
| Attention (D=512, 10 layers) | Triton | Same |
| RMSNorm | AITER CK / PyTorch | 1.42x with CK |
| RoPE | vLLM HIP | Standard + Proportional |
| GeGLU | AITER CK | After FP8 guard fix |

## Known Limitations

- BF16 TP=1 does NOT fit (model=62GB > GCD=64GB)
- FP8 quantization not available on MI250 (CDNA2)
- ASM kernels (hsa/gfx942/) cannot run on gfx90a
