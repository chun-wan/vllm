# Kimi K2.5 Pipeline Parallelism (PP) Fix for vLLM

## Branch
**https://github.com/chun-wan/vllm/tree/kimi-k25-pp-fix-v2**

## Problem Analysis: Why MLA FP8 + PP Failed Before

### Root Causes

The upstream vLLM failed to run Kimi K2.5 with PP + AITER MLA due to the following issues:

#### 1. Vision Config Attribute Name Mismatch
Kimi K2.5 model's `config.json` uses `vt_*` prefixed attribute names:
```json
{
  "vision_config": {
    "vt_hidden_size": 1152,
    "vt_num_hidden_layers": 27,
    "vt_num_attention_heads": 16,
    "vt_intermediate_size": 4304,
    "mm_hidden_size": 1152,
    "text_hidden_size": 7168
  }
}
```

But vLLM's `KimiK25VisionConfig` expected standard attribute names (`hidden_size`, `num_hidden_layers`, etc.), causing:
```
AttributeError: 'KimiK25VisionConfig' object has no attribute 'hidden_size'
```

#### 2. MM Projector Output Dimension Error
`linear_2` output dimension was set to `mm_hidden_size` (1152), but should be `text_hidden_size` (7168):
```
AssertionError: Tried to load weights of size torch.Size([7168]) 
to a parameter of size torch.Size([1152])
```

#### 3. Vision Encoder TP/DP Mode Issue
Vision Encoder weights are not Tensor Parallel sharded, but vLLM defaulted to TP mode, causing weight dimension mismatch:
```
AssertionError: Tried to load weights of size torch.Size([4304, 1152])
to a parameter of size torch.Size([4304, 7168])
```

#### 4. FP4 (MXFP4) Device Not Supported
AITER MLA attempted to use FP4 BMM, but the device does not support MXFP4:
```
AssertionError: MXFP4 is not available on your device
```

#### 5. get_vit_attn_backend Parameter Mismatch
ROCm vLLM's `mm_encoder_attention.py` passes `attn_backend_override` parameter, but upstream `vision.py` did not accept it:
```
TypeError: get_vit_attn_backend() got an unexpected keyword argument 'attn_backend_override'
```

---

## Fixes Applied

### 1. `vllm/transformers_utils/configs/kimi_k25.py`
```python
# Support vt_* prefixed attribute names
def __init__(
    self,
    # Standard attribute names
    num_attention_heads: int | None = None,
    num_hidden_layers: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    # Kimi K2.5 specific attribute names (vt_ prefix)
    vt_num_attention_heads: int = 16,
    vt_num_hidden_layers: int = 27,
    vt_hidden_size: int = 1152,
    vt_intermediate_size: int = 4304,
    text_hidden_size: int | None = None,
    ...
):
    # Use vt_* values as fallback
    self.num_attention_heads = num_attention_heads or vt_num_attention_heads
    self.num_hidden_layers = num_hidden_layers or vt_num_hidden_layers
    self.hidden_size = hidden_size or vt_hidden_size
    self.intermediate_size = intermediate_size or vt_intermediate_size
```

### 2. `vllm/model_executor/models/kimi_k25_vit.py`
```python
# MoonViT3dPretrainedModel: Map vt_* attributes
hidden_size = getattr(config, 'hidden_size', None) or getattr(config, 'vt_hidden_size', 1152)
num_hidden_layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'vt_num_hidden_layers', 27)

# KimiK25MultiModalProjector: Fix output dimension
out_dim = getattr(config, 'text_hidden_size', None) or config.mm_hidden_size
```

### 3. `vllm/model_executor/models/vision.py`
```python
# Add attn_backend_override parameter
def get_vit_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    attn_backend_override: AttentionBackendEnum | None = None,  # Added
) -> AttentionBackendEnum:

# Default to Data Parallel (not TP)
def is_vit_use_data_parallel():
    # Default to data parallel when not explicitly set to "tensor"
    return mm_encoder_tp_mode != "tensor"
```

---

## Environment Variables

```bash
export VLLM_ROCM_USE_AITER=1           # Enable AITER
export VLLM_ROCM_USE_AITER_MLA=1       # Enable AITER MLA backend
export VLLM_ROCM_USE_AITER_FP4BMM=0    # Disable FP4 BMM (MXFP4 not supported)
export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=0
```

---

## Verification

### Launch Command
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/Kimi-K2.5 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.85
```

### Test Request
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/Kimi-K2.5","prompt":"Hello","max_tokens":10}'
```

### Verified Results
- **vLLM Version**: 0.16.0rc2.dev86+gd7982daff.d20260211 (Upstream)
- **Backend**: AITER MLA (8 workers)
- **Configuration**: TP=4, PP=2
- **Text Generation**: Working

---

## Summary

| Issue | Root Cause | Fix |
|-------|------------|-----|
| Config attribute mismatch | Model uses `vt_*` prefix, vLLM expects standard names | Map `vt_*` attributes |
| Projector dimension error | Output 1152, should be 7168 | Use `text_hidden_size` |
| Vision Encoder TP failure | Weights not sharded, TP load fails | Force Data Parallel |
| FP4 not supported | Device lacks MXFP4 | Disable `FP4BMM` |
| API incompatibility | `attn_backend_override` param missing | Add parameter support |

Date: 2026-02-11
