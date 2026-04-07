# vLLM Gemma 4 MI355X Patches

## Base Image
- Docker: `vllm/vllm-openai-rocm:gemma4`
- vLLM Version: `0.18.2rc1.dev71+ge92668e83.rocm721`
- ROCm: 7.2.1
- PyTorch: 2.9.1+rocm7.2.1
- AITER: amd-aiter 0.1.10.post2
- GPU: AMD Instinct MI355X (gfx950)

## Patches Applied

### 1. patch_vllm_renderer.py
Skip `copy.deepcopy(tokenizer)` in `vllm/renderers/base.py`.
Gemma 4 tokenizer triggers Rust RefCell borrow error on deepcopy.

### 2. ATOM plugin (pip install git+https://github.com/ROCm/ATOM.git)
Auto-detected as vLLM OOT plugin via entry_points.
Files installed at: `atom/plugin/vllm/`

### 3. AITER env vars (for vLLM+AITER without ATOM)
```
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_LINEAR=0   # CK dispatcher overhead
VLLM_ROCM_USE_AITER_RMSNORM=1
```

## Launch Commands
See `start_vllm_atom_plugin.sh` and `patch_vllm_renderer.py`
