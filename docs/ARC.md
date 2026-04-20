# bonsai-almide arc plan

## Goal

Run Bonsai 1.7B / 4B / 8B end-to-end on Almide, matching or beating the llama.cpp reference implementation on native and adding a browser-deployable WASM path. The repo doubles as a public benchmark that Almide's AOT + egg fusion pipeline delivers on a real, non-trivial ML workload.

## Target spec (Bonsai-1.7B — confirmed from GGUF metadata)

- Format: `Q1_0` GGUF, 248 MB (GGUF v3, 310 tensors, 32 metadata entries)
- Architecture: `qwen3` (declared in metadata)
- Layers: 28 decoder blocks
- Hidden: 2048, FFN: 6144
- Attention: GQA — 16 query heads, 8 KV heads, head_dim 128 (key_length = value_length = 128)
- **Per-block Q/K norm**: `attn_q_norm` and `attn_k_norm` per layer (Qwen3-specific; not in standard Llama)
- RoPE: **YaRN scaling** — original_context 8192, scaling factor 4.0, extended context 32768, freq_base 1,000,000
- RMSNorm epsilon: ~1e-6 (9.999999974752427e-07)
- Tokenizer model: `gpt2` BPE with `qwen2` pre-tokenizer; vocab 151,669 tokens, merges 151,387
- EOS: 151645, PAD: 151643, no BOS
- Chat template: Qwen3 ChatML with tool-use support (embedded)
- LM head: **tied to `token_embd.weight`** (no separate `output.weight` tensor)
- Q1_0 dtype = **41** (GGML custom type for this fork). 197 of 310 tensors are Q1_0; the remaining 113 are F32 norm scales.

## Tensor inventory (per block)

Each of the 28 blocks has 11 tensors:
- `attn_norm.weight` [2048] F32
- `attn_q.weight` [2048, 2048] Q1_0
- `attn_q_norm.weight` [128] F32
- `attn_k.weight` [2048, 1024] Q1_0
- `attn_k_norm.weight` [128] F32
- `attn_v.weight` [2048, 1024] Q1_0
- `attn_output.weight` [2048, 2048] Q1_0
- `ffn_norm.weight` [2048] F32
- `ffn_gate.weight` [2048, 6144] Q1_0
- `ffn_up.weight` [2048, 6144] Q1_0
- `ffn_down.weight` [6144, 2048] Q1_0

Top-level: `token_embd.weight` [2048, 151669] Q1_0, `output_norm.weight` [2048] F32.

## Q1_0 packing format

- Group size: 128 weights per group
- Per group: 128 sign bits (16 bytes) + 1 FP16 scale (2 bytes) = 18 bytes for 128 weights
- Sign bit mapping: `0 → -scale`, `1 → +scale`
- Effective: 1.125 bits/weight
- Reference kernel: [Mintplex-Labs/prism-ml-llama.cpp](https://github.com/Mintplex-Labs/prism-ml-llama.cpp)

## Reference performance (to beat)

| Platform | Backend | TG128 tok/s | Speedup over FP16 |
|---|---|---|---|
| M4 Pro 48GB | llama.cpp Metal | 250 | 3.8x |
| RTX 4090 | llama.cpp CUDA | 674 | 3.0x |

## Milestones

### P0 — weights load + single forward step (native CLI)
- Download Bonsai 1.7B from Hugging Face
- Inspect safetensors layout, document tensor names / shapes / dtypes
- Load weights into Almide Matrix representation
- Run one forward step end-to-end, compare logits against HF reference (numerical match within tolerance)

### P1 — greedy generation + tok/s baseline (native)
- Implement KV cache
- Greedy sampling loop, 100-token generation
- Report tok/s on M-series Mac native
- Acceptance: coherent English output, tok/s measured

### P2 — WASM build + Node demo
- Compile bonsai pipeline to WASM via `almide build --target wasm`
- Run in Node.js with weights loaded from disk
- Report WASM tok/s

### P3 — browser demo + benchmark page
- Single HTML page: load weights, stream tokens in browser
- Benchmark table: bonsai-almide vs bonsai.cpp vs transformers.js
- Publish to GitHub Pages

### P4 — WebGPU backend (stretch)
- Add WebGPU kernels for 1-bit matmul
- Target 200 tok/s in browser

## Architecture diff vs existing Almide llama_block

| Concern | llama_block (existing) | Bonsai / Qwen3 (new) |
|---|---|---|
| Attention | MHA (`masked_multi_head_attention`) | **GQA 16/8** — new matrix intrinsic |
| Q/K norm | none | **per-block RMSNorm on Q and K** before RoPE (Qwen3-specific) |
| RoPE | standard | **YaRN scaling** (factor 4.0, orig 8192 → 32768) |
| Weights dtype | Float32 dense | **Q1_0 packed** (sign bits + fp16 scales/group 128) — new kernel |
| Linear | `linear_row_no_bias` (dense fp32) | `linear_q1_0_row` — new |
| Tokenizer | n/a | GPT-2 BPE, 151k vocab + qwen2 pre-tokenizer — load from GGUF metadata |
| Model structure | single block | 28 layers + KV cache + tied embed/lm_head |
| Weight loader | nn has GGUF F16/F32 | extend `nn/gguf.almd` with dtype 41 (Q1_0) decode |

## Technical risks

- **Q1_0 matmul kernel**: for each output row, accumulate per group of 128 activations. `sum_bit1 - sum_bit0` multiplied by `fp16 scale`. AVX-512 / NEON have fast popcount + mask paths; llama.cpp fork is the reference. For WASM SIMD128 we get `i8x16`, `v128.and`, `v128.bitselect` — workable but likely needs careful unrolling.
- **Weight loader**: Almide `nn` has GGML F16/F32 block decode, not Q1_0. Extend `ggml_whisper` (or fork into `ggml_bonsai`) to add Q1_0 block parsing.
- **Tokenizer**: Qwen3 BPE + chat template, embedded in GGUF metadata. Parse from GGUF rather than re-DL `tokenizer.json`. Almide has no tokenizer stdlib — port or inline.
- **Precision**: 1-bit quantization is fragile. Any fusion rewrite that re-orders ops can cause logit divergence — egg rewrites need an `@exact_numerics` gate or a numerical-tolerance verifier.
- **GQA fusion**: existing `masked_multi_head_attention` is MHA-only. Need a GQA variant that shares K/V across query groups (16 Q × 8 KV = 2 queries per KV head) without copying.
- **Tied embedding / lm_head**: Qwen3 often ties embed and lm_head; check whether Bonsai keeps them tied under Q1_0 or stores two separate 1-bit matrices.

## Deliverables

- `src/` — Bonsai model in Almide (tokenizer, model, generation)
- `examples/` — CLI demo, bench, browser demo
- `spec/` — correctness tests (logits match HF reference)
- `docs/` — this arc, benchmarks, architecture notes
- Upstream PRs to `almide/` and `almide/nn` for missing primitives
