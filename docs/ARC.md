# bonsai-almide arc plan

## Goal

Run Bonsai 1.7B / 4B / 8B end-to-end on Almide, matching or beating the llama.cpp reference implementation on native and adding a browser-deployable WASM path. The repo doubles as a public benchmark that Almide's AOT + egg fusion pipeline delivers on a real, non-trivial ML workload.

## Target spec (Bonsai-1.7B)

- Format: `Q1_0` GGUF, 248 MB
- Layers: 28 Transformer decoder blocks
- Attention: **GQA — 16 query heads, 8 KV heads**
- MLP: SwiGLU, Norm: RMSNorm, PE: RoPE
- Vocab: 151,936, Context: 32,768
- Base: Qwen3-1.7B dense, trained 1-bit from scratch
- Tokenizer: Qwen3 BPE, embedded in GGUF metadata
- Chat template: embedded in GGUF metadata
- Coverage: embeddings, attention, MLP, and LM head are all 1-bit

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

| Concern | llama_block (existing) | Bonsai (new) |
|---|---|---|
| Attention | MHA (`masked_multi_head_attention`) | **GQA 16/8** — new matrix intrinsic |
| Weights dtype | Float32 dense | **Q1_0 packed** (sign bits + fp16 scales/group 128) — new kernel |
| Linear | `linear_row_no_bias` (dense fp32) | `linear_q1_0_row` — new |
| Tokenizer | n/a | **Qwen3 BPE** — new module (embedded in GGUF metadata) |
| Model structure | single block | 28 layers + KV cache + tied embed + lm_head (all 1-bit) |
| Weight loader | nn has GGML F16/F32 | extend to GGUF Q1_0 block decode |

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
