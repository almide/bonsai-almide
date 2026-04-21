# bonsai-almide

**Bonsai 1-bit LLM compiled through Almide, running as a browser chat demo.**

Other languages: [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

## What this is

[Bonsai](https://huggingface.co/prism-ml) is a 1-bit LLM trained from scratch by
PrismML (Qwen3-1.7B architecture, 28 layers). Q1_0 format: one FP16 scale shared
across every 128 weights plus a sign bit per weight, yielding an effective
**1.125 bit/weight** and a **248 MB** checkpoint for a 1.7B model. Unlike most
1-bit releases this is a ground-up 1-bit training run, not a post-hoc quant.

This repo builds Bonsai through [Almide](https://github.com/almide/almide) — a
nanopass AOT-compiled language with Rust and WASM codegen targets — and runs it
as a browser chat. It's a real-use proof point for the Almide language and a
rare example of a 1-bit LLM running on pure CPU / pure WASM.

## Where it's at (2026-04-22)

**Coherent browser chat is live**: https://almide.github.io/bonsai-almide/

- Qwen3 chat template + KV cache streaming + sampling (temperature / top-k /
  repetition penalty)
- Stops on `<|im_end|>`
- Example: `"What is the capital of Japan?"` →
  `"The capitals of Japan include Tokyo, Kyoto..."`
- WASM artifact is 32 KB; model is 248 MB (cached in IndexedDB on first run)

### Numbers

| Backend | Measured | Reference |
|---|---|---|
| native M1 (Almide + NEON + super-intrinsic + KV) | 1.38 s/tok (0.725 tok/s) | — |
| browser WASM (scalar f64) | 1.5 s/tok (0.67 tok/s) | [webml-community/bonsai-webgpu](https://huggingface.co/spaces/webml-community/bonsai-webgpu) (WebGPU): **51 tok/s** |
| llama.cpp Metal Q4_K_M | — | 60.9 tok/s |

We are **76x slower than the WebGPU reference in the browser** at the time of
writing. That's the natural place to be for a scalar-f64 CPU path. The plan to
close the gap (SIMD → activation int8 → WebGPU backend, each a separate arc)
lives in [docs/PERF_ROADMAP.md](docs/PERF_ROADMAP.md).

## Model

- `prism-ml/Bonsai-1.7B-gguf` — `Bonsai-1.7B-Q1_0.gguf` (248 MB)
- 28 layers, GQA (16 Q / 8 KV heads), head_dim 128, SwiGLU, RoPE (θ=1,000,000),
  RMSNorm
- Vocab 151,936 (Qwen3 BPE)
- Tokenizer is fetched in the browser via `@huggingface/transformers` from
  `Qwen/Qwen3-1.7B`

## Running it locally

### WASM + node bench (recommended)
```bash
# One-time: download the model into weights/
#   huggingface-cli download prism-ml/Bonsai-1.7B-gguf --local-dir weights

npm install  # pulls @huggingface/transformers for the tokenizer + decode

almide build --fast --target wasm \
  examples/bonsai_wasm_entry.almd -o docs/bonsai.wasm

# Greedy (matches path 1 / path 4 in bench_verify_kv)
node bench/bench_wasm_kv_stream.mjs

# Chat-template mode (what the browser uses)
CHAT=1 TEMP=0.8 TOP_K=40 PENALTY=1.3 DECODE=1 \
  node bench/bench_wasm_kv_stream.mjs
```

### Native bench
```bash
almide build bench/bench_native_kv_super.almd -o bench/bench_super
./bench/bench_super
```

### Browser
Serve `docs/` over HTTP and open `index.html`:
```bash
python3 -m http.server -d docs 8000
# http://localhost:8000
```
Or just use the deployed copy at https://almide.github.io/bonsai-almide/.

## Status

- **Landed**: KV cache streaming, sampling (temperature + top-k + repetition
  penalty), Qwen3 chat template, EOS-aware streaming UI.
- **Next (short term)**: WASM SIMD128 Q1_0 kernel, 2-region allocator to drop
  the bytes-shuttling round-trip, `list.sort` on Float WASM codegen fix.
  (PERF_ROADMAP axes A and part of D.)
- **Long term**: LLVM/MLIR backend → WebGPU compute shader auto-lowering.
  (Axes C, D, E.)

## Arc documents

- [docs/ARC.md](docs/ARC.md) — original Bonsai browser demo plan
- [docs/PERF_ROADMAP.md](docs/PERF_ROADMAP.md) — the permanent 5-axis speedup plan
- [docs/WASM_ROADMAP.md](docs/WASM_ROADMAP.md) — earlier notes from the WASM port

## License

Apache-2.0, matching Bonsai upstream.
