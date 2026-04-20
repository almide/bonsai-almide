# bonsai-almide

**Bonsai 1-bit LLM inference in Almide.** AOT compiled. Native + WASM. No GPU required.

## Why this exists

[Bonsai](https://huggingface.co/prism-ml) is a 1-bit quantized LLM trained from scratch by PrismML. Q1_0 format: every weight is a single sign bit, with one FP16 scale shared across groups of 128 weights (**1.125 effective bits/weight**). 1.7B model fits in **248 MB**, runs 3.8x faster than FP16 on M-series Metal via a specialized llama.cpp fork.

This project runs Bonsai through [Almide](https://github.com/almide/almide), an AOT-compiled language with MLIR + egg-based fusion. The goal is to **match or beat the llama.cpp reference** on native, and to **deploy to the browser** without WebGPU as a baseline.

## Target benchmarks (1.7B, TG128)

| Backend            | Reference (llama.cpp)  | bonsai-almide target |
|--------------------|------------------------|----------------------|
| Native M-series    | 250 tok/s (Metal)      | **≥ 250 tok/s** (match), stretch 350 |
| Native x86 + AVX2  | ~FP16 reference × 3    | match                |
| WASM (browser)     | n/a                    | **60 tok/s**         |
| WebGPU (browser)   | n/a (stretch)          | **150 tok/s**        |
| Binary size        | 248 MB (model)         | 248 MB + **< 1 MB** runtime |

Numbers are targets. See `docs/ARC.md` for progress.

## Model

- `prism-ml/Bonsai-1.7B-gguf` — `Bonsai-1.7B-Q1_0.gguf` (248 MB)
- Architecture: **28 layers**, GQA (16 Q / 8 KV heads), SwiGLU, RoPE, RMSNorm
- Vocab 151,936, context 32,768, tokenizer embedded (Qwen3 BPE)
- Base: Qwen3-1.7B dense, re-trained from scratch as 1-bit (not post-hoc quantized)

## Status

Pre-P0 — repo scaffold. See [docs/ARC.md](docs/ARC.md) for the arc plan.

## Quick start (planned)

```bash
almide add bonsai@v0.1.0
```

```almide
import bonsai

effect fn main() -> Unit = {
  let model = bonsai.load("weights/bonsai-1.7b.safetensors")!
  let output = bonsai.generate(model, "The capital of Japan is", tokens: 20)
  println(output)
}
```

## License

Apache-2.0 (matching Bonsai upstream).
