# bonsai-almide

**通过 Almide 编译 Bonsai 1-bit LLM，在浏览器中作为聊天演示运行。**

其他语言: [English](README.md) · [日本語](README.ja.md)

## 这是什么

[Bonsai](https://huggingface.co/prism-ml) 是 PrismML 从零训练的 1-bit LLM
(Qwen3-1.7B 架构、28 层)。Q1_0 格式: 每 128 个权重共享一个 FP16 scale,
外加 1 bit 的符号位，有效 **1.125 bit/权重**、1.7B 模型 **248 MB**。
与多数 1-bit 发布不同, 这是**从头以 1-bit 为前提训练**的, 不是事后量化。

本仓库通过 [Almide](https://github.com/almide/almide) (nanopass 式 AOT 编译语言、
支持 Rust 和 WASM 双目标) 编译 Bonsai, 并以浏览器聊天的形式运行。既是 Almide
语言的实用证明, 也是一个罕见的**纯 CPU / 纯 WASM** 运行 1-bit LLM 的示例。

## 现状 (2026-04-22)

**浏览器 chat 可连贯运行**: https://almide.github.io/bonsai-almide/

- Qwen3 chat template + KV cache streaming + sampling
  (temperature / top-k / repetition penalty)
- 遇到 `<|im_end|>` 自动停止
- 示例: `"What is the capital of Japan?"` →
  `"The capitals of Japan include Tokyo, Kyoto..."`
- WASM 产物 32 KB、模型 248 MB (首次运行后缓存到 IndexedDB)

### 性能数据

| 后端 | 实测 | 参考值 |
|---|---|---|
| native M1 (Almide + NEON + super-intrinsic + KV) | 1.38 s/tok (0.725 tok/s) | — |
| 浏览器 WASM (scalar f64) | 1.5 s/tok (0.67 tok/s) | [webml-community/bonsai-webgpu](https://huggingface.co/spaces/webml-community/bonsai-webgpu) (WebGPU): **51 tok/s** |
| llama.cpp Metal Q4_K_M | — | 60.9 tok/s |

目前**浏览器相比 WebGPU 参考实现慢 76 倍**, 这是 scalar f64 CPU 路径的自然结果。
缩小差距的计划 (SIMD → activation int8 → WebGPU 后端, 各自独立的 arc) 见
[docs/PERF_ROADMAP.md](docs/PERF_ROADMAP.md)。

## 模型

- `prism-ml/Bonsai-1.7B-gguf` — `Bonsai-1.7B-Q1_0.gguf` (248 MB)
- 28 层、GQA (16 Q / 8 KV 头)、head_dim 128、SwiGLU、RoPE (θ=1,000,000)、
  RMSNorm
- 词表 151,936 (Qwen3 BPE)
- 浏览器侧通过 `@huggingface/transformers` 从 `Qwen/Qwen3-1.7B` 加载 tokenizer

## 本地运行

### WASM + node bench (推荐)
```bash
# 首次: 把模型下载到 weights/
#   huggingface-cli download prism-ml/Bonsai-1.7B-gguf --local-dir weights

npm install  # tokenizer 和 decode 用的 @huggingface/transformers

almide build --fast --target wasm \
  examples/bonsai_wasm_entry.almd -o docs/bonsai.wasm

# Greedy (应与 bench_verify_kv 的 path 1 / path 4 一致)
node bench/bench_wasm_kv_stream.mjs

# Chat-template 模式 (浏览器采用的方式)
CHAT=1 TEMP=0.8 TOP_K=40 PENALTY=1.3 DECODE=1 \
  node bench/bench_wasm_kv_stream.mjs
```

### Native bench
```bash
almide build bench/bench_native_kv_super.almd -o bench/bench_super
./bench/bench_super
```

### 浏览器
通过 HTTP 伺服 `docs/`:
```bash
python3 -m http.server -d docs 8000
# http://localhost:8000
```
或使用已部署版本 https://almide.github.io/bonsai-almide/。

## 状态

- **已完成**: KV cache streaming、sampling (temperature + top-k + repetition
  penalty)、Qwen3 chat template、基于 EOS 的流式 UI
- **下一步 (短期)**: WASM SIMD128 Q1_0 kernel、撤掉 bytes shuttling 用的 2-region
  allocator、`list.sort` on Float 的 WASM codegen 修复 (PERF_ROADMAP 轴 A 与 D 的一部分)
- **长期**: 通过 LLVM/MLIR 后端自动 lowering 到 WebGPU compute shader (轴 C / D / E)

## Arc 文档

- [docs/ARC.md](docs/ARC.md) — Bonsai 浏览器 demo 原始 arc 计划
- [docs/PERF_ROADMAP.md](docs/PERF_ROADMAP.md) — 5 轴永久提速计划
- [docs/WASM_ROADMAP.md](docs/WASM_ROADMAP.md) — WASM 移植早期笔记

## 许可

Apache-2.0 (与 Bonsai 上游一致)。
