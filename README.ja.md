# bonsai-almide

**Bonsai 1-bit LLM を Almide でコンパイルして、ブラウザで動く chat demo として提供。**

他言語: [English](README.md) · [简体中文](README.zh-CN.md)

## これは何

[Bonsai](https://huggingface.co/prism-ml) は PrismML が 1-bit でスクラッチ学習した
LLM (Qwen3-1.7B 相当、28 layer)。Q1_0 フォーマット: 重み 128 個に 1 つの FP16 scale
を共有 + 1 bit の sign で、実効 **1.125 bit/weight**、1.7B モデルで **248 MB**。
Post-hoc 量子化ではなく **1-bit 前提でのスクラッチ訓練**というのが特徴。

このリポジトリは Bonsai を [Almide](https://github.com/almide/almide) (nanopass 型
AOT コンパイル言語、Rust/WASM 2 target) でビルドして、browser 上の chat として
動かす。Almide 言語の実用証明実験台と、1-bit LLM を **pure CPU / pure WASM** で
動かす珍しいデモを兼ねる。

## 現状 (2026-04-22)

**browser で coherent な chat が動作中**: https://almide.github.io/bonsai-almide/

- Qwen3 chat template + KV cache streaming + sampling (temperature / top-k /
  repetition penalty)
- `<|im_end|>` で自動停止
- 例: `"What is the capital of Japan?"` →
  `"The capitals of Japan include Tokyo, Kyoto..."`
- WASM バイナリ 32 KB、モデル 248 MB (初回 IndexedDB キャッシュ)

### 性能数値

| Backend | 実測 | 参考 |
|---|---|---|
| native M1 (Almide + NEON + super-intrinsic + KV) | 1.38 s/tok (0.725 tok/s) | — |
| browser WASM (scalar f64) | 1.5 s/tok (0.67 tok/s) | [webml-community/bonsai-webgpu](https://huggingface.co/spaces/webml-community/bonsai-webgpu) (WebGPU): **51 tok/s** |
| llama.cpp Metal Q4_K_M | — | 60.9 tok/s |

現時点で **browser vs reference で 76x ギャップ**。スカラー f64 な CPU path の自然な
帰結で、SIMD → activation int8 → WebGPU の順で詰めていく計画は
[docs/PERF_ROADMAP.md](docs/PERF_ROADMAP.md) に整理済み。

## モデル

- `prism-ml/Bonsai-1.7B-gguf` — `Bonsai-1.7B-Q1_0.gguf` (248 MB)
- 28 layer、GQA (16 Q / 8 KV heads)、head_dim 128、SwiGLU、RoPE (θ=1,000,000)、
  RMSNorm
- Vocab 151,936 (Qwen3 BPE)
- tokenizer は browser 側で `@huggingface/transformers` 経由で `Qwen/Qwen3-1.7B`
  から取得

## ローカル実行

### WASM + node bench (推奨)
```bash
# 初回のみ: モデルを weights/ に配置
#   huggingface-cli download prism-ml/Bonsai-1.7B-gguf --local-dir weights

npm install  # tokenizer + decode 用 @huggingface/transformers

almide build --fast --target wasm \
  examples/bonsai_wasm_entry.almd -o docs/bonsai.wasm

# Greedy (bench_verify_kv の path 1 / path 4 と一致するはず)
node bench/bench_wasm_kv_stream.mjs

# Chat template モード (browser と同じ挙動)
CHAT=1 TEMP=0.8 TOP_K=40 PENALTY=1.3 DECODE=1 \
  node bench/bench_wasm_kv_stream.mjs
```

### Native bench
```bash
almide build bench/bench_native_kv_super.almd -o bench/bench_super
./bench/bench_super
```

### Browser
`docs/` を HTTP で serve:
```bash
python3 -m http.server -d docs 8000
# http://localhost:8000
```
またはデプロイ済の https://almide.github.io/bonsai-almide/。

## Status

- **Landed**: KV cache streaming、sampling (temperature + top-k + repetition
  penalty)、Qwen3 chat template、EOS 対応の streaming UI
- **Next (短期)**: WASM SIMD128 Q1_0 kernel、bytes shuttling 撤廃用の 2-region
  allocator、`list.sort` on Float の WASM codegen 修正 (PERF_ROADMAP 軸 A + D の一部)
- **Long term**: LLVM/MLIR backend 経由の WebGPU compute shader 自動導出 (軸 C/D/E)

## Arc ドキュメント

- [docs/ARC.md](docs/ARC.md) — Bonsai browser demo の元 arc 計画
- [docs/PERF_ROADMAP.md](docs/PERF_ROADMAP.md) — 高速化 5 軸の恒久計画
- [docs/WASM_ROADMAP.md](docs/WASM_ROADMAP.md) — WASM 移植初期メモ

## License

Apache-2.0 (Bonsai upstream に揃える)。
