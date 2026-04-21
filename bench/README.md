# Bench — Bonsai-1.7B Almide vs llama.cpp

Goal: llama.cpp 並の tok/s を Almide から狙うベースラインを作る。

Prompt: `"The capital of Japan is"` → token ids `[785, 6722, 315, 6323, 374]` (len 5).
各 target で prompt_eval + 7 token 追加 gen を計測 (合計 8 token)。

## Numbers (M1 Mac, 2026-04-21)

| target                                        | pp5 (ms) | tg (ms/tok) | tg tok/s | vs llama.cpp Metal |
| --------------------------------------------- | -------: | ----------: | -------: | -----------------: |
| Almide native `--release`                     |   18,526 |      24,013 |    0.042 |          1/1,452x  |
| Almide native `--release` + **KV**            |   17,799 |      14,382 |    0.070 |          1/870x    |
| Almide native `--release` + KV + **SIMD**     |   17,037 |      13,469 |    0.074 |          1/820x    |
| Almide native `--release` + KV + SIMD + **super** | 12,737 |    8,308 |    0.120 |          1/508x    |
| Almide native + **no-file-clone** (profile-driven) |  6,324 |  **1,380** |  **0.725** |          **1/84x** |
| Almide WASM (node)                            |    8,298 |      10,973 |    0.091 |          1/670x    |
| Almide WASM (node) + **KV**                   |    8,170 |       1,595 |    0.627 |          1/97x     |
| llama.cpp Q4_K_M, CPU (Accel)                 |       64 |        23.9 |     41.9 |          1/1.5x    |
| llama.cpp Q4_K_M, Metal                       |       51 |        16.4 |     60.9 |          1x        |

KV-cache delta:
- **native 1.7x** tg speedup (24.0 → 14.4 s/tok)
- **WASM 6.9x** tg speedup (11.0 → 1.6 s/tok); first gen step = 1.89 s, second = 1.33 s
- token 26194 (" Tokyo") still emitted first on every target.

WASM KV caveat: the current `predict_tokens_kv` accumulates scratch allocations for the running K/V matrices in Almide's bump allocator, so the 4 GB address space runs out at n_new ≈ 4. Above numbers are from `N_NEW=3`. Fixing peak memory (heap reclaim around the per-layer `append_rows`) is a separate task from the primitive work; the tg/tok number is already well past the closed-book "gen with cache" regime where llama.cpp's curve also lives.

注:
- Bonsai は Q1_0 (1.125 bit/weight, 248 MB), llama.cpp は Q4_K_M (4.5 bit, 1.2 GB) — 同モデル/同量子化で直接比較できる GGUF が未公開のため。
- Almide は KV cache 無しなので token 毎に全 28-layer forward を再実行 → context 長が増えると `tg` が伸びる。
- Almide native が WASM より遅いのは Q1_0 decode/matmul が scalar f64 loop で動いているため — WASM 側は LICM + range-iter 最適化が効いている。

## Repro

```bash
# native
almide run --release bench/bench_native.almd              # baseline (no KV)
almide run --release bench/bench_native_kv.almd           # + KV cache
almide run --release bench/bench_native_kv_super.almd     # + super-intrinsic (今の best)

# WASM
node bench/bench_wasm.mjs                    # baseline
N_NEW=3 node bench/bench_wasm_kv.mjs         # KV (n_new≥4 で OOM — task #18)

# llama.cpp
llama-bench -m weights/Qwen_Qwen3-1.7B-Q4_K_M.gguf -p 5 -n 8 -r 3        # Metal
llama-bench -m weights/Qwen_Qwen3-1.7B-Q4_K_M.gguf -p 5 -n 8 -r 3 -ngl 0 # CPU
```

重み:
- `weights/Bonsai-1.7B-Q1_0.gguf` (bonsai, auto-DL 済)
- `weights/Qwen_Qwen3-1.7B-Q4_K_M.gguf` — `hf download bartowski/Qwen_Qwen3-1.7B-GGUF Qwen_Qwen3-1.7B-Q4_K_M.gguf --local-dir weights`

## 完了した最適化 (2026-04-21)

| arc | PR | native tg | 寄与 |
| --- | -- | --------: | --- |
| ベースライン (no KV)                | —          | 24,013 ms | — |
| KV cache primitives              | almide #222 | 14,382 ms | 1.67x — KV + sq!=sk mask + rope_rotate_at + append_rows |
| Q1_0 NEON kernel + SIGN_LUT      | almide #223 | 13,469 ms | 1.07x — 当初予想 10-20x、実測 small、bottleneck が matmul じゃなかった判明 |
| qwen3_block_q1_0_kv super        | almide #224 |  8,308 ms | 1.63x — 1 layer を 1 runtime fn に統合、Matrix ownership churn 解消 |
| profile-driven file.clone 回避 + burn NEON | (bench) + almide #225 |  1,380 ms | **6.02x** — samply で `<GGUFFile as Clone>::clone` が inclusive 72% と判明。bench 側で `let raw = file.raw` + pure helper `tensor_offset(tensors, base, name)` で 248 MB Bytes clone を消す。同時に burn runtime の Q1_0 kernel も NEON + Cow borrow + single-row `is_small` 拡張 |

総合 **17.4x** (gap 1,452x → 84x)。場当たりじゃなく `samply` → bottleneck 特定 → 1 改修で 6x が出た — 次の PR を当てずっぽうで書かずに profile する価値の実例。

## Roadmap — 次の候補

次の profile (1.38 s/tok, 2026-04-21 22:10) ではこう見えている:
- `almide_rt_matrix_linear_q1_0_row_no_bias` — **86% self** ≈ 1.18 s. 実効 ~670 MFLOPS (M1 NEON peak 8 GFLOPS の 8%). この 1 関数が次の boss.
- `<GGUFFile as Clone>::clone` — 残り 5.5% inclusive ≈ 76 ms. bench-side workaround で削ったが、Almide codegen で borrow-inference を強化すれば workaround 不要に.

| 候補 | 見込み倍率 | 作業コスト | 備考 |
| ---- | --------: | --------: | ---- |
| **linear_q1_0 f32 path** (Matrix[T] arc 合流) | 2–4x | 中 | activations を f32 保持 → NEON 4-lane f32 SIMD. accuracy は Q1_0 にとって誤差. memory: project_matrix_dtype_design.md P5 と合流 |
| **popcount trick** for Q1_0 block dot | 1.5–2x | 小 | sum_signed = 2·masked_sum − all_sum の恒等式で sign expansion コスト削減. per-block 先に all_sum 計算 |
| **BorrowInference 強化 (codegen fix)** | bench 6% / 全言語恒久 | 中 | `file: GGUFFile` read-only callee を `&GGUFFile` 化. bench workaround 不要、他 user の罠減る. 「理想形」的 PR |
| **WASM `qwen3_block_q1_0_kv` dispatch**   | WASM で 1.5–2x  | 大 (200+ 行 emit_wasm) | WASM 既に KV で 1.6 s/tok, 相対効果小。primitive 統一感・次arc に続く基盤 |
| **WASM file.clone 回避 + v128 q1_0**     | WASM 2–3x | 中 | native で効いた同パターンを WASM に. v128 SIMD は 4-lane f64 |
| **WASM KV heap reclaim** (task #18)       | n_new 制限解除 | 中 | scratch を `__heap_restore` 境界で reclaim、4 GB cap 超えられる |
| **egg declarative rewrite** 化            | ~等価 + 長期 | 大 | super-intrinsic を rule table に昇格 → MLIR arc と合流 |
| **WASM SIMD128 + threads**                | 2-4x | 大 | SharedArrayBuffer + Atomics、browser compat 要検討 |
| **WebGPU**                                | 5-20x (stretch) | 特大 | 別 arc、llama.cpp Metal と同等を狙える |

### 優先の考え方

- **量りに効く順**: popcount trick → f32 path → WASM 同パターン適用 → WebGPU
- **理想形順** (MLIR/egg arc と整合): BorrowInference 強化 → egg rewrite → 上のどれか
- **short arc 1 day**: popcount trick (small PR) — Q1_0 block dot の次の 1 段
- **long arc**: egg rewrite は MLIR arc の主戦場

llama.cpp Q4_K_M との絶対値比較は元々不公平 (Bonsai は 1.125 bit vs 4.5 bit)、Q1_0 primitive だからこそ Almide の武器。最終目標は Bonsai/WASM/browser で "動く + sub-second/tok" の到達。現状 native sub-2s、WASM sub-2s、あと 1-2 段で llama.cpp CPU (42 tok/s) ラインに届く射程。
