# bonsai-almide 高速化 恒久ロードマップ

2026-04-22 起草。browser の Bonsai-1.7B-Q1_0 chat 体験を reference
(webml-community/bonsai-webgpu, 51 tok/s WebGPU) と並ぶ、さらに越える水準に
引き上げるための中長期計画。

## 現状ベースライン

| 指標 | 数値 | 備考 |
|---|---|---|
| browser WASM | 1.5 s/tok = 0.67 tok/s | scalar f64、KV streaming + bytes shuttling |
| native release (M1) | 1.38 ms/tok = 724 tok/s | NEON Q1_0 kernel + super-intrinsic |
| reference (WebGPU, webml-community) | 51 tok/s | transformers.js + ONNX Q1 + WebGPU compute |
| llama.cpp Metal Q4_K_M | 60.9 tok/s | apples-to-oranges (GPU) |
| **browser gap** | **76x** | これを closed まで持っていく |

### ホットスポット
1 step = 28 layer × layer forward、各 layer で `linear_q1_0_row_no_bias` を 6 回
(Q/K/V/output/gate/up/down)。profile: **linear_q1_0 self-time 86%**。Matmul を
速くすれば全体が速くなる。

## ギャップ分解

76x を倍率で分解:

| 要因 | 倍率 | 備考 |
|---|---|---|
| WASM SIMD128 化 | 3-4x | 現状 scalar f64 → f32×4 or i32×4 |
| activation int8 (BitNet式) | 2-4x | 帯域 8x 削減、i32 積和 |
| GPU 化 (WebGPU compute) | 10-15x | compute shader で parallel matmul |
| shape specialization / fusion | 1.5-2x | egg 宣言的 rewrite |

積の上限は 120-360x、ギャップ 76x は理論的には余裕。ただし **CPU だけの上限は 15-20 tok/s**、
50+ tok/s は **WebGPU 必須**。

## 戦略軸 (5 本)

### 軸 A: WASM CPU 詰め (短期、独立完結)
既存の hand-rolled emit_wasm を現状のまま、ホット path を SIMD 化して手数で稼ぐ。

- **A1. WASM SIMD128 Q1_0 kernel**: Q1_0 の inner loop (bit→±scale×128 乗算) を
  v128 packed f32 4-lane に書き直し。popcount + LUT で ternary 特化も検討。
- **A2. bytes shuttling 撤廃**: `__alloc_persist` / `__alloc_scratch` の
  2-region allocator で KV を persistent region に、scratch だけ restore。
  今の List[Bytes] round-trip を消して copy コスト削減。
- **A3. codegen papercut 解消**: `list.sort` on Float の WASM ICE (今回の
  sampling で `top_k_threshold` を自前 O(n*k) で書く羽目になった箇所) を正式対応。
  sampling overhead を本来の 1 ms レベルに戻す。
- **A4. GGUFFile parse cache**: `predict_step_kv_bytes` が毎 step 248MB を
  re-parse してる。JS 側で persistent `gguf_file_ptr` を保持、parse 1 回で済ませる。

**期待**: 2-4x → browser 3-6 tok/s。**完了定義**: 5 tok/s 突破。

### 軸 B: 量子化パス (中期、アーキに踏み込む)
Matrix が全部 f64 なのは歴史的経緯。f32 / int8 / ternary popcount まで降ろして
**帯域と計算量を同時に削る**。

- **B1. Matrix[T] arc 合流** (既存 arc): `Matrix[Float32]` を type-parametric 化、
  matmul の storage と計算を f32 に。f64→f32 で帯域 2x、計算 1.5x。
  project_matrix_dtype_design.md 系列と合流。
- **B2. activation int8 quantization**: per-token absmax で int8 化、
  Q1_0×int8 の i32 積和 matmul を実装 (bitnet.cpp I2_S path 参考)。
  `linear_q1_0_int8_row_no_bias` として新規 intrinsic 追加。
- **B3. ternary popcount math**: Q1_0 の sign bit と int8 activation を
  両方 ternary にできれば、matmul は `popcount(xor) × 2 - 128` で書ける。
  精度劣化とのトレードオフ、A/B 比較が必要。

**期待**: 追加 2-4x → browser 10-20 tok/s。**完了定義**: 15 tok/s 突破。

### 軸 C: WebGPU backend (長期、本丸)
CPU の物理天井 (15-20 tok/s) を超えて reference の 51 tok/s と並ぶ・越えるには、
GPU compute shader への dispatch が必須。

- **C1. runtime/wgsl/ 新設**: Q1_0 matmul、RoPE、RMS norm、softmax の WGSL 実装。
  先にこれを手で書いて性能の上限を把握する (1 weekend)。
- **C2. JS bridge**: `WebGPU.Device` / `GPUBuffer` を Almide WASM の matrix
  primitive の JS 側で intercept、ホット path だけ GPU dispatch に振り分け。
  cold path は WASM 続行。
- **C3. AOT codegen target `webgpu-hybrid`**: `emit_wasm` + WGSL コード生成を
  単一のバックエンドで管理。matrix.linear_q1_0_row_no_bias みたいな
  intrinsic は GPU 呼び出し、それ以外は WASM。

**期待**: 追加 10-20x → browser 100-300 tok/s。**完了定義**: 50 tok/s 超 (reference 超え)。

### 軸 D: Compiler 世界最高化 (長期、170万/月 billing の本丸)
Almide を「LLM コンパイル専用言語」ではなく、**宣言的に書けば勝手に最速 kernel を
導出する compiler**に押し上げる arc。Bonsai はその証明実験台。

- **D1. egg declarative rewrite arc** (既存 MLIR Stdlib Unification arc 合流):
  fusion rule を declarative に書き、saturation で最適な kernel chain を自動導出。
  `matrix-fusion` pattern の `@rewrite` attribute arc を拡張、より広範な rewrite
  (matmul fusion、attention flash 化、Q1_0 specific) へ。
- **D2. shape specialization pass**: Bonsai の hidden=2048, head_dim=128,
  n_layers=28 を AOT で const propagate、loop unroll + block tile を形状固定で展開。
  Runtime 分岐ゼロ化。
- **D3. ternary math dialect**: compiler が 1-bit weights を検出したら popcount 化を
  自動選択する高位表現。BitNet 移植を将来やるならここに乗る。

**期待**: 追加 1.5-3x、かつ **Almide 言語の "世界最高 compiler" 看板**。
**完了定義**: LLM dev が Bonsai を Almide で書き直して "手で書くより速い" と
言わせる (定性、だが MSR benchmark で測れる)。

### 軸 E: LLVM/MLIR backend (構造的、D と合流)
現状 `emit_wasm` と `emit_rust` (template) の 2 系統で生成してるせいで、今回の
`mha_core` mask バグみたいな**並行実装取りこぼし**が構造的に起こる。LLVM/MLIR に
寄せて `native/wasm/GPU` を統一バックエンドから生成する。

- **E1. MLIR dialect for Almide IR**: 既存の Almide IR を MLIR dialect として
  表現、lowering pass chain (Almide dialect → MLIR std → LLVM IR) を組む。
- **E2. LLVM WASM target に切替**: hand-rolled `emit_wasm` を順次廃止、
  `wasm32-unknown-unknown` target で LLVM に生成させる。副産物として
  auto-vectorize / LICM / inline が無料、素で 1.3-2x 速くなる見込み。
- **E3. MLIR → SPIR-V で軸 C を自動導出**: WebGPU shader を手書きする軸 C の
  C1/C3 を MLIR の SPIR-V dialect lowering で自動化。hand-written WGSL が要らなくなる。
- **E4. Runtime 統一**: native 用 Rust runtime と WASM 用 inlined runtime を、
  C (or Rust stable ABI) で 1 set に統一、LLVM が両 target に落とす。

**期待**: 単独の倍率ゲインは小さい (1.3-2x) が、**軸 B/C の実装コストが激減**
(LLVM/MLIR の lowering が自動でやる)、かつ mask バグみたいな構造的リスクを消す。

**完了定義**: `emit_wasm/*.rs` を全廃、cargo test + almide test + bonsai-almide
bench 全 green で LLVM WASM output が hand-rolled と同等以上の速度。

### 軸 E と WASM deploy の両立
LLVM を **AOT 専用**で使う限り、bonsai-almide.wasm の browser 動作に弊害ない
(LLVM は build time にだけ動く、runtime には同梱しない)。ランタイム JIT 同梱だけは
browser deploy 破綻するので禁止。

## 順序の推奨

現実主義: **A → B → C → D/E**。各段階で benchmark で止められる。
理想主義: **E を先に**、A-D は E の成果として自動導出を狙う。

### 提案: 並走 (期間 3-6 ヶ月)
```
Month 1-2:  A 全部 (SIMD + 2-region + codegen papercuts + parse cache) → 5 tok/s
Month 2-3:  B1 (f32) + B2 (int8 activation) → 15 tok/s
Month 3-4:  E1-E2 (MLIR dialect + LLVM WASM 切替)
Month 4-6:  E3 + C (WebGPU 自動導出) → 50-100 tok/s
Month 4-6:  D1-D3 並行 (compiler 世界最高 arc)
```

並走の理由:
- A は短期の目に見える改善、ユーザー体験に直結
- D/E は長期、Almide 言語の価値証明 (170万/月 billing 正当化) に不可欠
- B は A の次の自然な延長、だが E 完了後は fusion pass で一部自動化される

## 非目標

- **batching / multi-user**: browser 1 session only
- **speculative decoding**: 1.7B モデルには draft model を別途用意するコスト割に合わない
- **WebNN / ONNX runtime 組み込み**: 既存フレームワーク依存は「世界最高 compiler」
  ブランドと逆行
- **50 tok/s を CPU WASM だけで**: 物理限界、投資対効果悪い → WebGPU へ
- **LLVM runtime 同梱 (JIT)**: browser deploy 破綻、却下

## 成功基準 (phase ゲート)

| phase | tok/s 目標 | 測定: bonsai-almide browser、prompt "What is the capital of Japan?" |
|---|---|---|
| 現状 | 0.67 | baseline |
| phase 1 (A 完了) | 5+ | WASM SIMD + 2-region |
| phase 2 (B 完了) | 15+ | f32 + int8 activation |
| phase 3 (E1-2 完了) | 10+ (A-B 相当で OK) | LLVM WASM 切替、構造的負債解消 |
| phase 4 (C 完了) | 50+ | WebGPU、reference parity |
| phase 5 (D 完了) | 75+ | declarative fusion、reference 超え |

## リスク

- **LLVM 依存で compiler binary が肥大**: 現 ~20MB → 150MB。distribution / CI 負荷増。
  mitigation: 開発者向けは LLVM 版、配布は strip + cargo dist で。
- **MLIR dialect 設計でミスると E 全部が止まる**: prototype 先行、dog-food で仕様固めてから本実装。
- **WebGPU は browser 対応率が 2026 時点で ~60%**: Safari は追いついてきたが全員じゃない。
  fallback として WASM path も保持 (軸 A/B の遺産で十分)。
- **bonsai-almide 以外で Almide を使うユーザーがいない**: Bonsai でしか価値証明できないと、
  軸 D の一般性主張が弱い → **bitnet-almide や whisper-almide など並行デモ**を持つのが
  healthy。今日の session で bitnet は凍結したが、phase 3-4 あたりで再始動したい。

## 運用

- 月次で phase ゲートの tok/s を測定、ROADMAP 更新
- 各軸に PR arc を 1 本ずつ切り、進捗を `project_bonsai_llama_parity.md` memory に
  追記
- 軸 E の MLIR dialect 設計は別途 `docs/roadmap/active/mlir-backend.md` で
  Almide 本体側に分岐

---

**一行サマリー**: 短期は WASM SIMD で 5 tok/s、中期は int8 で 15 tok/s、長期は
LLVM/MLIR backend 経由で WebGPU を自動導出して 50-100 tok/s。Bonsai は Almide
「世界最高 compiler」路線の証明実験台、その価値で 170万/月 billing の看板を張る。
