# WASM target roadmap — L5 (browser demo)

L3 is green on the native Rust target (`almide run --release examples/probe_ref_match.almd`
matches llama.cpp byte-for-byte on "The capital of Japan is" → " Tokyo"). L5 requires that
same probe to compile to WebAssembly. The Almide WASM emitter is a hand-rolled backend that
dispatches each stdlib/matrix primitive directly, so every new primitive we added for
native-target perf needs a matching WASM dispatch arm.

## Done (2026-04-21)

- **`int.bits_to_f32`** → `i32_wrap_i64 → f32_reinterpret_i32 → f64_promote_f32`.
  Without this, every F32 tensor read in WASM silently underflows to ~1e-85 subnormals and
  the whole forward pass returns garbage.
- **`list.with_capacity(cap)`** → allocate the 4-byte len header, store 0, return the pointer.
  Discards `cap` because the WASM list layout `[len:i32][data...]` reallocates on every push.
  Semantically equivalent to `[]`; keeps WASM correct. (Native Rust target keeps the real
  pre-allocation benefit.)

## Pending — required before L5 is demo-usable

### 1. `matrix.from_q1_0_bytes(data: Bytes, offset: Int, rows: Int, cols: Int) -> Matrix`

**Cost** ≈ 1 day.

**Why it matters.** The Q1_0 block decoder is in the hot path. Reverting to the pure-Almide
implementation under WASM stacks the WASM codegen's ~2-3× native slowdown on top of the
~120× penalty we already measured for the pure-Almide decode — a cold 28-layer forward
that needs ~1.4 B weights decoded would run for hours in the browser and lose half of the
"L5 is demo-able" claim.

**Implementation sketch.**

```
register a native-like Rust helper `almide_rt_matrix_from_q1_0_bytes_wasm` in
emit_wasm/runtime.rs or rt_matrix.rs. Compile it directly as WASM.

The inner loop is:
  for b in 0..num_blocks {
      scale_raw = data[off + b*18] | data[off + b*18 + 1] << 8
      scale = fp16_to_f32(scale_raw)
      neg_scale = -scale
      bits_start = off + b*18 + 2
      for i in 0..128 {
          byte = data[bits_start + i/8]
          bit = (byte >> (i%7)) & 1
          flat[b*128 + i] = if bit { scale } else { neg_scale }
      }
  }

In WASM bytecode: scratch locals for scale/neg_scale/bits_start, i32_wrap_i64 on loop
vars, memory-access on `data.ptr + offset`, final Matrix wrap via `mk(rows, cols, flat)`
analogue (WASM matrix layout is `[rows:i32][cols:i32][data:f64×rows×cols]`).
```

The fp16 reconstruction logic is in `runtime/rs/burn/matrix_burn.rs::fp16_bits_to_f32` —
port it to WASM as an inline helper or emit it as a private Almide fn.

### 2. `matrix.rope_rotate(x: Matrix, n_heads: Int, head_dim: Int, theta_base: Float) -> Matrix`

**Cost** ≈ 1 day.

**Why it matters.** The pure-Almide RoPE we had before the intrinsic paid a per-row
per-head closure capture cost that ballooned layer-0 forward to ~60 s. In the browser that
becomes ~150 s/layer = >1 h per forward. Every Qwen3-family model hits this.

**Implementation sketch.**

```
Same structure as from_q1_0_bytes: allocate flat_out, loop over seq rows × heads × pair.
Need `f64.sin` / `f64.cos` (not in core WASM — use WASI preview imports or inline Taylor/
CORDIC, but preferably call the host's Math.sin via an imported fn). Easiest: add two
new WASM imports `host_sin` / `host_cos` that the JS glue supplies as `Math.sin`/`Math.cos`.
```

Inverse-frequency table is computed once per call via `f64.pow` (call into host too, or
use `2.0f64.powf(ln(theta_base) * exp)` derived).

## Nice to have — post-L5

### 3. CSE for LICM bindings

The Rust target still emits two `let __licm_N: Vec<u8> = file.clone().raw` when the outer
loop refers to the same expression twice. Not catastrophic (once per tensor extract, not
per iteration) but wastes 248 MB × duplication-count upfront. Track in a separate
`codegen-ideal-form` item.

### 4. Struct scalar field access without clone

`let start = file.data_offset + info.offset` generates `file.clone().data_offset +
info.clone().offset`. Scalar Copy fields should just read the field through a borrow.
Also a codegen-ideal-form item.

### 5. Inline Vec::push for known capacity

Even on the Rust target, `almide_rt_list_push` goes through a function call per element.
LLVM with LTO inlines it, but the generic `<A>` can still block auto-vectorisation. A
`list.push_all(xs, buf)` bulk primitive or inlining hint at call sites would close this.

## Deferred — tokeniser + generation (L4)

Explicitly out of scope for L5. The browser demo relies on a JS host (e.g. `tokenizers.js`
or a pre-tokenised input form) to keep scope tight; full Qwen3 BPE + KV cache + greedy
sampling loop lives in the separate L4 arc.
