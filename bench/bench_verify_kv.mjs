// bench_verify_kv — compare no-KV vs KV-streaming greedy outputs.
//
//   node bench/bench_verify_kv.mjs
//
// Runs predict_next and predict_prompt_kv_bytes/predict_step_kv_bytes
// with the same prompt on a tiny 3-token generation and prints both
// streams side-by-side so we can tell if the KV-cache path diverges
// from the verified non-KV reference.

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { makeWasi } from '../docs/wasi_shim.js';
import { AutoTokenizer } from '@huggingface/transformers';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function writeTokenList(instance, tokens) {
  const { __alloc, memory } = instance.exports;
  const ptr = (__alloc(4 + tokens.length * 8) >>> 0);
  const v = new DataView(memory.buffer);
  v.setInt32(ptr, tokens.length, true);
  for (let i = 0; i < tokens.length; i++) {
    v.setBigInt64(ptr + 4 + i * 8, BigInt(tokens[i]), true);
  }
  return ptr;
}

function writeBytesList(instance, arrs) {
  const { __alloc, memory } = instance.exports;
  const ps = new Array(arrs.length);
  for (let i = 0; i < arrs.length; i++) {
    const a = arrs[i];
    const p = (__alloc(4 + a.length) >>> 0);
    const v = new DataView(memory.buffer);
    v.setInt32(p, a.length, true);
    new Uint8Array(memory.buffer, p + 4, a.length).set(a);
    ps[i] = p;
  }
  const lp = (__alloc(4 + ps.length * 4) >>> 0);
  const v = new DataView(memory.buffer);
  v.setInt32(lp, ps.length, true);
  for (let i = 0; i < ps.length; i++) v.setInt32(lp + 4 + i * 4, ps[i], true);
  return lp;
}

function readResultInt(memory, rp) {
  const v = new DataView(memory.buffer);
  const tag = v.getInt32(rp, true);
  if (tag !== 0) throw new Error('err result');
  return Number(v.getBigInt64(rp + 4, true));
}

function readResultTupleIntKV(memory, rp) {
  let v = new DataView(memory.buffer);
  if (v.getInt32(rp, true) !== 0) throw new Error('err result');
  const tp = v.getInt32(rp + 4, true) >>> 0;
  v = new DataView(memory.buffer);
  const nextTok = Number(v.getBigInt64(tp, true));
  const kp = v.getInt32(tp + 8, true) >>> 0;
  const vp = v.getInt32(tp + 12, true) >>> 0;
  return {
    next: nextTok,
    keys: copyBytesListOut(memory, kp),
    values: copyBytesListOut(memory, vp),
  };
}

function copyBytesListOut(memory, lp) {
  let v = new DataView(memory.buffer);
  const n = v.getInt32(lp, true);
  const out = [];
  for (let i = 0; i < n; i++) {
    v = new DataView(memory.buffer);
    const bp = v.getInt32(lp + 4 + i * 4, true) >>> 0;
    const bl = v.getInt32(bp, true);
    const copy = new Uint8Array(bl);
    copy.set(new Uint8Array(memory.buffer, bp + 4, bl));
    out.push(copy);
  }
  return out;
}

async function main() {
  const wasmBuf = readFileSync(resolve(repoRoot, 'docs/bonsai.wasm'));
  const modelBuf = readFileSync(resolve(repoRoot, 'weights/Bonsai-1.7B-Q1_0.gguf'));
  const wasi = makeWasi({ stdout: () => {}, stderr: () => {} });
  const { instance } = await WebAssembly.instantiate(wasmBuf, wasi.imports);
  wasi.setMemory(instance.exports.memory);
  if (instance.exports._initialize) instance.exports._initialize();
  const {
    __alloc, __heap_save, __heap_restore,
    predict_next, predict_prompt_kv_bytes, predict_step_kv_bytes,
  } = instance.exports;
  const memory = instance.exports.memory;

  const modelPtr = (__alloc(4 + modelBuf.byteLength) >>> 0);
  new DataView(memory.buffer).setInt32(modelPtr, modelBuf.byteLength, true);
  new Uint8Array(memory.buffer).set(new Uint8Array(modelBuf), modelPtr + 4);

  const prompt = [785, 6722, 315, 6323, 374];
  const N = 4;

  // ── path 1: no-KV greedy (context grows, whole forward each step) ──
  console.log('=== path 1: no-KV greedy ===');
  const mark1 = __heap_save() >>> 0;
  const noKvGen = [];
  const ctx = [...prompt];
  for (let i = 0; i < N; i++) {
    const p = writeTokenList(instance, ctx);
    const rp = predict_next(modelPtr, p);
    const tok = readResultInt(memory, rp);
    __heap_restore(mark1);
    ctx.push(tok);
    noKvGen.push(tok);
    console.log(`  step ${i}: ${tok}`);
  }

  // ── path 2: KV-streaming greedy ──
  console.log('=== path 2: KV-streaming greedy (predict_prompt → predict_step) ===');
  const mark2 = __heap_save() >>> 0;
  const kvGen = [];
  const promptPtr = writeTokenList(instance, prompt);
  const rp0 = predict_prompt_kv_bytes(modelPtr, promptPtr, 0.0, 1n, 0.0, 1.0) >>> 0;
  const r0 = readResultTupleIntKV(memory, rp0);
  __heap_restore(mark2);
  kvGen.push(r0.next);
  console.log(`  step 0 (prompt-eval): ${r0.next}`);
  let curKeys = r0.keys, curValues = r0.values;
  let lastTok = r0.next;
  let pos = prompt.length;
  for (let i = 1; i < N; i++) {
    const kp = writeBytesList(instance, curKeys);
    const vp = writeBytesList(instance, curValues);
    const recentPtr = writeTokenList(instance, []);
    const rpi = predict_step_kv_bytes(
      modelPtr, kp, vp, BigInt(lastTok), BigInt(pos),
      recentPtr, 0.0, 1n, 0.0, 1.0,
    ) >>> 0;
    const r = readResultTupleIntKV(memory, rpi);
    __heap_restore(mark2);
    curKeys = r.keys; curValues = r.values;
    lastTok = r.next;
    pos += 1;
    kvGen.push(r.next);
    console.log(`  step ${i}: ${r.next}`);
  }

  // ── path 4: predict_prompt_kv_bytes on FULL context (empty KV) ──
  // Treats [prompt + Tokyo + ...] as a single prompt-eval. Bypasses
  // step-wise cache resumption. If forward is correct, should match
  // path-1 (no-KV) exactly.
  console.log('=== path 4: predict_prompt_kv_bytes on FULL context per step ===');
  const mark4 = __heap_save() >>> 0;
  const fullGen = [];
  const fullCtx = [...prompt];
  for (let i = 0; i < N; i++) {
    const p = writeTokenList(instance, fullCtx);
    const rp = predict_prompt_kv_bytes(modelPtr, p, 0.0, 1n, 0.0, 1.0) >>> 0;
    const r = readResultTupleIntKV(memory, rp);
    __heap_restore(mark4);
    fullGen.push(r.next);
    fullCtx.push(r.next);
    console.log(`  step ${i}: ${r.next}`);
  }

  // ── path 3: predict_tokens_kv (Almide-side List[Matrix] kv) ──
  console.log('=== path 3: predict_tokens_kv (internal KV, no bytes shuttling) ===');
  const mark3 = __heap_save() >>> 0;
  const promptPtr3 = writeTokenList(instance, prompt);
  const rp3 = instance.exports.predict_tokens_kv(modelPtr, promptPtr3, BigInt(N)) >>> 0;
  let v3 = new DataView(memory.buffer);
  if (v3.getInt32(rp3, true) !== 0) throw new Error('tokens_kv err');
  const lp = Number(v3.getBigInt64(rp3 + 4, true)) >>> 0;
  const len = new DataView(memory.buffer).getInt32(lp, true);
  const tokensKv = [];
  for (let i = 0; i < len; i++) {
    tokensKv.push(Number(new DataView(memory.buffer).getBigInt64(lp + 4 + i * 8, true)));
  }
  __heap_restore(mark3);
  console.log(`  ${tokensKv.join(', ')}`);

  console.log(`\nno-KV:          ${noKvGen.join(', ')}`);
  console.log(`KV-stream:      ${kvGen.join(', ')}`);
  console.log(`KV-full-prompt: ${fullGen.join(', ')}`);
  console.log(`KV-internal:    ${tokensKv.join(', ')}`);

  const tok = await AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B');
  console.log(`no-KV          → ${JSON.stringify(tok.decode(noKvGen, { skip_special_tokens: false }))}`);
  console.log(`KV-stream      → ${JSON.stringify(tok.decode(kvGen, { skip_special_tokens: false }))}`);
  console.log(`KV-full-prompt → ${JSON.stringify(tok.decode(fullGen, { skip_special_tokens: false }))}`);
  console.log(`KV-internal    → ${JSON.stringify(tok.decode(tokensKv, { skip_special_tokens: false }))}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
