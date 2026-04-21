// bench_cached_kv — confirm parse-once + cached predict entries work,
// and measure the per-step savings vs the re-parsing path.
//
//   node bench/bench_cached_kv.mjs
//   N_GEN=5 node bench/bench_cached_kv.mjs

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { makeWasi } from '../docs/wasi_shim.js';

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

function readResultI32(memory, rp) {
  const v = new DataView(memory.buffer);
  if (v.getInt32(rp, true) !== 0) throw new Error('result tagged err');
  return v.getInt32(rp + 4, true) >>> 0;
}

function readResultTupleIntKV(memory, rp) {
  let v = new DataView(memory.buffer);
  if (v.getInt32(rp, true) !== 0) {
    const ep = v.getInt32(rp + 4, true) >>> 0;
    const el = v.getInt32(ep, true);
    throw new Error('err: ' + new TextDecoder().decode(new Uint8Array(memory.buffer, ep + 4, el)));
  }
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
    init_model, predict_prompt_kv_cached, predict_step_kv_cached,
    predict_prompt_kv_bytes,
  } = instance.exports;
  const memory = instance.exports.memory;

  const modelPtr = (__alloc(4 + modelBuf.byteLength) >>> 0);
  new DataView(memory.buffer).setInt32(modelPtr, modelBuf.byteLength, true);
  new Uint8Array(memory.buffer).set(new Uint8Array(modelBuf), modelPtr + 4);
  console.log(`model written at ptr=${modelPtr}`);

  // Parse once; capture the model mark AFTER init_model so the GGUFFile
  // record is below the restore point and survives __heap_restore.
  const t0 = performance.now();
  const filePtr = readResultI32(memory, init_model(modelPtr) >>> 0);
  const parseMs = performance.now() - t0;
  const modelMark = __heap_save() >>> 0;
  console.log(`init_model: parsed at ptr=${filePtr} in ${parseMs.toFixed(0)} ms; modelMark=${modelMark}`);

  const prompt = [785, 6722, 315, 6323, 374];
  const nGen = Number(process.env.N_GEN || 3);

  // --- cached path ---
  console.log('=== cached path ===');
  const promptPtr = writeTokenList(instance, prompt);
  const tCp0 = performance.now();
  const rp0 = (predict_prompt_kv_cached(
    filePtr, promptPtr, 0.0, 1n, 0.0, 1.0,
  ) >>> 0);
  const r0 = readResultTupleIntKV(memory, rp0);
  const cpPromptMs = performance.now() - tCp0;
  __heap_restore(modelMark);
  console.log(`  prompt eval (cached): tok=${r0.next} in ${cpPromptMs.toFixed(0)} ms`);

  let curKeys = r0.keys, curValues = r0.values;
  let lastTok = r0.next, pos = prompt.length;
  for (let i = 1; i < nGen; i++) {
    const kp = writeBytesList(instance, curKeys);
    const vp = writeBytesList(instance, curValues);
    const rpi = writeTokenList(instance, []);
    const ts = performance.now();
    const rpv = (predict_step_kv_cached(
      filePtr, kp, vp, BigInt(lastTok), BigInt(pos),
      rpi, 0.0, 1n, 0.0, 1.0,
    ) >>> 0);
    const r = readResultTupleIntKV(memory, rpv);
    const stepMs = performance.now() - ts;
    __heap_restore(modelMark);
    console.log(`  step ${i} (cached): tok=${r.next} in ${stepMs.toFixed(0)} ms`);
    curKeys = r.keys; curValues = r.values;
    lastTok = r.next; pos += 1;
  }

  // --- re-parsing path for comparison (run twice to warm up JIT) ---
  console.log('=== re-parsing path (old) ===');
  for (let trial = 0; trial < 2; trial++) {
    const reParseMark = __heap_save() >>> 0;
    const promptPtr2 = writeTokenList(instance, prompt);
    const tRp0 = performance.now();
    const rpOld = (predict_prompt_kv_bytes(
      modelPtr, promptPtr2, 0.0, 1n, 0.0, 1.0,
    ) >>> 0);
    const rOld = readResultTupleIntKV(memory, rpOld);
    const rpPromptMs = performance.now() - tRp0;
    __heap_restore(reParseMark);
    console.log(`  trial ${trial + 1} prompt eval (old): tok=${rOld.next} in ${rpPromptMs.toFixed(0)} ms`);
  }

  // --- second cached prompt call, after JIT warm ---
  console.log('=== cached path (second call, JIT warm) ===');
  const promptPtrX = writeTokenList(instance, prompt);
  const tX0 = performance.now();
  const rpX = (predict_prompt_kv_cached(
    filePtr, promptPtrX, 0.0, 1n, 0.0, 1.0,
  ) >>> 0);
  const rX = readResultTupleIntKV(memory, rpX);
  const cpMs2 = performance.now() - tX0;
  __heap_restore(modelMark);
  console.log(`  prompt eval (cached, warm): tok=${rX.next} in ${cpMs2.toFixed(0)} ms`);
}

main().catch((e) => { console.error(e); process.exit(1); });
