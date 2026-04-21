// bench_wasm_kv_stream — prove the streaming KV-bytes path works.
//
//   node bench/bench_wasm_kv_stream.mjs
//
// Calls predict_prompt_kv_bytes once on "The capital of Japan is" (5
// tokens), then one predict_step_kv_bytes for a single gen step. Copies
// the kv bytes out between calls + __heap_restores so the bump allocator
// stays flat — matches what the browser demo will do. First gen token
// must be 26194 (" Tokyo") for the path to be numerically correct.

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { makeWasi } from '../docs/wasi_shim.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function writeTokenList(instance, tokens) {
  const { __alloc, memory } = instance.exports;
  const ptr = (__alloc(4 + tokens.length * 8) >>> 0);
  const view = new DataView(memory.buffer);
  view.setInt32(ptr, tokens.length, true);
  for (let i = 0; i < tokens.length; i++) {
    view.setBigInt64(ptr + 4 + i * 8, BigInt(tokens[i]), true);
  }
  return ptr;
}

function readResultTupleIntKV(memory, rp) {
  let v = new DataView(memory.buffer);
  const tag = v.getInt32(rp, true);
  if (tag !== 0) {
    const errPtr = v.getInt32(rp + 4, true) >>> 0;
    const el = v.getInt32(errPtr, true);
    throw new Error('Almide err: ' + new TextDecoder().decode(new Uint8Array(memory.buffer, errPtr + 4, el)));
  }
  const tuplePtr = v.getInt32(rp + 4, true) >>> 0;
  v = new DataView(memory.buffer);
  const nextTok = Number(v.getBigInt64(tuplePtr, true));
  const keysListPtr = v.getInt32(tuplePtr + 8, true) >>> 0;
  const valuesListPtr = v.getInt32(tuplePtr + 12, true) >>> 0;
  return {
    next: nextTok,
    keys: copyBytesListOut(memory, keysListPtr),
    values: copyBytesListOut(memory, valuesListPtr),
  };
}

function copyBytesListOut(memory, listPtr) {
  let v = new DataView(memory.buffer);
  const len = v.getInt32(listPtr, true);
  const out = [];
  for (let i = 0; i < len; i++) {
    v = new DataView(memory.buffer);
    const bp = v.getInt32(listPtr + 4 + i * 4, true) >>> 0;
    const bl = v.getInt32(bp, true);
    const copy = new Uint8Array(bl);
    copy.set(new Uint8Array(memory.buffer, bp + 4, bl));
    out.push(copy);
  }
  return out;
}

function writeBytesList(instance, arrs) {
  const { __alloc, memory } = instance.exports;
  const bytesPtrs = new Array(arrs.length);
  for (let i = 0; i < arrs.length; i++) {
    const a = arrs[i];
    const p = (__alloc(4 + a.length) >>> 0);
    const v = new DataView(memory.buffer);
    v.setInt32(p, a.length, true);
    new Uint8Array(memory.buffer, p + 4, a.length).set(a);
    bytesPtrs[i] = p;
  }
  const listPtr = (__alloc(4 + bytesPtrs.length * 4) >>> 0);
  const v = new DataView(memory.buffer);
  v.setInt32(listPtr, bytesPtrs.length, true);
  for (let i = 0; i < bytesPtrs.length; i++) {
    v.setInt32(listPtr + 4 + i * 4, bytesPtrs[i], true);
  }
  return listPtr;
}

async function main() {
  const wasmBuf = readFileSync(resolve(repoRoot, 'docs/bonsai.wasm'));
  const modelBuf = readFileSync(resolve(repoRoot, 'weights/Bonsai-1.7B-Q1_0.gguf'));

  const wasi = makeWasi({ stdout: () => {}, stderr: (s) => process.stderr.write(s + '\n') });
  const { instance } = await WebAssembly.instantiate(wasmBuf, wasi.imports);
  wasi.setMemory(instance.exports.memory);
  if (instance.exports._initialize) instance.exports._initialize();
  const {
    __alloc, __heap_save, __heap_restore,
    predict_prompt_kv_bytes, predict_step_kv_bytes,
  } = instance.exports;
  const memory = instance.exports.memory;

  console.log('# bench_wasm_kv_stream — streaming KV-bytes round-trip');
  console.log('health: ' + instance.exports.wasm_health_check());

  const t_write0 = performance.now();
  const modelPtr = (__alloc(4 + modelBuf.byteLength) >>> 0);
  new DataView(memory.buffer).setInt32(modelPtr, modelBuf.byteLength, true);
  new Uint8Array(memory.buffer).set(new Uint8Array(modelBuf), modelPtr + 4);
  const t_write1 = performance.now();
  console.log(`model write: ${(t_write1 - t_write0).toFixed(0)} ms`);

  const sessionMark = (__heap_save() >>> 0);
  const prompt = [785, 6722, 315, 6323, 374];
  const nGen = Number(process.env.N_GEN || 1);
  const promptPtr = writeTokenList(instance, prompt);

  console.log(`prompt eval (${prompt.length} tokens)…`);
  const t0 = performance.now();
  const rp0 = (predict_prompt_kv_bytes(modelPtr, promptPtr) >>> 0);
  const { next: firstTok, keys: keys0, values: values0 } = readResultTupleIntKV(memory, rp0);
  const t1 = performance.now();
  __heap_restore(sessionMark);
  console.log(`  first tok = ${firstTok} (expect 26194)  in ${(t1 - t0).toFixed(0)} ms`);
  console.log(`  kv: ${keys0.length} layers × ${keys0[0]?.length ?? 0} B/layer = ${(keys0.length * (keys0[0]?.length ?? 0) / 1024 / 1024).toFixed(2)} MB`);

  let curKeys = keys0, curValues = values0;
  let lastTok = firstTok;
  let pos = prompt.length;
  const generated = [firstTok];
  for (let step = 1; step < nGen; step++) {
    const keysListPtr = writeBytesList(instance, curKeys);
    const valuesListPtr = writeBytesList(instance, curValues);
    const ts = performance.now();
    const rpi = (predict_step_kv_bytes(
      modelPtr, keysListPtr, valuesListPtr,
      BigInt(lastTok), BigInt(pos),
    ) >>> 0);
    const r = readResultTupleIntKV(memory, rpi);
    const elapsed = performance.now() - ts;
    __heap_restore(sessionMark);
    curKeys = r.keys;
    curValues = r.values;
    lastTok = r.next;
    pos += 1;
    generated.push(r.next);
    console.log(`  step ${step}: next = ${r.next}  in ${elapsed.toFixed(0)} ms  peak heap ${(memory.buffer.byteLength / 1024 / 1024).toFixed(0)} MB`);
  }
  console.log(`generated: ${generated.join(', ')}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
