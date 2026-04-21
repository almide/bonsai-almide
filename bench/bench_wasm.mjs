// bench_wasm — Almide WASM tok/s 計測.
//
// node bench/bench_wasm.mjs
//
// 同じプロンプト [785, 6722, 315, 6323, 374] ("The capital of Japan is")
// で 8 tok 生成し prompt-eval + tg のウォールタイムを出す.

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { makeWasi } from '../docs/wasi_shim.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function readResultInt(memory, rp) {
  const v = new DataView(memory.buffer);
  const tag = v.getInt32(rp, true);
  if (tag !== 0) {
    const payload = v.getBigInt64(rp + 4, true);
    const p = Number(payload);
    const l = v.getInt32(p, true);
    const s = new TextDecoder().decode(new Uint8Array(memory.buffer, p + 4, l));
    throw new Error('Almide err: ' + s);
  }
  return Number(v.getBigInt64(rp + 4, true));
}

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

async function main() {
  const wasmBuf = readFileSync(resolve(repoRoot, 'docs/bonsai.wasm'));
  const modelBuf = readFileSync(resolve(repoRoot, 'weights/Bonsai-1.7B-Q1_0.gguf'));

  const wasi = makeWasi({ stdout: () => {}, stderr: (s) => process.stderr.write(s + '\n') });
  const { instance } = await WebAssembly.instantiate(wasmBuf, wasi.imports);
  wasi.setMemory(instance.exports.memory);
  if (instance.exports._initialize) instance.exports._initialize();
  const { __alloc, __heap_save, __heap_restore, predict_next } = instance.exports;
  const memory = instance.exports.memory;

  console.log('# bench_wasm — Bonsai-1.7B-Q1_0 on Almide WASM (wasmtime / node)');
  console.log('health: ' + instance.exports.wasm_health_check());

  const t_write0 = performance.now();
  const modelPtr = (__alloc(4 + modelBuf.byteLength) >>> 0);
  new DataView(memory.buffer).setInt32(modelPtr, modelBuf.byteLength, true);
  new Uint8Array(memory.buffer).set(new Uint8Array(modelBuf), modelPtr + 4);
  const t_write1 = performance.now();
  console.log(`model write into WASM memory: ${(t_write1 - t_write0).toFixed(0)} ms`);

  const sessionMark = __heap_save() >>> 0;

  let context = [785, 6722, 315, 6323, 374];
  const promptLen = context.length;
  const nGen = 8;

  const t_pp0 = performance.now();
  {
    const tp = writeTokenList(instance, context);
    const rp = predict_next(modelPtr, tp);
    const next = readResultInt(memory, rp);
    __heap_restore(sessionMark);
    context.push(next);
    console.log(`  first next-tok id: ${next}`);
  }
  const t_pp1 = performance.now();
  const ppMs = t_pp1 - t_pp0;
  console.log(`prompt_eval (${promptLen} tok): ${ppMs.toFixed(0)} ms`);

  const t_g0 = performance.now();
  for (let step = 1; step < nGen; step++) {
    const tp = writeTokenList(instance, context);
    const rp = predict_next(modelPtr, tp);
    const next = readResultInt(memory, rp);
    __heap_restore(sessionMark);
    context.push(next);
  }
  const t_g1 = performance.now();
  const gMs = t_g1 - t_g0;
  const totalMs = ppMs + gMs;
  console.log(`gen ${nGen} tokens total: ${totalMs.toFixed(0)} ms`);
  console.log(`  avg per-token (incl prompt eval): ${(totalMs / nGen).toFixed(0)} ms`);
  const tgAvgMs = gMs / (nGen - 1);
  console.log(`  tg-only avg: ${tgAvgMs.toFixed(0)} ms/tok`);
  console.log(`  tg tok/s: ${(1000 / tgAvgMs).toFixed(3)}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
