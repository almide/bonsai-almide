// bench_wasm_kv — Almide WASM tok/s 計測 (KV cache 版).
//
// node bench/bench_wasm_kv.mjs
//
// 同じ prompt で 8 tok 生成。`predict_tokens_kv` に prompt + n_new を
// 渡すと KV cache を内部で保持したまま全部 gen して結果だけ返す。

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

function readResultList(memory, rp) {
  // WASM memory may have grown during the call, so rebind the view each
  // time we touch it. Using the stale buffer triggers "offset outside
  // the bounds of the DataView" when a memory.grow happened mid-call.
  let v = new DataView(memory.buffer);
  const tag = v.getInt32(rp, true);
  if (tag !== 0) {
    const p = Number(v.getBigInt64(rp + 4, true));
    v = new DataView(memory.buffer);
    const l = v.getInt32(p, true);
    const s = new TextDecoder().decode(new Uint8Array(memory.buffer, p + 4, l));
    throw new Error('Almide err: ' + s);
  }
  const lp = Number(v.getBigInt64(rp + 4, true));
  v = new DataView(memory.buffer);
  const len = v.getInt32(lp, true);
  const out = [];
  for (let i = 0; i < len; i++) {
    out.push(Number(v.getBigInt64(lp + 4 + i * 8, true)));
  }
  return out;
}

async function main() {
  const wasmBuf = readFileSync(resolve(repoRoot, 'docs/bonsai.wasm'));
  const modelBuf = readFileSync(resolve(repoRoot, 'weights/Bonsai-1.7B-Q1_0.gguf'));

  const wasi = makeWasi({ stdout: () => {}, stderr: (s) => process.stderr.write(s + '\n') });
  const { instance } = await WebAssembly.instantiate(wasmBuf, wasi.imports);
  wasi.setMemory(instance.exports.memory);
  if (instance.exports._initialize) instance.exports._initialize();
  const { __alloc, predict_tokens_kv } = instance.exports;
  const memory = instance.exports.memory;

  console.log('# bench_wasm_kv — Bonsai-1.7B-Q1_0 on Almide WASM (node, KV cache)');
  console.log('health: ' + instance.exports.wasm_health_check());

  const t_write0 = performance.now();
  const modelPtr = (__alloc(4 + modelBuf.byteLength) >>> 0);
  new DataView(memory.buffer).setInt32(modelPtr, modelBuf.byteLength, true);
  new Uint8Array(memory.buffer).set(new Uint8Array(modelBuf), modelPtr + 4);
  const t_write1 = performance.now();
  console.log(`model write into WASM memory: ${(t_write1 - t_write0).toFixed(0)} ms`);

  const prompt = [785, 6722, 315, 6323, 374];
  const nNew = Number(process.env.N_NEW || 1);
  const promptPtr = writeTokenList(instance, prompt);

  const t0 = performance.now();
  const rp = (predict_tokens_kv(modelPtr, promptPtr, BigInt(nNew)) >>> 0);
  const tokens = readResultList(memory, rp);
  console.log(`  peak heap: ${(memory.buffer.byteLength / 1024 / 1024).toFixed(0)} MB`);
  const t1 = performance.now();
  const totalMs = t1 - t0;
  console.log(`predict_tokens_kv returned ${tokens.length} tokens: ${tokens.join(', ')}`);
  console.log(`total (pp${prompt.length} + gen${nNew}): ${totalMs.toFixed(0)} ms`);
  console.log(`  avg per-token: ${(totalMs / nNew).toFixed(0)} ms`);
  // The first token is the prompt-eval result (counts pp cost). Approximate
  // pp/gen split by attributing the first call's duration — but we only
  // have aggregate; report the whole number as "effective tok/s" too.
  console.log(`  effective tok/s (gen-only lower bound): ${(1000 * nNew / totalMs).toFixed(3)}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
