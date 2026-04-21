// Runs probe_prompt_then_step which keeps kv in Almide memory (no bytes
// shuttling). If the returned token == 13 → serialization is the bug.
// If it == 25 → step-wise forward itself is the bug.

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

async function main() {
  const wasmBuf = readFileSync(resolve(repoRoot, 'docs/bonsai.wasm'));
  const modelBuf = readFileSync(resolve(repoRoot, 'weights/Bonsai-1.7B-Q1_0.gguf'));
  const wasi = makeWasi({ stdout: () => {}, stderr: () => {} });
  const { instance } = await WebAssembly.instantiate(wasmBuf, wasi.imports);
  wasi.setMemory(instance.exports.memory);
  if (instance.exports._initialize) instance.exports._initialize();
  const { __alloc, probe_prompt_then_step } = instance.exports;
  const memory = instance.exports.memory;

  const modelPtr = (__alloc(4 + modelBuf.byteLength) >>> 0);
  new DataView(memory.buffer).setInt32(modelPtr, modelBuf.byteLength, true);
  new Uint8Array(memory.buffer).set(new Uint8Array(modelBuf), modelPtr + 4);

  const prompt = [785, 6722, 315, 6323, 374];
  const followTok = 26194; // " Tokyo"
  const promptPtr = writeTokenList(instance, prompt);

  const rp = probe_prompt_then_step(modelPtr, promptPtr, BigInt(followTok)) >>> 0;
  const v = new DataView(memory.buffer);
  if (v.getInt32(rp, true) !== 0) throw new Error('err result');
  const tok = Number(v.getBigInt64(rp + 4, true));
  console.log(`probe_prompt_then_step returned: ${tok}`);
  console.log(`  if 13 → bug is in bytes serialization`);
  console.log(`  if 25 → bug is in step-wise forward itself`);
}

main().catch((e) => { console.error(e); process.exit(1); });
