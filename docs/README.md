# Bonsai × Almide — Browser Demo

Live at: https://almide.github.io/bonsai-almide/

Open `index.html` in a modern browser (Chrome, Safari, Firefox). The first
visit downloads 248 MB of Bonsai Q1_0 weights from Hugging Face CDN and
caches them in IndexedDB; subsequent visits are instant.

## Files

- `index.html` — demo page (fetches GGUF, loads WASM, chat UI)
- `wasi_shim.js` — minimal WASI imports for Almide's WASM runtime
- `bonsai.wasm` — pure-Almide inference engine, ~22 KB
