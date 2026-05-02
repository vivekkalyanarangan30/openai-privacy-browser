# openai/privacy-filter — browser demo

A single-page demo that loads [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter)
(1.5B-param sparse-MoE PII token classifier — 50 M active params, 8 transformer blocks, 128 experts top-4)
**fully in the browser** via [Transformers.js v4](https://huggingface.co/docs/transformers.js)
and masks PII in an email as you type. The right-hand diagnostics panel surfaces
system-health and model-metric numbers live so you can read perf without opening
DevTools.

OpenAI ships a pre-quantized `model_q4f16.onnx` (~810 MB) inside the source repo;
their model card recommends it as the browser path. We use that artifact directly —
no re-quantization, no offline eval pipeline. The product *is* the UI, and the
UI itself is the perf report.

## What you get

**Left column — Input & Masked Output**
- Pre-filled multi-line sample email containing all 8 PII categories.
- 250 ms-debounced inference re-runs as you type.
- Inline `[<LABEL>]` chips replace each detected span (color-coded per label).
- Sortable table of detected entities with score and word.

**Right column — Diagnostics**

*System Health*
- Chosen `device` / `dtype`, GPU adapter info, `shader-f16` support.
- WebAssembly threads + SIMD, `crossOriginIsolated`.
- Live JS heap (used / total) + `performance.measureUserAgentSpecificMemory()` when isolated.

*Model Metrics — Load*
- Total bytes downloaded + time + MB/s.
- Pipeline-init time, total cold-load time, first-inference latency.
- Estimated weight footprint, JS-heap delta after load.

*Model Metrics — Inference (last + rolling)*
- Per-call breakdown: tokenize ms · forward+decode ms · total ms · input tokens · output spans.
- Last 50 inferences: p50 / p95 / mean / max + tokens/sec + inferences/min + sparkline.

*Event log* — bounded, per-event timing breakdown, scrolls.

*Copy-as-Markdown* — drops the entire diagnostics state into your clipboard.

## Running it locally

Requires Node ≥ 22 and npm.

```bash
npm install
npm run dev
# open http://localhost:5173
# click "Load model" — the ~810 MB ONNX downloads once and is cached in IndexedDB
```

The dev and preview servers serve with `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`, which is required for
`measureUserAgentSpecificMemory()` and SharedArrayBuffer-backed WASM threads.

## Auto-precision

`src/diagnostics.ts` picks the runtime at probe time:

| Capability | Choice |
| --- | --- |
| WebGPU + `shader-f16` | `device:"webgpu", dtype:"q4f16"` (~810 MB on disk, ~825 MB GPU) |
| WebGPU only | `device:"webgpu", dtype:"q4"` (~917 MB on disk) |
| **no WebGPU** | **not viable — see below** |

### Why this model needs WebGPU

Every quantized ONNX variant of `openai/privacy-filter` (q4, q4f16, q8) gathers
embeddings via the `GatherBlockQuantized` op. ONNX Runtime Web only implements that
kernel in the **WebGPU** execution provider. On a WASM/CPU runtime, session creation
fails with:

```
ERROR_CODE: 9, ERROR_MESSAGE: Could not find an implementation for
GatherBlockQuantized(1) node with name '/model/embed_tokens/Gather_Quant'
```

The only ONNX variants that *would* load on WASM are `model.onnx` (FP32, 5.6 GB) and
`model_fp16.onnx` (2.8 GB) — neither is realistic for a browser. So the demo shows a
red banner explaining the situation when no WebGPU adapter is found, instead of
silently degrading.

WebGPU is on by default in Chrome/Edge 113+ and Safari 18+ (macOS 14+, iOS 18+).
Firefox ships it behind `dom.webgpu.enabled`. SwiftShader's software WebGPU
(used by the headless e2e tests here) also works.

URL flags drive deterministic Playwright runs: `?device=webgpu`, `?device=wasm`,
`?autoload=1` skips the click.

## End-to-end tests

```bash
npx playwright install chromium     # one-time, ~170 MB
npm run test:e2e
```

Two scenarios run serially against `npm run preview`, both on WebGPU
(SwiftShader software adapter — runs headless in CI):

1. **End-to-end masking** — opens the demo, waits for the model to download and
   reach Ready, then asserts `[private_email]`, `[private_person]`, `[private_phone]`
   chips render on the prefilled email. Saves `screenshots/webgpu.png`.
2. **Live debounce** — replaces the textarea content with new text, asserts a *second*
   inference fires after the debounce window, and the masked output reflects the new
   content (different chip count, only `private_email` + `private_person`).
   Saves `screenshots/webgpu-live.png`.

Persistent profiles in `.pw-profiles/webgpu/` cache the ~810 MB ONNX in IndexedDB
between scenarios so the second one is fast.

A WASM scenario was tried and dropped — see *Why this model needs WebGPU* above.

A `screenshots/perf-report.json` is written with the per-scenario device/dtype/latency
numbers Playwright scrapes from the diagnostics panel.

## Hosting on HuggingFace Spaces

The demo is a plain static site, so the right home on HF is a **Space** with the
**Static SDK** — no Docker, no Python runtime needed. The model itself stays at
`openai/privacy-filter`; the Space only hosts this UI.

One-time prereqs:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login        # paste a write token from https://huggingface.co/settings/tokens
```

Deploy (builds `dist/` and pushes):

```bash
scripts/deploy-hf-space.sh <user-or-org>/<space-name>
# e.g. scripts/deploy-hf-space.sh vivekkalyanarangan/privacy-filter-browser
```

Your Space lives at `https://huggingface.co/spaces/<user>/<space-name>`; the running
app at `https://<user>-<space-name>.static.hf.space`. First load downloads the
~810 MB ONNX from `openai/privacy-filter`; subsequent visits are instant via
IndexedDB cache.

### Caveats on Spaces hosting

- **`measureUserAgentSpecificMemory()` will read "n/a"** — HF Spaces' static SDK
  emits `Cross-Origin-Opener-Policy: same-origin` but **not**
  `Cross-Origin-Embedder-Policy: require-corp`, so `crossOriginIsolated` is false.
  The diagnostics panel falls back to JS-heap numbers gracefully. If you need true
  cross-origin isolation, switch the Space to the **Docker SDK** and serve `dist/`
  from an nginx/Caddy that emits both headers — the app code itself doesn't
  need to change.
- **WebGPU on the Space side is the user's browser**, not the server, so anyone
  on Chrome/Edge 113+ or Safari 18+ gets the GPU path. WebGPU-less visitors
  see the red banner the local app shows (no silent CPU fallback).

## Project layout

```
src/
  main.ts          # UI + worker glue + debounce + render
  worker.ts        # singleton Transformers.js pipeline + perf timing
  diagnostics.ts   # WebGPU / shader-f16 / WASM-threads probe + auto-precision
  health.ts        # rolling stats, event log, memory polling
  mask.ts          # spans → DocumentFragment (inline [LABEL] chips)
  ui/
    layout.css
    sample-email.ts
tests/
  e2e.spec.ts
playwright.config.ts
vite.config.ts     # COOP/COEP, ESM workers
```

## Attribution

- Model: [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) — Apache-2.0, OpenAI.
- Runtime: [`@huggingface/transformers`](https://www.npmjs.com/package/@huggingface/transformers) v4.

Inference is 100 % local. Text never leaves the tab.
