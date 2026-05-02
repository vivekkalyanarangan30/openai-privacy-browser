// Main thread: builds the UI, wires the worker, owns the diagnostics state.
//
// Layout:
//   Left  — textarea + masked-text + entities table
//   Right — diagnostics panel (System Health + Model Metrics + Event Log)
//

import { pickRuntime, type RuntimeChoice, type Device } from "./diagnostics";
import {
  EventLog,
  InferenceMetrics,
  emptyLoadMetrics,
  fmtBytes,
  fmtMs,
  fmtSecs,
  snapshotMemory,
  type LoadMetrics,
  type MemorySnapshot,
} from "./health";
import { renderMasked, type Span, normalizeSpans } from "./mask";
import { SAMPLE_EMAIL } from "./ui/sample-email";

type AppStatus = "idle" | "downloading" | "initializing" | "ready" | "inferring" | "error";

type WorkerOutMsg =
  | { type: "ready" }
  | { type: "runtime"; runtime: RuntimeChoice }
  | { type: "progress"; evt: { status?: string; file?: string; loaded?: number; total?: number; progress?: number } }
  | { type: "result"; id: number; spans: Span[] }
  | {
      type: "perf";
      phase: "load";
      downloadMs: number;
      initMs: number;
      totalLoadMs: number;
      downloadedBytes: number;
    }
  | {
      type: "perf";
      phase: "load-file";
      file: string;
      bytes: number | null;
      tWallMs: number;
    }
  | {
      type: "perf";
      phase: "infer";
      id: number;
      tokenizeMs: number;
      inferMs: number;
      totalMs: number;
      inputTokens: number;
      outputSpans: number;
      isFirst: boolean;
    }
  | { type: "error"; phase: string; id?: number; message: string };

interface AppState {
  status: AppStatus;
  statusDetail: string;
  forceDevice?: Device;
  runtime: RuntimeChoice | null;
  download: { loaded: number; total: number; file: string };
  load: LoadMetrics;
  metrics: InferenceMetrics;
  log: EventLog;
  memory: MemorySnapshot | null;
  text: string;
  lastSpans: Span[];
  pendingId: number;
  inferStartMs: number | null;
}

const DEBOUNCE_MS = 250;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Partial<Record<string, string>> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined) continue;
    if (k === "class") e.className = v;
    else if (k === "data-testid") e.setAttribute("data-testid", v);
    else if (k.startsWith("data-")) e.setAttribute(k, v);
    else (e as unknown as Record<string, unknown>)[k] = v;
  }
  for (const c of children) {
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function kv(k: string, v: string | Node, valueClass = ""): HTMLElement {
  const kEl = el("span", { class: "k" }, [k]);
  const vEl = el("span", { class: `v ${valueClass}`.trim() });
  if (typeof v === "string") vEl.textContent = v;
  else vEl.appendChild(v);
  const wrap = document.createDocumentFragment();
  wrap.appendChild(kEl);
  wrap.appendChild(vEl);
  // Return a wrapper to satisfy the CSS grid layout (we'll use parent .kv container).
  const span = el("div", { style: "display:contents" });
  span.appendChild(kEl);
  span.appendChild(vEl);
  return span;
}

function statusPill(s: AppStatus, detail: string): HTMLElement {
  const cls =
    s === "ready" ? "s-ready" :
    s === "error" ? "s-error" :
    s === "inferring" ? "s-inferring" :
    "s-loading";
  const text =
    s === "idle" ? "Idle" :
    s === "downloading" ? `Downloading ${detail}` :
    s === "initializing" ? "Initializing" :
    s === "ready" ? "Ready" :
    s === "inferring" ? "Inferring" :
    `Error: ${detail}`;
  return el("span", { class: `status-pill ${cls}`, "data-testid": "status-pill" }, [text]);
}

function sparklinePath(values: number[], width: number, height: number): string {
  if (values.length < 2) return "";
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  const step = width / (values.length - 1);
  return values
    .map((v, i) => {
      const x = i * step;
      const y = height - ((v - min) / range) * height;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function paramCount(): number {
  // openai/privacy-filter has 1.5B total parameters per the official model card.
  return 1_500_000_000;
}

function estGpuBytes(dtype: string): number {
  // Rough bytes-per-param for the dominant weight format.
  // q4f16 packs INT4 weights with FP16 scales: ~0.55 B/param effective.
  const map: Record<string, number> = { q4f16: 0.55, q4: 0.6, q8: 1.05, fp16: 2.05 };
  return Math.round(paramCount() * (map[dtype] ?? 1.0));
}

const state: AppState = {
  status: "idle",
  statusDetail: "",
  runtime: null,
  download: { loaded: 0, total: 0, file: "" },
  load: emptyLoadMetrics(),
  metrics: new InferenceMetrics(),
  log: new EventLog(40),
  memory: null,
  text: SAMPLE_EMAIL,
  lastSpans: [],
  pendingId: 0,
  inferStartMs: null,
};

let worker: Worker | null = null;
let renderRaf: number | null = null;

function scheduleRender() {
  if (renderRaf !== null) return;
  renderRaf = requestAnimationFrame(() => {
    renderRaf = null;
    render();
  });
}

state.log.subscribe(scheduleRender);

function startWorker(forceDevice?: Device) {
  state.forceDevice = forceDevice;
  state.status = "initializing";
  state.statusDetail = "starting worker";
  state.log.push("ui", "Load model clicked");
  worker?.terminate();
  worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  worker.addEventListener("message", onWorkerMessage);
  worker.addEventListener("error", (e) => {
    state.status = "error";
    state.statusDetail = e.message || "worker error";
    state.log.push("error", `Worker error: ${e.message}`);
    scheduleRender();
  });
  // Capture the JS heap baseline before the model loads.
  void snapshotMemory().then((m) => {
    state.memory = m;
    state.load.heapBeforeBytes = m.jsHeapUsedBytes;
    scheduleRender();
  });
  worker.postMessage({ type: "init", forceDevice });
  scheduleRender();
}

function onWorkerMessage(ev: MessageEvent<WorkerOutMsg>) {
  const msg = ev.data;
  switch (msg.type) {
    case "runtime": {
      state.runtime = msg.runtime;
      state.log.push("load", `Runtime: ${msg.runtime.device}/${msg.runtime.dtype} — ${msg.runtime.reason}`);
      scheduleRender();
      break;
    }
    case "progress": {
      const e = msg.evt;
      if (e.file) state.download.file = e.file;
      if (typeof e.loaded === "number") state.download.loaded = e.loaded;
      if (typeof e.total === "number") state.download.total = e.total;
      if (e.status === "download" || e.status === "progress") {
        state.status = "downloading";
        const pct = e.total ? Math.round(((e.loaded ?? 0) / e.total) * 100) : 0;
        state.statusDetail = `${e.file ?? ""} ${pct}%`;
      } else if (e.status === "ready") {
        state.status = "initializing";
        state.statusDetail = "compiling";
      }
      scheduleRender();
      break;
    }
    case "perf": {
      if (msg.phase === "load") {
        state.load.downloadMs = msg.downloadMs;
        state.load.initMs = msg.initMs;
        state.load.totalLoadMs = msg.totalLoadMs;
        state.load.downloadBytes = msg.downloadedBytes;
        state.load.estimatedModelMemBytes = state.runtime
          ? estGpuBytes(state.runtime.dtype)
          : 0;
        state.log.push(
          "load",
          `Loaded: ${fmtBytes(msg.downloadedBytes)} in ${fmtSecs(msg.downloadMs)} (init ${fmtMs(msg.initMs)}, total ${fmtSecs(msg.totalLoadMs)})`,
        );
      } else if (msg.phase === "load-file") {
        state.log.push("load", `Fetched ${msg.file} (${fmtBytes(msg.bytes)})`);
      } else if (msg.phase === "infer") {
        state.metrics.push({
          tokenizeMs: msg.tokenizeMs,
          inferMs: msg.inferMs,
          totalMs: msg.totalMs,
          inputTokens: msg.inputTokens,
          outputSpans: msg.outputSpans,
        });
        if (msg.isFirst) {
          state.load.firstInferenceMs = msg.totalMs;
          // Snapshot heap after first inference for the heap-delta metric.
          void snapshotMemory().then((m) => {
            state.memory = m;
            state.load.heapAfterFirstInferBytes = m.jsHeapUsedBytes;
            scheduleRender();
          });
        }
        state.log.push(
          "infer",
          `tokenize ${fmtMs(msg.tokenizeMs)} · forward+decode ${fmtMs(msg.inferMs)} · total ${fmtMs(msg.totalMs)} · ${msg.inputTokens} tok → ${msg.outputSpans} span(s)`,
        );
      }
      scheduleRender();
      break;
    }
    case "ready": {
      state.status = "ready";
      state.statusDetail = "";
      state.log.push("load", `Ready — ${state.runtime?.device}/${state.runtime?.dtype}`);
      // Auto-run the first inference on the prefilled email.
      kickInference();
      scheduleRender();
      break;
    }
    case "result": {
      if (msg.id !== state.pendingId) return;
      state.lastSpans = normalizeSpans(state.text, msg.spans);
      state.status = "ready";
      state.statusDetail = "";
      scheduleRender();
      break;
    }
    case "error": {
      state.status = "error";
      state.statusDetail = msg.message;
      state.log.push("error", `${msg.phase}: ${msg.message}`);
      scheduleRender();
      break;
    }
  }
}

let debounceTimer: number | null = null;
function debouncedInference() {
  if (debounceTimer !== null) window.clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(() => {
    debounceTimer = null;
    kickInference();
  }, DEBOUNCE_MS);
}

function kickInference() {
  if (!worker || state.status === "downloading" || state.status === "initializing") return;
  if (!state.text) {
    state.lastSpans = [];
    scheduleRender();
    return;
  }
  state.pendingId += 1;
  state.status = "inferring";
  state.inferStartMs = performance.now();
  state.log.push("ui", `dispatch infer #${state.pendingId} (${state.text.length} chars)`);
  worker.postMessage({ type: "infer", id: state.pendingId, text: state.text });
  scheduleRender();
}

// --- rendering ---

function render() {
  const root = document.getElementById("app");
  if (!root) return;
  root.replaceChildren(buildHeader(), buildLayout(), buildFooter());
}

function buildHeader(): HTMLElement {
  return el("header", { class: "app-header" }, [
    el("div", {}, [
      el("h1", {}, ["openai/privacy-filter — browser demo"]),
      el("div", { class: "subtitle" }, [
        "live PII masking via Transformers.js v4 · 100% in-browser inference",
      ]),
    ]),
    statusPill(state.status, state.statusDetail),
  ]);
}

function buildLayout(): HTMLElement {
  const children: Node[] = [];
  if (state.runtime?.device === "wasm") {
    children.push(buildWasmBanner());
  }
  children.push(
    el("div", { class: "layout" }, [buildLeft(), buildRight()]),
  );
  const wrap = el("div", {});
  for (const c of children) wrap.appendChild(c);
  return wrap;
}

function buildWasmBanner(): HTMLElement {
  return el(
    "div",
    {
      class: "panel",
      style:
        "background:#5c1f1f;color:#ffd9d9;border-color:#7a3030;margin-bottom:12px;padding:10px 14px;",
      "data-testid": "wasm-banner",
    },
    [
      el("strong", {}, ["WebGPU not available."]),
      " ",
      state.runtime?.reason ?? "",
    ],
  );
}

function buildLeft(): HTMLElement {
  const textarea = el("textarea", { id: "input", "data-testid": "input" });
  textarea.value = state.text;
  textarea.addEventListener("input", () => {
    state.text = (textarea as HTMLTextAreaElement).value;
    debouncedInference();
  });

  const maskedDiv = el("div", { id: "masked-output", "data-testid": "masked-output" });
  if (state.lastSpans.length === 0 && state.status !== "ready") {
    maskedDiv.classList.add("empty");
    maskedDiv.textContent = state.status === "idle" ? "Click Load model to begin." : "Waiting for first inference…";
  } else {
    maskedDiv.appendChild(renderMasked(state.text, state.lastSpans));
  }

  const buttons = el("div", { class: "row", style: "margin-top:8px;" }, [
    el(
      "button",
      {
        class: "btn",
        "data-testid": "btn-reset",
      },
      ["Reset to sample"],
    ),
    el("button", { class: "btn", "data-testid": "btn-clear" }, ["Clear"]),
    state.status === "idle"
      ? el("button", { class: "btn btn-primary", "data-testid": "btn-load" }, ["Load model"])
      : el("span", {}),
  ]);

  buttons.querySelector('[data-testid="btn-reset"]')?.addEventListener("click", () => {
    state.text = SAMPLE_EMAIL;
    (textarea as HTMLTextAreaElement).value = SAMPLE_EMAIL;
    debouncedInference();
    scheduleRender();
  });
  buttons.querySelector('[data-testid="btn-clear"]')?.addEventListener("click", () => {
    state.text = "";
    (textarea as HTMLTextAreaElement).value = "";
    state.lastSpans = [];
    scheduleRender();
  });
  buttons.querySelector('[data-testid="btn-load"]')?.addEventListener("click", () => {
    startWorker();
  });

  const left = el("section", { class: "panel" }, [
    el("h2", {}, ["Input"]),
    textarea,
    buttons,
    el("h2", {}, ["Masked output"]),
    maskedDiv,
    buildEntitiesTable(),
  ]);
  return left;
}

function buildEntitiesTable(): HTMLElement {
  const wrap = el("div", {});
  wrap.appendChild(el("h3", {}, [`Detected entities (${state.lastSpans.length})`]));
  if (state.lastSpans.length === 0) {
    wrap.appendChild(el("div", { class: "note" }, ["No entities yet."]));
    return wrap;
  }
  const table = el("table", { class: "entities-table", "data-testid": "entities-table" });
  table.appendChild(
    el("thead", {}, [
      el("tr", {}, [
        el("th", {}, ["Label"]),
        el("th", {}, ["Score"]),
        el("th", {}, ["Word"]),
      ]),
    ]),
  );
  const tbody = el("tbody", {});
  for (const s of state.lastSpans) {
    tbody.appendChild(
      el("tr", {}, [
        el("td", { class: "label-cell" }, [
          el("span", { class: `pii pii-${s.entity_group}` }, [s.entity_group]),
        ]),
        el("td", {}, [(s.score * 100).toFixed(2) + "%"]),
        el("td", { class: "word-cell" }, [state.text.slice(s.start, s.end) || s.word]),
      ]),
    );
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

function buildRight(): HTMLElement {
  const panel = el("section", { class: "panel", "data-testid": "diagnostics-panel" });
  panel.appendChild(el("h2", {}, ["Diagnostics"]));
  panel.appendChild(buildSystemHealth());
  panel.appendChild(buildModelMetrics());
  panel.appendChild(buildEventLog());
  panel.appendChild(buildCopyButton());
  return panel;
}

function buildSystemHealth(): HTMLElement {
  const wrap = el("div", {});
  wrap.appendChild(el("h3", {}, ["System Health"]));
  const r = state.runtime;
  const m = state.memory;
  const grid = el("div", { class: "kv" });
  grid.appendChild(kv("device", el("span", { "data-testid": "device" }, [r?.device ?? "—"])));
  grid.appendChild(kv("dtype", el("span", { "data-testid": "dtype" }, [r?.dtype ?? "—"])));
  grid.appendChild(kv("WebGPU", r?.hasWebGPU ? "yes" : "no"));
  grid.appendChild(kv("shader-f16", r?.hasShaderF16 ? "yes" : "no"));
  grid.appendChild(kv(
    "GPU adapter",
    r?.adapterInfo
      ? `${r.adapterInfo.vendor ?? "?"} · ${r.adapterInfo.architecture ?? "?"} · ${r.adapterInfo.description ?? r.adapterInfo.device ?? ""}`.trim()
      : "—",
  ));
  grid.appendChild(kv("WASM threads", r?.hasWasmThreads ? "yes" : "no"));
  grid.appendChild(kv("WASM SIMD", r?.hasWasmSimd ? "yes" : "no"));
  grid.appendChild(kv("crossOriginIsolated", r?.crossOriginIsolated ? "yes" : "no"));
  grid.appendChild(kv("JS heap (used / total)", `${fmtBytes(m?.jsHeapUsedBytes ?? null)} / ${fmtBytes(m?.jsHeapTotalBytes ?? null)}`));
  grid.appendChild(kv("Browser RAM (UA)", fmtBytes(m?.browserBytes ?? null)));
  wrap.appendChild(grid);
  return wrap;
}

function buildModelMetrics(): HTMLElement {
  const wrap = el("div", {});
  wrap.appendChild(el("h3", {}, ["Model Metrics — Load"]));
  const load = state.load;
  const downloadMBs =
    load.downloadMs && load.downloadMs > 0 && load.downloadBytes
      ? load.downloadBytes / 1024 / 1024 / (load.downloadMs / 1000)
      : null;
  const heapDelta =
    load.heapBeforeBytes != null && load.heapAfterFirstInferBytes != null
      ? load.heapAfterFirstInferBytes - load.heapBeforeBytes
      : null;
  const dl = el("div", { class: "kv" });
  dl.appendChild(kv("Downloaded", el("span", { "data-testid": "load-downloaded" }, [fmtBytes(load.downloadBytes)])));
  dl.appendChild(kv("Download time", fmtSecs(load.downloadMs)));
  dl.appendChild(kv("Throughput", downloadMBs == null ? "n/a" : `${downloadMBs.toFixed(1)} MB/s`));
  dl.appendChild(kv("Pipeline init", fmtMs(load.initMs)));
  dl.appendChild(kv("Total load time", fmtSecs(load.totalLoadMs)));
  dl.appendChild(kv("First inference", fmtMs(load.firstInferenceMs)));
  dl.appendChild(kv("Est. weight footprint", fmtBytes(load.estimatedModelMemBytes)));
  dl.appendChild(kv("JS-heap delta after load", fmtBytes(heapDelta)));
  wrap.appendChild(dl);

  // download progress bar
  if (state.status === "downloading" && state.download.total > 0) {
    const prog = el("div", { class: "progress" }, [
      el("div", { class: "bar" }),
    ]);
    const pct = (state.download.loaded / state.download.total) * 100;
    (prog.firstChild as HTMLElement).style.width = `${pct.toFixed(1)}%`;
    wrap.appendChild(prog);
    wrap.appendChild(
      el("div", { class: "note" }, [
        `${state.download.file} · ${fmtBytes(state.download.loaded)} / ${fmtBytes(state.download.total)}`,
      ]),
    );
  }

  wrap.appendChild(el("h3", {}, ["Model Metrics — Inference (last)"]));
  const m = state.metrics;
  const last = m.total.last();
  const lastTok = m.tokenize.last();
  const lastInf = m.infer.last();
  const lastInputTokens = m.inputTokens.last();
  const inf = el("div", { class: "kv" });
  inf.appendChild(kv("Tokenize", fmtMs(lastTok)));
  inf.appendChild(kv("Forward + decode", fmtMs(lastInf)));
  inf.appendChild(kv("Total", el("span", { "data-testid": "last-infer-ms" }, [fmtMs(last)])));
  inf.appendChild(kv("Input tokens", lastInputTokens != null ? String(Math.round(lastInputTokens)) : "n/a"));
  inf.appendChild(kv("Output spans", String(state.lastSpans.length)));
  wrap.appendChild(inf);

  wrap.appendChild(el("h3", {}, ["Model Metrics — Inference (rolling)"]));
  const r = el("div", { class: "kv" });
  r.appendChild(kv("p50", fmtMs(m.total.percentile(50))));
  r.appendChild(kv("p95", fmtMs(m.total.percentile(95))));
  r.appendChild(kv("mean", fmtMs(m.total.mean())));
  r.appendChild(kv("max", fmtMs(m.total.max())));
  r.appendChild(kv("samples", String(m.total.size())));
  const tps = m.tokensPerSec();
  r.appendChild(kv("tokens / sec", tps == null ? "n/a" : tps.toFixed(1)));
  r.appendChild(kv("inferences / min", String(m.inferencesPerMin())));
  wrap.appendChild(r);

  // sparkline of last latencies
  const vs = m.total.values();
  if (vs.length >= 2) {
    const W = 280;
    const H = 36;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "sparkline");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("preserveAspectRatio", "none");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", sparklinePath(vs, W, H));
    svg.appendChild(path);
    wrap.appendChild(svg);
  }

  return wrap;
}

function buildEventLog(): HTMLElement {
  const wrap = el("div", {});
  wrap.appendChild(el("h3", {}, ["Event log"]));
  const list = el("div", { class: "event-log", "data-testid": "event-log" });
  const entries = state.log.list();
  if (entries.length === 0) {
    list.appendChild(el("div", { class: "ev kind-ui" }, ["(empty — load the model to see events)"]));
  } else {
    for (const e of entries.slice().reverse()) {
      const ts = (e.tWallMs / 1000).toFixed(2);
      list.appendChild(el("div", { class: `ev kind-${e.kind}` }, [`[t+${ts}s] ${e.message}`]));
    }
  }
  wrap.appendChild(list);
  return wrap;
}

function buildCopyButton(): HTMLElement {
  const btn = el("button", { class: "btn", style: "margin-top:10px;", "data-testid": "btn-copy-md" }, [
    "Copy diagnostics as Markdown",
  ]);
  btn.addEventListener("click", async () => {
    const md = renderDiagnosticsMarkdown();
    try {
      await navigator.clipboard.writeText(md);
      state.log.push("ui", "Diagnostics copied to clipboard");
    } catch {
      state.log.push("error", "Clipboard write failed — printing to console");
      // eslint-disable-next-line no-console
      console.log(md);
    }
    scheduleRender();
  });
  return btn;
}

function renderDiagnosticsMarkdown(): string {
  const r = state.runtime;
  const m = state.memory;
  const L = state.load;
  const M = state.metrics;
  const lines: string[] = [];
  lines.push(`# openai/privacy-filter — browser diagnostics`);
  lines.push("");
  lines.push(`## System Health`);
  lines.push(`- device: \`${r?.device ?? "—"}\``);
  lines.push(`- dtype: \`${r?.dtype ?? "—"}\``);
  lines.push(`- WebGPU: ${r?.hasWebGPU ? "yes" : "no"}`);
  lines.push(`- shader-f16: ${r?.hasShaderF16 ? "yes" : "no"}`);
  if (r?.adapterInfo) {
    lines.push(`- adapter: ${r.adapterInfo.vendor ?? "?"} · ${r.adapterInfo.architecture ?? "?"} · ${r.adapterInfo.description ?? r.adapterInfo.device ?? ""}`);
  }
  lines.push(`- WASM threads: ${r?.hasWasmThreads ? "yes" : "no"} · SIMD: ${r?.hasWasmSimd ? "yes" : "no"}`);
  lines.push(`- crossOriginIsolated: ${r?.crossOriginIsolated ? "yes" : "no"}`);
  lines.push(`- JS heap: ${fmtBytes(m?.jsHeapUsedBytes ?? null)} / ${fmtBytes(m?.jsHeapTotalBytes ?? null)}`);
  lines.push(`- browser RAM: ${fmtBytes(m?.browserBytes ?? null)}`);
  lines.push("");
  lines.push(`## Load`);
  lines.push(`- downloaded: ${fmtBytes(L.downloadBytes)} in ${fmtSecs(L.downloadMs)}`);
  lines.push(`- pipeline init: ${fmtMs(L.initMs)}`);
  lines.push(`- total load: ${fmtSecs(L.totalLoadMs)}`);
  lines.push(`- first inference: ${fmtMs(L.firstInferenceMs)}`);
  lines.push(`- est. weight footprint: ${fmtBytes(L.estimatedModelMemBytes)}`);
  if (L.heapBeforeBytes != null && L.heapAfterFirstInferBytes != null) {
    lines.push(`- JS heap delta: ${fmtBytes(L.heapAfterFirstInferBytes - L.heapBeforeBytes)}`);
  }
  lines.push("");
  lines.push(`## Inference (rolling)`);
  lines.push(`- last total: ${fmtMs(M.total.last())} (tokenize ${fmtMs(M.tokenize.last())} · forward+decode ${fmtMs(M.infer.last())})`);
  lines.push(`- p50: ${fmtMs(M.total.percentile(50))} · p95: ${fmtMs(M.total.percentile(95))} · mean: ${fmtMs(M.total.mean())} · max: ${fmtMs(M.total.max())}`);
  const tps = M.tokensPerSec();
  lines.push(`- throughput: ${tps == null ? "n/a" : tps.toFixed(1) + " tok/s"} · ${M.inferencesPerMin()} infer/min`);
  lines.push("");
  lines.push(`## Recent events`);
  for (const e of state.log.list().slice(-10)) {
    lines.push(`- [t+${(e.tWallMs / 1000).toFixed(2)}s] ${e.kind} — ${e.message}`);
  }
  return lines.join("\n");
}

function buildFooter(): HTMLElement {
  return el("footer", { class: "app-footer" }, [
    "Model: openai/privacy-filter (Apache-2.0). Inference is 100% local — text never leaves the tab.",
  ]);
}

// --- bootstrap ---

async function bootstrap() {
  // Show the runtime probe on the panel even before the user clicks Load model.
  const probe = await pickRuntime();
  state.runtime = probe;
  state.memory = await snapshotMemory();
  state.log.push("ui", `Probe: ${probe.device}/${probe.dtype} — ${probe.reason}`);

  // Honor URL flags so Playwright can drive WebGPU/WASM modes deterministically.
  const params = new URLSearchParams(location.search);
  const force = params.get("device");
  if (force === "wasm" || force === "webgpu") state.forceDevice = force;
  if (params.get("autoload") === "1") {
    // Used by the e2e tests to skip the click.
    queueMicrotask(() => startWorker(state.forceDevice));
  }

  // periodic memory polling
  window.setInterval(async () => {
    state.memory = await snapshotMemory();
    scheduleRender();
  }, 2000);

  scheduleRender();
}

void bootstrap();
