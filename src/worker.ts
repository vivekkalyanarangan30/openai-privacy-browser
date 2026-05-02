/// <reference lib="webworker" />
//
// Worker singleton. Loads the Transformers.js token-classification pipeline once,
// posts perf events for every load and inference. The main thread owns all UI state.
//

import { pipeline, env, AutoTokenizer } from "@huggingface/transformers";
import { pickRuntime, type Device, type RuntimeChoice } from "./diagnostics";

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_ID = "openai/privacy-filter";

type InitMsg = { type: "init"; forceDevice?: Device };
type InferMsg = { type: "infer"; id: number; text: string };
type InMsg = InitMsg | InferMsg;

type ProgressEvt = {
  status?: string;
  name?: string;
  file?: string;
  loaded?: number;
  total?: number;
  progress?: number;
};

let classifier: ((text: string, opts: Record<string, unknown>) => Promise<unknown>) | null = null;
let tokenizer: ((text: string) => { input_ids: { size?: number; data?: ArrayLike<number>; length?: number } }) | null =
  null;
let runtime: RuntimeChoice | null = null;
let firstInferenceMs: number | null = null;
let downloadStartMs: number | null = null;
let downloadEndMs: number | null = null;
let initStartMs: number | null = null;
let initEndMs: number | null = null;
let downloadedBytes = 0;
const fileTotals = new Map<string, number>();

function post(m: unknown) {
  (self as DedicatedWorkerGlobalScope).postMessage(m);
}

function onProgress(evt: ProgressEvt) {
  // Track total bytes downloaded across all files of the model.
  if (evt.file) {
    const total = typeof evt.total === "number" ? evt.total : 0;
    if (total > 0) fileTotals.set(evt.file, total);
  }
  if (evt.status === "download" || evt.status === "progress") {
    if (downloadStartMs === null) downloadStartMs = performance.now();
  }
  if (evt.status === "done" && evt.file) {
    // a file finished — accumulate its size
    const total = fileTotals.get(evt.file);
    if (total) downloadedBytes += total;
    downloadEndMs = performance.now();
    post({
      type: "perf",
      phase: "load-file",
      file: evt.file,
      bytes: total ?? null,
      tWallMs: performance.now(),
    });
  }
  // Forward raw progress events for the UI's progress bar.
  post({ type: "progress", evt });
}

async function init(forceDevice?: Device) {
  if (classifier) return;
  runtime = await pickRuntime(forceDevice);
  post({ type: "runtime", runtime });

  initStartMs = performance.now();
  try {
    // Load the tokenizer separately so we can time tokenize independently of inference.
    tokenizer = (await AutoTokenizer.from_pretrained(MODEL_ID, {
      progress_callback: onProgress,
    })) as unknown as typeof tokenizer;

    const pipelineOpts: Record<string, unknown> = {
      progress_callback: onProgress,
      device: runtime.device,
      dtype: runtime.dtype,
    };

    classifier = (await pipeline("token-classification", MODEL_ID, pipelineOpts)) as unknown as typeof classifier;

    initEndMs = performance.now();
    if (downloadEndMs === null) downloadEndMs = initStartMs;
    if (downloadStartMs === null) downloadStartMs = initStartMs;

    const downloadMs = Math.max(0, (downloadEndMs as number) - (downloadStartMs as number));
    const initMs = Math.max(0, (initEndMs as number) - (downloadEndMs as number));
    const totalLoadMs = (initEndMs as number) - (downloadStartMs as number);

    post({
      type: "perf",
      phase: "load",
      downloadMs,
      initMs,
      totalLoadMs,
      downloadedBytes,
    });
    post({ type: "ready" });
  } catch (e) {
    post({ type: "error", phase: "init", message: (e as Error)?.message ?? String(e) });
  }
}

function tokenCount(text: string): number {
  if (!tokenizer) return 0;
  try {
    const out = tokenizer(text);
    const ids = out?.input_ids as unknown;
    if (ids && typeof ids === "object") {
      const o = ids as { size?: number; data?: ArrayLike<number>; dims?: number[]; length?: number };
      if (typeof o.length === "number") return o.length;
      if (Array.isArray(o.data)) return (o.data as ArrayLike<number>).length;
      if (Array.isArray(o.dims)) return o.dims[o.dims.length - 1] ?? 0;
    }
  } catch {
    // ignore — fall through to 0
  }
  return 0;
}

async function infer(id: number, text: string) {
  if (!classifier) {
    post({ type: "error", phase: "infer", id, message: "Pipeline not initialized." });
    return;
  }
  const t0 = performance.now();
  const tokens = tokenCount(text);
  const t1 = performance.now();
  let spans: unknown = [];
  try {
    spans = await classifier(text, { aggregation_strategy: "simple" });
  } catch (e) {
    post({ type: "error", phase: "infer", id, message: (e as Error)?.message ?? String(e) });
    return;
  }
  const t2 = performance.now();
  if (firstInferenceMs === null) firstInferenceMs = t2 - t0;

  post({ type: "result", id, spans });
  post({
    type: "perf",
    phase: "infer",
    id,
    tokenizeMs: t1 - t0,
    inferMs: t2 - t1,
    totalMs: t2 - t0,
    inputTokens: tokens,
    outputSpans: Array.isArray(spans) ? spans.length : 0,
    isFirst: firstInferenceMs === t2 - t0,
  });
}

self.addEventListener("message", (e: MessageEvent<InMsg>) => {
  const msg = e.data;
  if (!msg) return;
  if (msg.type === "init") void init(msg.forceDevice);
  else if (msg.type === "infer") void infer(msg.id, msg.text);
});
