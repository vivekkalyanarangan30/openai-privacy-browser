// Rolling perf stats, load metrics, an event log, and memory polling.
// All state is owned by the main thread; the worker pushes perf events here.

export class RollingStats {
  private buf: number[] = [];
  constructor(private cap = 50) {}
  push(v: number) {
    this.buf.push(v);
    if (this.buf.length > this.cap) this.buf.shift();
  }
  size() {
    return this.buf.length;
  }
  last(): number | null {
    return this.buf.length ? this.buf[this.buf.length - 1] : null;
  }
  mean(): number | null {
    if (!this.buf.length) return null;
    return this.buf.reduce((a, b) => a + b, 0) / this.buf.length;
  }
  max(): number | null {
    if (!this.buf.length) return null;
    return Math.max(...this.buf);
  }
  percentile(p: number): number | null {
    if (!this.buf.length) return null;
    const sorted = [...this.buf].sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
    return sorted[idx];
  }
  values() {
    return [...this.buf];
  }
  reset() {
    this.buf = [];
  }
}

export type LoadMetrics = {
  downloadBytes: number;
  downloadMs: number | null;
  initMs: number | null;
  totalLoadMs: number | null;
  firstInferenceMs: number | null;
  estimatedModelMemBytes: number;
  heapBeforeBytes: number | null;
  heapAfterFirstInferBytes: number | null;
};

export function emptyLoadMetrics(): LoadMetrics {
  return {
    downloadBytes: 0,
    downloadMs: null,
    initMs: null,
    totalLoadMs: null,
    firstInferenceMs: null,
    estimatedModelMemBytes: 0,
    heapBeforeBytes: null,
    heapAfterFirstInferBytes: null,
  };
}

export type InferenceSample = {
  tokenizeMs: number;
  inferMs: number;
  totalMs: number;
  inputTokens: number;
  outputSpans: number;
};

export class InferenceMetrics {
  total = new RollingStats(50);
  tokenize = new RollingStats(50);
  infer = new RollingStats(50);
  inputTokens = new RollingStats(50);
  private timestamps: number[] = []; // wall-clock ms of each completed inference

  push(s: InferenceSample) {
    this.total.push(s.totalMs);
    this.tokenize.push(s.tokenizeMs);
    this.infer.push(s.inferMs);
    this.inputTokens.push(s.inputTokens);
    const now = performance.now();
    this.timestamps.push(now);
    const cutoff = now - 60_000;
    while (this.timestamps.length && this.timestamps[0] < cutoff) {
      this.timestamps.shift();
    }
  }

  inferencesPerMin(): number {
    return this.timestamps.length;
  }

  tokensPerSec(): number | null {
    const meanTokens = this.inputTokens.mean();
    const meanTotal = this.total.mean();
    if (meanTokens === null || meanTotal === null || meanTotal === 0) return null;
    return meanTokens / (meanTotal / 1000);
  }
}

export type EventEntry = {
  tWallMs: number;
  kind: "load" | "infer" | "ui" | "error";
  message: string;
};

export class EventLog {
  private entries: EventEntry[] = [];
  private subs = new Set<() => void>();
  constructor(private cap = 30, private startedAt = performance.now()) {}

  push(kind: EventEntry["kind"], message: string) {
    const e: EventEntry = { tWallMs: performance.now() - this.startedAt, kind, message };
    this.entries.push(e);
    if (this.entries.length > this.cap) this.entries.shift();
    for (const fn of this.subs) fn();
  }

  list(): EventEntry[] {
    return [...this.entries];
  }

  subscribe(fn: () => void): () => void {
    this.subs.add(fn);
    return () => this.subs.delete(fn);
  }
}

// Memory polling — main-thread only. Returns a snapshot.
export type MemorySnapshot = {
  jsHeapUsedBytes: number | null;
  jsHeapTotalBytes: number | null;
  browserBytes: number | null; // measureUserAgentSpecificMemory
  isolated: boolean;
};

interface PerfWithMemory extends Performance {
  memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number };
  measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>;
}

export async function snapshotMemory(): Promise<MemorySnapshot> {
  const perf = performance as PerfWithMemory;
  const isolated = (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
  const out: MemorySnapshot = {
    jsHeapUsedBytes: perf.memory ? perf.memory.usedJSHeapSize : null,
    jsHeapTotalBytes: perf.memory ? perf.memory.totalJSHeapSize : null,
    browserBytes: null,
    isolated,
  };
  if (isolated && typeof perf.measureUserAgentSpecificMemory === "function") {
    try {
      const r = await perf.measureUserAgentSpecificMemory();
      out.browserBytes = r.bytes;
    } catch {
      // some browsers throttle this; ignore
    }
  }
  return out;
}

export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return "n/a";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function fmtMs(n: number | null | undefined, digits = 1): string {
  if (n == null) return "n/a";
  if (n < 1) return `${n.toFixed(2)} ms`;
  return `${n.toFixed(digits)} ms`;
}

export function fmtSecs(ms: number | null | undefined): string {
  if (ms == null) return "n/a";
  return `${(ms / 1000).toFixed(2)} s`;
}
