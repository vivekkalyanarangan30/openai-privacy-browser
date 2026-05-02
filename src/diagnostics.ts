// Capability probe + auto-precision picker.
// Returns the device/dtype to hand to the Transformers.js pipeline,
// plus a snapshot of runtime features the UI surfaces in the System Health panel.

export type Device = "webgpu" | "wasm";
export type Dtype = "q4f16" | "q4" | "q8" | "fp16";

export type AdapterInfo = {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
};

export type RuntimeChoice = {
  device: Device;
  dtype: Dtype;
  reason: string;
  adapterInfo: AdapterInfo | null;
  hasShaderF16: boolean;
  hasWebGPU: boolean;
  hasWasmThreads: boolean;
  hasWasmSimd: boolean;
  crossOriginIsolated: boolean;
};

const SIMD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60,
  0x00, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
  0x41, 0x00, 0xfd, 0x0f, 0xfd, 0x62, 0x0b,
]);

async function probeWasmSimd(): Promise<boolean> {
  try {
    return WebAssembly.validate(SIMD_PROBE);
  } catch {
    return false;
  }
}

function probeWasmThreads(): boolean {
  if (typeof WebAssembly === "undefined") return false;
  if (typeof SharedArrayBuffer === "undefined") return false;
  try {
    new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
    return true;
  } catch {
    return false;
  }
}

async function probeWebGPU(): Promise<{
  hasWebGPU: boolean;
  hasShaderF16: boolean;
  adapterInfo: AdapterInfo | null;
}> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { hasWebGPU: false, hasShaderF16: false, adapterInfo: null };
  }
  try {
    const adapter = await navigator.gpu!.requestAdapter();
    if (!adapter) {
      return { hasWebGPU: false, hasShaderF16: false, adapterInfo: null };
    }
    const hasShaderF16 = adapter.features.has("shader-f16");
    const info = adapter.info;
    const adapterInfo: AdapterInfo = info
      ? {
          vendor: info.vendor || undefined,
          architecture: info.architecture || undefined,
          device: info.device || undefined,
          description: info.description || undefined,
        }
      : {};
    return { hasWebGPU: true, hasShaderF16, adapterInfo };
  } catch {
    return { hasWebGPU: false, hasShaderF16: false, adapterInfo: null };
  }
}

export async function pickRuntime(force?: Device): Promise<RuntimeChoice> {
  const [{ hasWebGPU, hasShaderF16, adapterInfo }, hasWasmSimd] = await Promise.all([
    probeWebGPU(),
    probeWasmSimd(),
  ]);
  const hasWasmThreads = probeWasmThreads();
  const isolated =
    typeof self !== "undefined" && (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;

  let device: Device;
  let dtype: Dtype;
  let reason: string;

  // Note: every *quantized* ONNX variant of openai/privacy-filter (q4, q4f16, q8)
  // uses the `GatherBlockQuantized` op for the embedding lookup, and ONNX Runtime
  // Web only implements that kernel in the WebGPU execution provider. The WASM EP
  // would error out at session creation. So WASM is an explicit *blocker* here —
  // we surface that, rather than silently fall back to a 2.8 GB fp16 download
  // that no reasonable user would wait through on CPU.
  if (force === "wasm") {
    device = "wasm";
    dtype = "fp16";
    reason =
      "Forced WASM. Note: the model's quantized variants need WebGPU; WASM would have to download fp16 (~2.8 GB) — expect very slow load and ~10 s/sentence inference.";
  } else if (force === "webgpu" && hasWebGPU) {
    device = "webgpu";
    dtype = hasShaderF16 ? "q4f16" : "q4";
    reason = hasShaderF16
      ? "Forced WebGPU; shader-f16 available → q4f16."
      : "Forced WebGPU; no shader-f16 → q4.";
  } else if (hasWebGPU) {
    device = "webgpu";
    dtype = hasShaderF16 ? "q4f16" : "q4";
    reason = hasShaderF16
      ? "WebGPU with shader-f16 → q4f16 (smallest, ~810 MB)."
      : "WebGPU without shader-f16 → q4 (~917 MB).";
  } else {
    device = "wasm";
    dtype = "fp16";
    reason =
      "No WebGPU adapter detected. This 1.5B-MoE model needs WebGPU (its quantized variants use the GatherBlockQuantized op, only implemented in ORT's WebGPU EP). WASM would require a ~2.8 GB fp16 download and is not recommended.";
  }

  return {
    device,
    dtype,
    reason,
    adapterInfo,
    hasShaderF16,
    hasWebGPU,
    hasWasmThreads,
    hasWasmSimd,
    crossOriginIsolated: isolated,
  };
}
