import { test, expect, chromium, type BrowserContext } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const SCREENSHOT_DIR = path.resolve("screenshots");
fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

// Persist IndexedDB across tests so the ~810 MB ONNX download is reused.
// Single profile because all tests run on WebGPU — the only viable path
// for openai/privacy-filter (its quantized ONNX variants use the
// `GatherBlockQuantized` op which is WebGPU-only in ORT-Web).
const PROFILE_ROOT = path.resolve(".pw-profiles");
fs.mkdirSync(PROFILE_ROOT, { recursive: true });

const PREVIEW_URL = "http://localhost:4173";
const READY_TIMEOUT = 13 * 60 * 1000;

const REQUIRED_LABELS = ["private_email", "private_person", "private_phone"] as const;

const PERF_REPORT_PATH = path.resolve("screenshots", "perf-report.json");
const accumulatedReport: Record<string, unknown> = {};
function persistReport() {
  fs.writeFileSync(PERF_REPORT_PATH, JSON.stringify(accumulatedReport, null, 2));
}

const WEBGPU_ARGS = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan,UseSkiaRenderer",
  "--use-angle=swiftshader-webgl",
  "--enable-webgpu-developer-features",
];

async function launchContext(profile: string, args: string[] = WEBGPU_ARGS): Promise<BrowserContext> {
  return await chromium.launchPersistentContext(path.join(PROFILE_ROOT, profile), {
    headless: true,
    args,
    viewport: { width: 1400, height: 900 },
  });
}

async function waitForReady(page: import("@playwright/test").Page) {
  const handle = await page.waitForFunction(
    () => {
      const pill = document.querySelector('[data-testid="status-pill"]');
      const t = pill?.textContent?.trim() ?? "";
      if (t.startsWith("Ready")) return "ready";
      if (t.startsWith("Error")) return "error";
      return false;
    },
    undefined,
    { timeout: READY_TIMEOUT },
  );
  const value = await handle.jsonValue();
  if (value === "error") {
    const msg = await page.locator('[data-testid="status-pill"]').textContent();
    throw new Error(`Worker reported error before Ready: ${msg ?? "(no detail)"}`);
  }
}

test.describe.serial("openai/privacy-filter browser demo", () => {
  test("A — end-to-end PII masking on WebGPU", async () => {
    const context = await launchContext("webgpu");
    try {
      const page = await context.newPage();
      page.on("console", (msg) => {
        if (msg.type() === "error") console.log(`[browser console error] ${msg.text()}`);
      });
      page.on("pageerror", (err) => console.log(`[browser pageerror] ${err.message}`));

      await page.goto(`${PREVIEW_URL}/?autoload=1&device=webgpu`, { timeout: 30_000 });
      await page.waitForSelector('[data-testid="status-pill"]', { timeout: 30_000 });
      await waitForReady(page);

      // Wait for the masked output to render at least 3 PII chips.
      await page.waitForFunction(
        () => (document.querySelector('[data-testid="masked-output"]')?.querySelectorAll(".pii").length || 0) >= 3,
        undefined,
        { timeout: 90_000 },
      );

      const presentLabels = await page.$$eval('[data-testid="masked-output"] .pii', (chips) =>
        chips.map((c) => c.getAttribute("data-label") || ""),
      );
      const device = (await page.locator('[data-testid="device"]').textContent())?.trim() || null;
      const dtype = (await page.locator('[data-testid="dtype"]').textContent())?.trim() || null;
      const lastInferMs = (await page.locator('[data-testid="last-infer-ms"]').textContent())?.trim() || null;

      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, "webgpu.png"),
        fullPage: true,
      });

      accumulatedReport.webgpu = { device, dtype, lastInferMs, presentLabels };
      persistReport();

      expect(device).toBe("webgpu");
      for (const label of REQUIRED_LABELS) {
        expect(presentLabels, `expected chip ${label}`).toContain(label);
      }
      expect(lastInferMs, "lastInferMs should be populated").not.toBeNull();
    } finally {
      await context.close();
    }
  });

  test("B — live debounce: typing re-triggers inference and re-masks", async () => {
    const context = await launchContext("webgpu");
    try {
      const page = await context.newPage();
      await page.goto(`${PREVIEW_URL}/?autoload=1&device=webgpu`, { timeout: 30_000 });
      await waitForReady(page);

      // Wait for the prefilled email's masked render to settle.
      await page.waitForFunction(
        () => (document.querySelector('[data-testid="masked-output"]')?.querySelectorAll(".pii").length || 0) > 0,
        undefined,
        { timeout: 90_000 },
      );
      const initialChipCount = await page.$$eval(
        '[data-testid="masked-output"] .pii',
        (chips) => chips.length,
      );
      expect(initialChipCount).toBeGreaterThan(0);

      // Replace textarea content with new text — should trigger debounced inference.
      await page.locator('[data-testid="input"]').fill(
        "Please email me at second.email@example.org so I can confirm. — Alice Walker",
      );

      // The new content has private_email + private_person but no phone/address/etc.
      await page.waitForFunction(
        () => {
          const el = document.querySelector('[data-testid="masked-output"]');
          if (!el) return false;
          return (
            el.querySelectorAll(".pii-private_email").length > 0 &&
            el.querySelectorAll(".pii-private_person").length > 0 &&
            // The original 12 chips from the prefilled email should be gone.
            el.querySelectorAll(".pii").length < 6
          );
        },
        undefined,
        { timeout: 60_000 },
      );

      const eventLines = await page.$$eval('[data-testid="event-log"] .ev', (els) =>
        els.map((e) => e.textContent ?? ""),
      );
      const dispatchCount = eventLines.filter((l) => l.includes("dispatch infer")).length;
      expect(dispatchCount, "expected at least 2 dispatched inferences").toBeGreaterThanOrEqual(2);

      await page.screenshot({
        path: path.join(SCREENSHOT_DIR, "webgpu-live.png"),
        fullPage: true,
      });

      accumulatedReport.live = { dispatchCount, eventLineCount: eventLines.length };
      persistReport();
    } finally {
      await context.close();
    }
  });
});
