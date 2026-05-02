// spans → DocumentFragment renderer.
// Walks the input text and replaces each detected span with a styled chip.

export type Span = {
  start: number;
  end: number;
  entity_group: string;
  word: string;
  score: number;
};

const KNOWN_LABELS = new Set([
  "private_person",
  "private_email",
  "private_phone",
  "private_address",
  "private_url",
  "private_date",
  "account_number",
  "secret",
]);

function shortLabel(label: string): string {
  if (label.startsWith("private_")) return label.slice("private_".length).toUpperCase();
  if (label === "account_number") return "ACCOUNT";
  if (label === "secret") return "SECRET";
  return label.toUpperCase();
}

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

// Some pipelines return spans where word starts with a leading space or BPE marker.
// We trust the start/end character offsets if present; otherwise we re-discover them.
export function normalizeSpans(text: string, spans: Span[]): Span[] {
  const out: Span[] = [];
  let cursor = 0;
  for (const raw of spans) {
    let { start, end, word } = raw;
    if (
      typeof start !== "number" ||
      typeof end !== "number" ||
      start < 0 ||
      end > text.length ||
      end <= start
    ) {
      // Fallback: search for the word from the cursor.
      const trimmed = (word ?? "").trim();
      if (!trimmed) continue;
      const found = text.indexOf(trimmed, cursor);
      if (found < 0) continue;
      start = found;
      end = found + trimmed.length;
    }
    start = clamp(start, 0, text.length);
    end = clamp(end, start, text.length);
    cursor = end;
    out.push({ ...raw, start, end });
  }
  // De-overlap by keeping the higher-score span when two collide.
  out.sort((a, b) => a.start - b.start || b.score - a.score);
  const dedup: Span[] = [];
  for (const s of out) {
    const last = dedup[dedup.length - 1];
    if (last && s.start < last.end) {
      if (s.score > last.score) dedup[dedup.length - 1] = s;
      continue;
    }
    dedup.push(s);
  }
  return dedup;
}

export function renderMasked(text: string, spans: Span[]): DocumentFragment {
  const frag = document.createDocumentFragment();
  const normalized = normalizeSpans(text, spans);
  let cursor = 0;
  for (const s of normalized) {
    if (s.start > cursor) {
      frag.appendChild(document.createTextNode(text.slice(cursor, s.start)));
    }
    const chip = document.createElement("span");
    const labelClass = KNOWN_LABELS.has(s.entity_group)
      ? `pii pii-${s.entity_group}`
      : "pii pii-unknown";
    chip.className = labelClass;
    chip.dataset.label = s.entity_group;
    chip.dataset.score = s.score.toFixed(4);
    chip.textContent = `[${shortLabel(s.entity_group)}]`;
    chip.title = `${s.entity_group} · ${(s.score * 100).toFixed(2)}% · "${text.slice(s.start, s.end)}"`;
    frag.appendChild(chip);
    cursor = s.end;
  }
  if (cursor < text.length) {
    frag.appendChild(document.createTextNode(text.slice(cursor)));
  }
  return frag;
}
