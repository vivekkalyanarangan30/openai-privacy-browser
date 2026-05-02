#!/usr/bin/env bash
# Build the Vite demo and push it to a HuggingFace Space (Static SDK).
#
# Prereqs (one-time):
#   pip install -U "huggingface_hub[cli]"
#   hf auth login          # paste a write token from https://huggingface.co/settings/tokens
#
# Usage:
#   scripts/deploy-hf-space.sh <user-or-org>/<space-name>
#   scripts/deploy-hf-space.sh vivekkalyanarangan/privacy-filter-browser
#
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <user-or-org>/<space-name>" >&2
  exit 1
fi
REPO="$1"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/4] Building production bundle..."
npm run build

echo "[2/4] Writing HuggingFace Space frontmatter into dist/README.md..."
cat > dist/README.md <<EOF
---
title: openai/privacy-filter — browser demo
emoji: 🔒
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: apache-2.0
short_description: Mask PII in your browser. 100% local inference.
---

A live browser demo of [\`openai/privacy-filter\`](https://huggingface.co/openai/privacy-filter)
running entirely client-side via Transformers.js v4 + ONNX Runtime Web (WebGPU).
See the right-hand diagnostics panel for live load, latency and throughput numbers.

Source: <https://github.com/${HF_SOURCE_REPO:-vivekkalyanarangan30/openai-privacy-browser}>
EOF

# .gitattributes — mark the bundled WASM blob as LFS so the push doesn't fail
cat > dist/.gitattributes <<'EOF'
*.wasm filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
EOF

echo "[3/4] Ensuring Space exists (creates it if missing)..."
hf repo create "$REPO" --repo-type space --space_sdk static --exist-ok

echo "[4/4] Uploading dist/ to https://huggingface.co/spaces/$REPO ..."
hf upload --repo-type space "$REPO" dist/ . --commit-message "deploy: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo
echo "✅ Done."
echo "   Space:  https://huggingface.co/spaces/$REPO"
echo "   App:    https://${REPO/\//-}.static.hf.space"
