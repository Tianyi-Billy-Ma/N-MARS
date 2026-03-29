#!/bin/bash
# ---------------------------------------------------------------------------
# Install Flash Attention 2 into the project .venv using uv.
#
# Strategy:
#   1. Auto-detect Python, PyTorch, CUDA, and CXX11 ABI from the active venv.
#   2. Query the Dao-AILab/flash-attention GitHub releases for a matching
#      pre-built wheel (avoids 15+ min source compilation).
#   3. If no matching wheel exists, fall back to building from source via
#      `uv pip install flash-attn --no-build-isolation`.
#
# Usage:
#   bash scripts/env/install_flash_attn.sh [--tag TAG] [--version VERSION]
#
# Examples:
#   bash scripts/env/install_flash_attn.sh              # latest release
#   bash scripts/env/install_flash_attn.sh --tag v2.8.3 # specific tag
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVER="${SCRIPT_DIR}/_resolve_flash_attn_wheel.py"

# ── Activate venv ──────────────────────────────────────────────────────────
echo "=> Activating the local .venv..."
if [ -f "./.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ./.venv/bin/activate
else
  echo "❌ Error: .venv not found. Please run 'uv sync' first."
  exit 1
fi

# ── Sanity checks ─────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "❌ Error: 'uv' is not installed. See https://docs.astral.sh/uv/"
  exit 1
fi

if ! python -c "import torch" &>/dev/null; then
  echo "❌ Error: PyTorch is not installed in the venv. Run 'uv sync' first."
  exit 1
fi

# ── Resolve wheel URL ─────────────────────────────────────────────────────
echo "=> Detecting environment and resolving pre-built wheel..."

# Run the resolver: stderr goes to the terminal, stdout is captured.
# We use a temp file so we can capture both the URL and the exit code cleanly.
TMPFILE=$(mktemp)
RESOLVE_EXIT=0
python "$RESOLVER" "$@" >"$TMPFILE" 2>&1 || RESOLVE_EXIT=$?

# Everything except the last line is diagnostic output (goes to stderr).
# The last line on success is the wheel URL.
if [ "$RESOLVE_EXIT" -eq 0 ]; then
  # Print all diagnostic lines to stderr, capture only the last line (the URL)
  WHEEL_URL=$(tail -n1 "$TMPFILE")
  head -n-1 "$TMPFILE" >&2
else
  # Print all output as diagnostics
  cat "$TMPFILE" >&2
  WHEEL_URL=""
fi
rm -f "$TMPFILE"

if [ "$RESOLVE_EXIT" -eq 2 ]; then
  echo "❌ Environment detection failed (see above). Cannot continue."
  exit 1
fi

# ── Install ───────────────────────────────────────────────────────────────
if [ -n "$WHEEL_URL" ]; then
  echo ""
  echo "=> Installing pre-built wheel..."
  echo "   URL: ${WHEEL_URL}"
  echo ""
  uv pip install "$WHEEL_URL"
  echo ""
  echo "✅ Flash Attention installed from pre-built wheel!"
else
  echo ""
  echo "⚠️  No matching pre-built wheel found."
  echo "=> Falling back to source build (this may take 15+ minutes)..."
  echo ""

  # flash-attn's setup.py imports torch at build time, so we need
  # --no-build-isolation to let it see the venv's torch.
  uv pip install flash-attn --no-build-isolation
  echo ""
  echo "✅ Flash Attention installed from source!"
fi

# ── Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "=> Verifying installation..."
python -c "
import flash_attn
print(f'   flash-attn version: {flash_attn.__version__}')
"
echo "🎉 Done."
