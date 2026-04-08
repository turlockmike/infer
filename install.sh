#!/bin/sh
set -e

INSTALL_DIR="${INFER_INSTALL_DIR:-/usr/local/bin}"
BIN="$INSTALL_DIR/infer"
RAW="https://raw.githubusercontent.com/turlockmike/infer/main/infer"

echo "Installing infer to $BIN..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required" >&2; exit 1
fi

pip install --quiet openai

if [ -w "$INSTALL_DIR" ]; then
  curl -fsSL "$RAW" -o "$BIN"
else
  curl -fsSL "$RAW" | sudo tee "$BIN" > /dev/null
fi

chmod +x "$BIN" 2>/dev/null || sudo chmod +x "$BIN"

echo "Installed: $(infer --version 2>/dev/null || echo ok)"
echo ""
echo "Configure a provider:"
echo "  infer config set url http://localhost:11434/v1"
echo "  infer config set model gemma4:latest"
echo "  infer config set api_key ollama"
