#!/bin/sh
set -e

REPO="turlockmike/infer"
INSTALL_DIR="${INFER_INSTALL_DIR:-/usr/local/bin}"
BIN="$INSTALL_DIR/infer"

# Detect platform
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  ARCH="x64" ;;
  arm64|aarch64) ARCH="arm64" ;;
  *) echo "error: unsupported arch $ARCH" >&2; exit 1 ;;
esac

ARTIFACT="infer-${OS}-${ARCH}"
TAG=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | cut -d'"' -f4)

if [ -z "$TAG" ]; then
  echo "error: could not find latest release" >&2; exit 1
fi

URL="https://github.com/${REPO}/releases/download/${TAG}/${ARTIFACT}"

echo "Installing infer ${TAG} (${ARTIFACT}) to ${BIN}..."

if [ -w "$INSTALL_DIR" ]; then
  curl -fsSL "$URL" -o "$BIN"
  chmod +x "$BIN"
else
  curl -fsSL "$URL" | sudo tee "$BIN" > /dev/null
  sudo chmod +x "$BIN"
fi

echo "Done."
echo ""
echo "Quick start:"
echo "  export INFER_API_KEY=sk-...            # OpenAI"
echo "  export INFER_URL=https://api.openai.com/v1"
echo "  export INFER_MODEL=gpt-4o"
echo ""
echo "  infer \"what directory am i in\""
echo "  cat crash.log | infer \"why did this fail\""
