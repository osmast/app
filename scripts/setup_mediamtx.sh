#!/usr/bin/env bash
# Install the repository MediaMTX configuration system-wide (macOS / Homebrew).
#
#   ./scripts/setup_mediamtx.sh
#
# The existing configuration is backed up with a timestamp suffix first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/config/mediamtx.yml"
CONFIG_DIR="${MEDIAMTX_CONFIG_DIR:-/opt/homebrew/etc/mediamtx}"
DEST="$CONFIG_DIR/mediamtx.yml"
STAMP="$(date +%Y%m%d-%H%M%S)"

if ! command -v mediamtx >/dev/null 2>&1; then
    echo "error: mediamtx not found — install it with 'brew install mediamtx'" >&2
    exit 1
fi

if [[ ! -f "$SRC" ]]; then
    echo "error: missing $SRC" >&2
    exit 1
fi

mkdir -p "$CONFIG_DIR"

if [[ -f "$DEST" ]]; then
    cp "$DEST" "$DEST.$STAMP.bak"
    echo "backed up existing config -> $DEST.$STAMP.bak"
fi

cp "$SRC" "$DEST"
echo "installed $SRC -> $DEST"
echo
echo "Restart the server to apply the new configuration:"
echo "  brew services restart mediamtx     # when managed by brew"
echo "  pkill -f mediamtx; mediamtx        # when running it directly"
