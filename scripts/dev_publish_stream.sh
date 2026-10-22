#!/usr/bin/env bash
# Publish a synthetic test pattern to MediaMTX over SRT — useful to exercise the
# pipeline without a phone attached.
#
#   ./scripts/dev_publish_stream.sh [host]      # default host: 127.0.0.1
set -euo pipefail

HOST="${1:-${DAPHNIA_STREAM_HOST:-127.0.0.1}}"
PORT="${DAPHNIA_SRT_PORT:-8890}"
STREAM="${DAPHNIA_STREAM_PATH:-live/cam}"
URL="srt://${HOST}:${PORT}?streamid=publish:${STREAM}"

for tool in ffmpeg srt-live-transmit; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: $tool not found — install it with 'brew install ffmpeg srt'" >&2
        exit 1
    fi
done

echo "publishing test pattern -> $URL  (Ctrl-C to stop)"
ffmpeg -hide_banner -loglevel warning -re \
    -f lavfi -i "testsrc=size=1280x720:rate=30" \
    -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -g 30 \
    -f mpegts - \
| srt-live-transmit -ll warning file://con "$URL"
