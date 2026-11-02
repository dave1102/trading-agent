#!/usr/bin/env bash
# Download 1-minute candles for the meme rule lab, in a separate one-off container.
# Leaves the memescan and trendpaper containers running as they are.
# Run on the server as root:  bash /opt/trading-agent/lab_fetch.sh
# Expects /opt/trading-agent/src.tar.gz (made on the PC with: git archive --format=tar.gz).
set -euo pipefail

BASE=/opt/trading-agent
rm -rf "$BASE/app" && mkdir -p "$BASE/app"
tar -xzf "$BASE/src.tar.gz" -C "$BASE/app"
docker build --quiet -t memescan:latest "$BASE/app"
chown -R 10001:10001 "$BASE/data"

docker rm -f memelab >/dev/null 2>&1 || true
docker run -d --name memelab \
  --memory 300m --cpus 0.5 \
  -v "$BASE/data:/app/data" \
  memescan:latest \
  python -u -m trading_agent.memelab fetch --log data/memecoins/lab.log >/dev/null

sleep 20
tail -n 3 "$BASE/data/memecoins/lab.log"
