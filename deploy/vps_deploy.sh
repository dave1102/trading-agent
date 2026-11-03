#!/usr/bin/env bash
# Install or update the paper-trading containers on the server:
#   memescan    - new meme coin scanner
#   trendpaper  - forward paper test of the multi-coin trend strategy
#   memelab     - downloads 1-minute candles every 6 hours (rule lab and forward tests)
# Run on the server as root:  bash /opt/trading-agent/vps_deploy.sh
# Expects /opt/trading-agent/src.tar.gz (made on the PC with: git archive --format=tar.gz).
# Touches nothing outside /opt/trading-agent except these three containers and their image.
set -euo pipefail

BASE=/opt/trading-agent
mkdir -p "$BASE/data/memecoins" "$BASE/data/trend"
rm -rf "$BASE/app" && mkdir -p "$BASE/app"
tar -xzf "$BASE/src.tar.gz" -C "$BASE/app"

docker build --quiet -t memescan:latest "$BASE/app"
chown -R 10001:10001 "$BASE/data"   # the containers' unprivileged user

docker rm -f memescan >/dev/null 2>&1 || true
docker run -d --name memescan \
  --restart unless-stopped \
  --memory 300m --cpus 0.5 \
  -v "$BASE/data:/app/data" \
  memescan:latest >/dev/null

docker rm -f trendpaper >/dev/null 2>&1 || true
docker run -d --name trendpaper \
  --restart unless-stopped \
  --memory 300m --cpus 0.5 \
  -v "$BASE/data:/app/data" \
  memescan:latest \
  python -u -m trading_agent.trendpaper run --log data/trend/trend.log >/dev/null

docker rm -f memelab >/dev/null 2>&1 || true
docker run -d --name memelab \
  --restart unless-stopped \
  --memory 300m --cpus 0.5 \
  -v "$BASE/data:/app/data" \
  memescan:latest \
  python -u -m trading_agent.memelab fetch --every 6 --log data/memecoins/lab.log >/dev/null

sleep 3
docker ps --filter name=memescan --filter name=trendpaper --filter name=memelab --format 'container {{.Names}}: {{.Status}}'
