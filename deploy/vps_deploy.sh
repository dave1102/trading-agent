#!/usr/bin/env bash
# Install or update the meme coin scanner on the server, inside its own Docker container.
# Run on the server as root:  bash /opt/trading-agent/vps_deploy.sh
# Expects /opt/trading-agent/src.tar.gz (made on the PC with: git archive --format=tar.gz).
# Touches nothing outside /opt/trading-agent except the "memescan" container and image.
set -euo pipefail

BASE=/opt/trading-agent
mkdir -p "$BASE/data/memecoins"
rm -rf "$BASE/app" && mkdir -p "$BASE/app"
tar -xzf "$BASE/src.tar.gz" -C "$BASE/app"

docker build --quiet -t memescan:latest "$BASE/app"
chown -R 10001:10001 "$BASE/data"   # the container's unprivileged user

docker rm -f memescan >/dev/null 2>&1 || true
docker run -d --name memescan \
  --restart unless-stopped \
  --memory 300m --cpus 0.5 \
  -v "$BASE/data:/app/data" \
  memescan:latest >/dev/null

sleep 3
docker ps --filter name=memescan --format 'container {{.Names}}: {{.Status}}'
