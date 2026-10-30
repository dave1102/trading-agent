#!/usr/bin/env bash
# Status of the meme coin scanner container on the server (called by scanner_status.cmd on the PC).
if [ "$(docker inspect -f '{{.State.Running}}' memescan 2>/dev/null)" = "true" ]; then
  echo "Scanner is RUNNING on the server ($(docker inspect -f '{{.State.StartedAt}}' memescan | cut -c1-16) UTC start)"
  docker stats --no-stream --format 'Using: CPU {{.CPUPerc}}, memory {{.MemUsage}}' memescan
else
  echo "Scanner is NOT running. Start it with start_scanner.cmd."
fi
echo
echo "Latest log lines:"
tail -n 15 /opt/trading-agent/data/memecoins/scanner.log 2>/dev/null
