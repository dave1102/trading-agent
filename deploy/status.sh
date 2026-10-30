#!/usr/bin/env bash
# Status of the meme coin scanner container on the server (called by scanner_status.cmd on the PC).
if [ "$(docker inspect -f '{{.State.Running}}' memescan 2>/dev/null)" = "true" ]; then
  echo "Scanner is RUNNING on the server ($(docker inspect -f '{{.State.StartedAt}}' memescan | cut -c1-16) UTC start)"
  docker stats --no-stream --format 'Using: CPU {{.CPUPerc}}, memory {{.MemUsage}}' memescan
else
  echo "Scanner is NOT running. Start it with start_scanner.cmd."
fi
if [ "$(docker inspect -f '{{.State.Running}}' trendpaper 2>/dev/null)" = "true" ]; then
  echo "Trend paper test is RUNNING too"
else
  echo "Trend paper test is NOT running."
fi
echo
echo "Latest scanner log lines:"
tail -n 15 /opt/trading-agent/data/memecoins/scanner.log 2>/dev/null
echo
echo "Latest trend paper log lines:"
tail -n 3 /opt/trading-agent/data/trend/trend.log 2>/dev/null
