@echo off
rem Shows whether the meme scanner is running on the server, and its latest log lines.
rem Needs the "vps" SSH shortcut on this PC (see README). If your Wi-Fi blocks SSH, try a hotspot.
ssh -o ConnectTimeout=20 vps "bash /opt/trading-agent/app/deploy/status.sh"
echo.
pause
