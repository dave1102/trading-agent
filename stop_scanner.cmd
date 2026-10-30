@echo off
rem Stops the meme scanner and the trend paper test on the server.
rem Start them again with start_scanner.cmd.
ssh -o ConnectTimeout=20 vps "docker stop memescan trendpaper >/dev/null && echo Scanner and trend paper test stopped."
echo.
pause
