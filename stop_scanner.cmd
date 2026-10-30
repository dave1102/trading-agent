@echo off
rem Stops the meme scanner on the server. Start it again with start_scanner.cmd.
rem (It also starts again by itself if the server reboots.)
ssh -o ConnectTimeout=20 vps "docker stop memescan >/dev/null && echo Scanner stopped."
echo.
pause
