@echo off
rem Starts the meme scanner on the server again after stop_scanner.cmd (paper only:
rem no wallet, no keys, no real trades). Check it with scanner_status.cmd.
ssh -o ConnectTimeout=20 vps "docker start memescan >/dev/null && echo Scanner started on the server."
echo.
pause
