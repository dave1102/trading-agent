@echo off
rem Shows whether the background meme scanner is running, and its latest log lines.
cd /d "%~dp0"
powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' OR Name='python.exe'\" | Where-Object { $_.CommandLine -like '*trading_agent.memescan run*' }; if ($p) { 'Scanner is RUNNING (started ' + ($p | Sort-Object CreationDate | Select-Object -First 1).CreationDate + ')' } else { 'Scanner is NOT running. Double-click start_scanner.cmd or restart the PC.' }; ''; 'Latest log lines:'; Get-Content 'data\memecoins\scanner.log' -Tail 15 -ErrorAction SilentlyContinue"
echo.
pause
