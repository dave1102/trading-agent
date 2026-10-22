@echo off
rem Stops the background meme scanner. It starts again at your next sign-in
rem (or double-click start_scanner.cmd).
powershell -NoProfile -Command "$p = Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' OR Name='python.exe'\" | Where-Object { $_.CommandLine -like '*trading_agent.memescan run*' }; if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; 'Scanner stopped.' } else { 'Scanner was not running.' }"
echo.
pause
