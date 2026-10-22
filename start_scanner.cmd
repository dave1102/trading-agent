@echo off
rem Starts the meme coin scanner in the background with no window (paper only:
rem no wallet, no keys, no real trades). The Startup shortcut runs the same thing.
rem Check it with scanner_status.cmd; stop it with stop_scanner.cmd.
rem Output goes to data\memecoins\scanner.log. Starting it twice is safe: the second copy exits.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m trading_agent.memescan run --every 15 --log "data\memecoins\scanner.log"
echo Scanner started in the background. Use scanner_status.cmd to check it.
timeout /t 3 >nul
