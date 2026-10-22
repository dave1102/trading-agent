@echo off
rem Starts the meme coin scanner (paper only: no wallet, no keys, no real trades).
rem Used by the shortcut in the Windows Startup folder; you can also double-click it.
rem Output goes to data\memecoins\scanner.log. Close this window to stop the scanner.
title Meme coin scanner (paper only)
cd /d "%~dp0"
if not exist "data\memecoins" mkdir "data\memecoins"
echo Meme coin scanner running. Log: %~dp0data\memecoins\scanner.log
echo Close this window to stop it.
".venv\Scripts\python.exe" -u -m trading_agent.memescan run --every 15 >> "data\memecoins\scanner.log" 2>&1
