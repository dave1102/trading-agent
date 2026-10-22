@echo off
rem Weekly meme scanner report (paper only). Run by the Windows scheduled task
rem "Meme scanner weekly report"; you can also double-click it any time.
rem Saves data\memecoins\report_YYYY-MM-DD.txt and opens it in Notepad.
cd /d "%~dp0"
if not exist "data\memecoins" mkdir "data\memecoins"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%d
set OUT=data\memecoins\report_%TODAY%.txt
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -m trading_agent.memescan report > "%OUT%" 2>&1
start "" notepad "%OUT%"
