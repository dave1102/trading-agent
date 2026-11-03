@echo off
rem Weekly meme scanner report (paper only). Run by the Windows scheduled task
rem "Meme scanner weekly report"; you can also double-click it any time.
rem Fetches the report from the server, saves data\memecoins\report_YYYY-MM-DD.txt
rem and opens it in Notepad.
cd /d "%~dp0"
if not exist "data\memecoins" mkdir "data\memecoins"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%d
set OUT=data\memecoins\report_%TODAY%.txt
ssh -o ConnectTimeout=30 vps "docker exec -e PYTHONIOENCODING=utf-8 trendpaper python -m trading_agent.trendpaper report; echo; echo ==========; echo; docker exec -e PYTHONIOENCODING=utf-8 memescan python -m trading_agent.memescan report; echo; echo ==========; echo; docker exec -e PYTHONIOENCODING=utf-8 memelab python -m trading_agent.memelab forward" > "%OUT%" 2>&1
if errorlevel 1 echo. >> "%OUT%" & echo Could not reach the server. If your Wi-Fi blocks SSH, try a phone hotspot and run weekly_report.cmd again. >> "%OUT%"
start "" notepad "%OUT%"
