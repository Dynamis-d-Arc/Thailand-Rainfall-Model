@echo off
rem Launches the Rainwatch dashboard server and opens it in the browser.
cd /d "%~dp0.."
start "" http://localhost:8901
".venv\Scripts\python.exe" "Dashboard\rainwatch_server.py"
pause
