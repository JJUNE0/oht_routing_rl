@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "PythonCode\run_headless.py" --mode stage2 %*
if errorlevel 1 pause
