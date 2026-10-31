@echo off
rem Opens the ML Experiment Auditor web page at http://127.0.0.1:8000 (close this window to stop it).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The Python environment is missing. Set it up first, see "Setup" in README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m auditor.cli serve %*
pause
