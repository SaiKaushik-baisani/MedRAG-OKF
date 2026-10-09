@echo off
REM One-click start for Windows. Creates a virtual environment on first run.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
python app.py
pause
