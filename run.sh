#!/usr/bin/env bash
# One-click start for macOS / Linux. Creates a virtual environment on first run.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
python app.py
