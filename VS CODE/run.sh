#!/usr/bin/env bash
# One-command start (macOS / Linux): creates .venv, installs packages, launches the app.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env
streamlit run app.py
