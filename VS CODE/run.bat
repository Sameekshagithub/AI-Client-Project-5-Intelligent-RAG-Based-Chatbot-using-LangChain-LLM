@echo off
REM One-command start (Windows): creates .venv, installs packages, launches the app.
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt
if not exist .env copy .env.example .env
streamlit run app.py
