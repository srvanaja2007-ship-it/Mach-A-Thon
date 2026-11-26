@echo off
cd /d "%~dp0"
echo Starting Orbit-to-Economy Unified Mission Control on http://127.0.0.1:9000 ...
start "" http://127.0.0.1:9000
python portal.py
