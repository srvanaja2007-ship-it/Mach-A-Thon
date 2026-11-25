@echo off
setlocal
title ClaimPulse — Satellite Disaster Risk ^& Economic Decision Engine

cd /d "%~dp0"

echo ===================================================================
echo   ClaimPulse — Satellite Disaster Risk ^& Economic Decision Engine
echo ===================================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ to run ClaimPulse.
    pause
    exit /b 1
)

python app_launcher.py %*

if "%1"=="--cli" (
    pause
)
