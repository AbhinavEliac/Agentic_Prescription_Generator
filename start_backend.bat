@echo off
title RxAgent Python FastAPI Backend
cd /d "%~dp0rx_extractor_app"
echo ========================================================
echo Starting Agentic Prescription FastAPI Backend (Port 8080)...
echo ========================================================
if exist "env\Scripts\python.exe" (
    env\Scripts\python.exe api_server.py
) else (
    python api_server.py
)
pause
