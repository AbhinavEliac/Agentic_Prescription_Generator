@echo off
title RxAgent Python FastAPI Backend
set "PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\Scripts;C:\Users\ADMIN\AppData\Local\Programs\Python\Python313;C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\Scripts;%PATH%"
cd /d "%~dp0rx_extractor_app"
echo ========================================================
echo Starting Agentic Prescription FastAPI Backend (Port 8080)...
echo ========================================================

if exist "env\Scripts\python.exe" (
    echo Checking Python dependencies...
    env\Scripts\python.exe -c "import fastapi" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [Setup] Installing missing fastapi dependency...
        env\Scripts\pip.exe install fastapi
    )
    echo [Backend] Launching FastAPI server on port 8080...
    env\Scripts\python.exe api_server.py
) else if exist "..\env\Scripts\python.exe" (
    echo Checking root venv dependencies...
    ..\env\Scripts\python.exe -c "import fastapi" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [Setup] Installing missing fastapi dependency...
        ..\env\Scripts\pip.exe install fastapi
    )
    echo [Backend] Launching FastAPI server on port 8080...
    ..\env\Scripts\python.exe api_server.py
) else (
    echo Using system Python...
    python api_server.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Backend encountered an error.
    pause
)


