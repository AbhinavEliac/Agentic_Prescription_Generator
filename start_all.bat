@echo off
title RxAgent Full Stack Launcher
echo ========================================================
echo Launching RxAgent Full Stack System...
echo ========================================================

start "RxAgent Python Backend (:8080)" "%~dp0start_backend.bat"
timeout /t 2 /nobreak >nul
start "RxAgent Node.js Studio (:3000)" "%~dp0start_frontend.bat"

echo.
echo Both servers are launching in separate windows!
echo Open your browser at: http://localhost:3000
echo.
pause
