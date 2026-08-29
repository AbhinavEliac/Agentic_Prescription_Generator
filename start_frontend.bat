@echo off
title RxAgent Node.js Frontend App
set "PATH=C:\Program Files\nodejs;%PATH%"
cd /d "%~dp0rx_node_app"
echo ========================================================
echo Starting RxAgent Node.js Studio App on http://localhost:3000...
echo ========================================================
node server.js
pause
