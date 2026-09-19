@echo off
title Warehouse Forecast - web server

echo ============================================
echo   Starting the web server
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Install it from python.org and check "Add python.exe to PATH".
    pause
    exit /b 1
)

if not exist "models\global\meta.json" (
    echo [WARNING] No trained model found in models\global
    echo The site will still open, but forecasts will use simple statistics.
    echo To train the model first, run: run_project.bat
    echo.
)

echo Opening http://localhost:8000 in your browser...
echo Press Ctrl+C in this window to stop the server.
echo.

start "" http://localhost:8000
python scripts\serve.py --port 8000

pause
