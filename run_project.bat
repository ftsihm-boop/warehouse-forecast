@echo off
title Warehouse Forecast - setup and run

echo ============================================
echo   Setup and run
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on this computer.
    echo.
    echo Download and install it from python.org
    echo IMPORTANT: on the first setup screen, check the box
    echo "Add python.exe to PATH"
    echo.
    pause
    exit /b 1
)

echo [1/4] Installing required libraries - first run may take
echo       a couple of minutes, next runs will be faster...
echo.
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

REM --- Looking for real data. Check order: ------------------------------
REM   1) already converted dataset (parquet, no cleaning needed)
REM   2) your own file data\input.xlsx or data\input.csv
REM   3) nothing found -> generate synthetic demo data as fallback
set TRAIN_DATA=
set SKIP_CLEAN=

if exist "data\1c_canonical.parquet" (
    set TRAIN_DATA=data\1c_canonical.parquet
    set SKIP_CLEAN=--skip-clean
    echo [2/4] Found ready dataset: data\1c_canonical.parquet
    goto :train
)

if exist "data\input.xlsx" (
    set TRAIN_DATA=data\input.xlsx
    echo [2/4] Found your file: data\input.xlsx
    goto :train
)

if exist "data\input.csv" (
    set TRAIN_DATA=data\input.csv
    echo [2/4] Found your file: data\input.csv
    goto :train
)

echo [2/4] No real data found (data\input.xlsx or
echo       data\1c_canonical.parquet), generating demo data...
python scripts\make_synthetic.py --days 730 --out data\clean.csv
if errorlevel 1 goto :error
set TRAIN_DATA=data\clean.csv

:train
echo.
echo [3/4] Training model on file: %TRAIN_DATA%
echo       (all further text in Russian below comes straight from
echo       the Python scripts and displays correctly)
echo.
python -m src.train --data %TRAIN_DATA% %SKIP_CLEAN% --out models\global
if errorlevel 1 goto :error

echo.
echo [4/4] Running full demo: cleaning -^> forecast -^> order plan -^> savings
echo ============================================
echo.
python scripts\run_demo.py --data %TRAIN_DATA% %SKIP_CLEAN%
if errorlevel 1 (
    echo.
    echo ============================================
    echo   Model trained and saved successfully ^(models\global^),
    echo   but the demo script crashed - see the error text above.
    echo   Send that text over and we will fix it.
    echo ============================================
    pause
    exit /b 1
)

echo.
echo ============================================
echo   DONE. See the results printed above.
echo   Trained on: %TRAIN_DATA%
echo ============================================
pause
exit /b 0

:error
echo.
echo [ERROR] Something went wrong - see the error text above.
pause
exit /b 1
