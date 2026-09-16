@echo off
chcp 65001 >nul
title Warehouse Forecast - запуск проекта

echo ============================================
echo   Установка и запуск проекта
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ОШИБКА] Python не найден на этом компьютере.
    echo.
    echo Скачайте и установите его с сайта python.org
    echo ВАЖНО: на первом экране установщика обязательно
    echo поставьте галочку "Add python.exe to PATH"
    echo.
    pause
    exit /b 1
)

echo [1/4] Устанавливаю библиотеки — при первом запуске
echo       может занять пару минут, дальше будет быстрее...
echo.
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не получилось установить зависимости.
    pause
    exit /b 1
)

REM --- Ищем настоящие данные. Порядок проверки: ------------------------
REM   1) уже готовый конвертированный датасет (parquet, без очистки)
REM   2) ваш собственный файл data\input.xlsx или data\input.csv
REM   3) если ничего не нашли — генерируем синтетику как раньше
set TRAIN_DATA=
set SKIP_CLEAN=

if exist "data\1c_canonical.parquet" (
    set TRAIN_DATA=data\1c_canonical.parquet
    set SKIP_CLEAN=--skip-clean
    echo [2/4] Найден готовый датасет data\1c_canonical.parquet
    goto :train
)

if exist "data\input.xlsx" (
    set TRAIN_DATA=data\input.xlsx
    echo [2/4] Найден ваш файл data\input.xlsx — учимся на нём
    goto :train
)

if exist "data\input.csv" (
    set TRAIN_DATA=data\input.csv
    echo [2/4] Найден ваш файл data\input.csv — учимся на нём
    goto :train
)

echo [2/4] Настоящих данных не найдено ^(data\input.xlsx или
echo       data\1c_canonical.parquet^), генерирую тестовые...
python scripts\make_synthetic.py --days 730 --out data\clean.csv
if errorlevel 1 goto :error
set TRAIN_DATA=data\clean.csv

:train
echo.
echo [3/4] Обучаю модель на файле: %TRAIN_DATA%
python -m src.train --data %TRAIN_DATA% %SKIP_CLEAN% --out models\global
if errorlevel 1 goto :error

echo.
echo [4/4] Запускаю полную демонстрацию: очистка -^> прогноз -^> закупка -^> эффект
echo ============================================
echo.
python scripts\run_demo.py --data %TRAIN_DATA%

echo.
echo ============================================
echo   ГОТОВО. Результат работы — выше в этом окне.
echo   Обучено на: %TRAIN_DATA%
echo ============================================
pause
exit /b 0

:error
echo.
echo [ОШИБКА] Что-то пошло не так — см. текст ошибки выше.
pause
exit /b 1
