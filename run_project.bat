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
    echo Прочитайте текст ошибки выше — обычно дело в
    echo отсутствии интернета или старой версии pip.
    pause
    exit /b 1
)

echo.
echo [2/4] Генерирую тестовые данные о продажах...
python scripts\make_synthetic.py --days 730 --out data\clean.csv
if errorlevel 1 goto :error

echo.
echo [3/4] Обучаю модель прогноза спроса...
python -m src.train --data data\clean.csv --out models\global
if errorlevel 1 goto :error

echo.
echo [4/4] Запускаю полную демонстрацию: очистка -^> прогноз -^> закупка -^> эффект
echo ============================================
echo.
python scripts\run_demo.py --data data\clean.csv

echo.
echo ============================================
echo   ГОТОВО. Результат работы — выше в этом окне.
echo ============================================
pause
exit /b 0

:error
echo.
echo [ОШИБКА] Что-то пошло не так — см. текст ошибки выше.
pause
exit /b 1
