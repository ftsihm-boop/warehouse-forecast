# Образ для развёртывания системы управления складскими запасами.
#
# ЧТО ЗАПУСКАЕТСЯ
# ---------------
# scripts/serve.py — он отдаёт и сайт, и программный интерфейс с одного
# адреса. Прежняя версия образа запускала api/main.py на FastAPI, но там
# нет двух ключевых эндпоинтов: расчёта экономического эффекта и прогресса
# обработки файла. С таким образом на сервере оказалась бы урезанная
# система, а именно экономический расчёт — главный результат работы.
#
# ЗАВИСИМОСТИ
# -----------
# Ставится только то, что нужно на сервере (requirements-runtime.txt):
# FastAPI, uvicorn и CatBoost в образ не попадают. CatBoost не нужен
# потому, что модель сохранена в формате sklearn и читается им же, а он
# один тянет за собой сотни мегабайт.

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Зависимости ставятся отдельным слоем: пока requirements не меняется,
# при пересборке этот шаг берётся из кеша и деплой занимает секунды.
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY web/ ./web/
COPY models/ ./models/

# Приложение работает не под root: если через загрузку файла найдут дыру,
# у процесса не должно быть прав на всю систему.
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8000

# Платформа считает приложение живым, только если отвечает /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]

CMD ["python", "scripts/serve.py"]
