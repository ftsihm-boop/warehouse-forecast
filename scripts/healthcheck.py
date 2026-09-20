"""
Проверка живости приложения для контейнера.

Отдельным файлом, а не строкой внутри Dockerfile: однострочник с
кавычками и экранированием внутри HEALTHCHECK читается плохо и легко
ломается при правке. Здесь же видно, что именно считается «живым».

Код возврата 0 — приложение отвечает, 1 — нет. Именно так контейнер
сообщает платформе о своём состоянии.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

PORT = os.environ.get("PORT", "8000")
URL = f"http://127.0.0.1:{PORT}/health"


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=4) as r:
            if r.status != 200:
                print(f"health вернул {r.status}", file=sys.stderr)
                return 1
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"health недоступен: {e}", file=sys.stderr)
        return 1

    if data.get("status") != "ok":
        print(f"health ответил: {data}", file=sys.stderr)
        return 1

    # Отсутствие модели не делает приложение мёртвым — прогноз посчитается
    # по статистике. Но в журнал это попасть должно: на сервере такое
    # означает, что модель не доехала в образ.
    if not data.get("model_loaded"):
        print("внимание: обученная модель не загружена", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
