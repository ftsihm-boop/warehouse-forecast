"""
Проверка, что обученная модель читается в текущем окружении.

ЗАЧЕМ
-----
Модель сохраняется в формате того движка, которым обучалась, и другим
движком не читается. Если на сервер уедет модель CatBoost, а самого
CatBoost в образе нет, приложение спокойно запустится — и будет считать
прогноз по статистике. Ничего не упадёт, в журнале будет тишина, а цифры
на экране окажутся заметно хуже. Заметить это без специальной проверки
невозможно.

Поэтому проверка встроена в сборку образа. Если модель не читается,
сборка падает с понятным сообщением, и разбираться приходится в журнале
сборки, а не на работающем сайте.

ЗАПУСК
------
    python scripts/verify_model.py               models/global
    python scripts/verify_model.py <каталог>     другой каталог
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "global"

    if not (target / "meta.json").exists():
        print(f"ОШИБКА: в {target} нет обученной модели (не найден meta.json).")
        print("Без неё сервер будет считать прогноз по статистике.")
        print("Проверьте, что каталог models/global/ попал в репозиторий "
              "и не исключён в .gitignore и .dockerignore.")
        return 1

    from src.model import QuantileModel

    try:
        model = QuantileModel.load(target)
    except Exception as e:  # noqa: BLE001
        print(f"ОШИБКА: модель не читается — {e}")
        print()
        print("Две частые причины:")
        print("  1) модель сохранена в scikit-learn другой версии: pickle читается")
        print("     только той же версией (в requirements-runtime.txt она закреплена);")
        print("  2) модель обучена CatBoost, а в этом окружении его нет — тогда")
        print("     снимите решётку со строки catboost в requirements-runtime.txt")
        print("     или переобучите: python -m src.train --data <файл> --out models/global --backend sklearn")
        return 1

    n = len(model.features or [])
    print(f"Модель прочитана: бэкенд {model.backend}, признаков {n}, "
          f"квантили {model.alpha_low} и {model.alpha_high}")

    if n == 0:
        print("ВНИМАНИЕ: в модели не записан список признаков.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
