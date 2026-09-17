"""
Проверка автонастройки страхового запаса.

ЭТОТ СКРИПТ ЗАПУСКАТЬ НЕ ОБЯЗАТЕЛЬНО.
Система подбирает уровень страхового запаса сама: при каждом расчёте
она берёт цены из файла, считает экономику каждого товара по модели
газетчика и уточняет результат коротким бэктестом. Ничего настраивать
руками не нужно.

Скрипт нужен для другого — показать, КАК система пришла к своему ответу.
Он строит полную кривую «страховой запас против эффекта» и печатает
таблицу, по которой видно, что автоматически выбранное значение
действительно близко к оптимуму.

Для пояснительной записки это хороший материал: видно и теоретический
оптимум, и эмпирическую проверку, и то, что они согласуются.

ЗАПУСК
------
    python scripts/tune_service.py --data data/1c_canonical.parquet --skip-clean

    # с экономикой конкретного магазина
    python scripts/tune_service.py --data data/1c_canonical.parquet --skip-clean \
        --unit-cost 250 --margin 0.30 --holding-rate 0.45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import Economics, backtest  # noqa: E402
from src.cleaning import load_and_clean  # noqa: E402
from src.economics import autotune_service_factor  # noqa: E402
from src.model import QuantileModel  # noqa: E402
from src.schema import DATE, FLAG_CENSORED, FLAG_FILLED, FLAG_RETURN, FLAG_WINSORIZED  # noqa: E402

pd.set_option("display.width", 200)

# Что перебираем: множитель страхового запаса.
# 0.0 — вообще без страхового запаса (только медианный прогноз)
# 1.0 — полный расчётный запас под 90% уровень сервиса
DEFAULT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="models/global")
    ap.add_argument("--skip-clean", action="store_true")
    ap.add_argument("--test-days", type=int, default=180)
    ap.add_argument("--lead-time", type=int, default=7)
    ap.add_argument("--review-period", type=int, default=7)
    ap.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    # экономические допущения — их стоит подставить под реальный магазин
    ap.add_argument("--unit-cost", type=float, default=60.0,
                    help="закупочная цена единицы, ₽")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="наценка (доля от цены продажи), 0.25 = 25%%")
    ap.add_argument("--holding-rate", type=float, default=0.35,
                    help="стоимость хранения, доля от стоимости товара в год")
    ap.add_argument("--order-cost", type=float, default=300.0,
                    help="стоимость размещения одного заказа, ₽")
    a = ap.parse_args()

    # --- данные -------------------------------------------------------------
    path = Path(a.data)
    if a.skip_clean or path.suffix == ".parquet":
        df = (pd.read_parquet(path) if path.suffix == ".parquet"
              else pd.read_csv(path, parse_dates=[DATE]))
        df[DATE] = pd.to_datetime(df[DATE])
        for col in (FLAG_FILLED, FLAG_CENSORED, FLAG_WINSORIZED, FLAG_RETURN):
            if col not in df.columns:
                df[col] = False
    else:
        df, _ = load_and_clean(path)

    model_dir = Path(a.model)
    model = QuantileModel.load(model_dir) if (model_dir / "meta.json").exists() else None
    if model is None:
        print("Обученная модель не найдена — расчёт пойдёт по статистике.\n")

    # Цены из файла важнее аргументов командной строки: если в выгрузке
    # есть настоящие цифры, считаем по ним, а не по допущениям.
    econ_from_data = Economics.from_data(df, holding_rate_year=a.holding_rate,
                                         order_cost=a.order_cost)
    manual = (a.unit_cost != 60.0 or a.margin != 0.25)
    if manual:
        econ = Economics(unit_cost=a.unit_cost, margin=a.margin,
                         holding_rate_year=a.holding_rate, order_cost=a.order_cost)
        print("Используются цены, заданные в командной строке.\n")
    else:
        econ = econ_from_data
        print("Цены взяты из файла (если они там есть).\n")

    print("=" * 96)
    print("ПОДБОР УРОВНЯ СТРАХОВОГО ЗАПАСА")
    print("=" * 96)
    print(f"Допущения: закупочная цена {econ.unit_cost:.0f} ₽, наценка "
          f"{econ.margin:.0%}, хранение {econ.holding_rate_year:.0%} годовых, "
          f"размещение заказа {econ.order_cost:.0f} ₽")
    print(f"Прибыль с единицы: {econ.profit_per_unit:.1f} ₽ · "
          f"хранение единицы: {econ.holding_cost_per_unit_day() * 365:.1f} ₽/год")
    print()

    rows = []
    for factor in a.grid:
        table, summary = backtest(
            df, model, econ=econ, test_days=a.test_days,
            horizon=30, lead_time=a.lead_time,
            review_period=a.review_period, service_factor=factor)
        if not summary:
            continue
        rows.append({
            "Страх. запас": f"× {factor:.2f}",
            "Остаток, ₽": summary["Средний остаток to-be, ₽"],
            "Изм. остатка": summary["Снижение остатка, %"],
            "Упущено, ₽": summary["Упущенная выручка to-be, ₽"],
            "Хранение, ₽": summary["Затраты на хранение to-be, ₽"],
            "Эффект/мес, ₽": summary["Эффект в месяц, ₽"],
        })
        print(f"  проверено × {factor:.2f} -> эффект "
              f"{summary['Эффект в месяц, ₽']:>8} ₽/мес")

    if not rows:
        print("\nНедостаточно данных для бэктеста "
              "(нужно от ~270 дней истории по товару).")
        return

    res = pd.DataFrame(rows)
    print()
    print("=" * 96)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 96)
    print(res.to_string(index=False))

    best = res.loc[res["Эффект/мес, ₽"].idxmax()]
    print()
    print("=" * 96)
    print(f"ЛУЧШАЯ НАСТРОЙКА ПО ПЕРЕБОРУ: страховой запас {best['Страх. запас']}")
    print(f"  эффект {best['Эффект/мес, ₽']} ₽/мес · "
          f"изменение остатка {best['Изм. остатка']}% · "
          f"упущенная выручка {best['Упущено, ₽']} ₽")

    # --- сверяем с тем, что система выбирает автоматически ------------------
    auto = autotune_service_factor(df, model, econ=econ)
    print()
    print("ЧТО СИСТЕМА ВЫБИРАЕТ САМА (без этого скрипта):")
    print(f"  теоретический оптимум (модель газетчика): × {auto['theoretical']}")
    print(f"  после проверки на ваших данных:           × {auto['service_factor']}")
    print("=" * 96)
    print()
    print("Если автоматический выбор близок к лучшей строке таблицы —")
    print("автонастройка работает корректно, и ничего задавать вручную")
    print("не нужно. Таблица выше — обоснование для пояснительной записки.")


if __name__ == "__main__":
    main()
