"""
Генератор синтетических данных продуктового магазина.

Нужен для двух вещей:
  1. демонстрации на защите — мы сами задаём сезонность и можем показать,
     что модель её восстановила, а скользящее среднее нет;
  2. отладки модуля очистки — умеет портить данные ровно теми способами,
     которые встречаются в реальных выгрузках.

Запуск:
    python scripts/make_synthetic.py --days 730 --out data/synthetic.csv
    python scripts/make_synthetic.py --days 730 --messy --out data/messy.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# профиль товара: (базовый спрос, амплитуда недельной, амплитуда годовой,
#                  пик года (день), разброс, вероятность промо)
PRODUCTS: dict[str, dict] = {
    "Молоко (1 литр)":   dict(base=22, weekly=0.25, yearly=0.08, peak=350, cv=0.25, promo=0.03),
    "Хлеб белый":        dict(base=40, weekly=0.30, yearly=0.05, peak=350, cv=0.20, promo=0.02),
    "Пиво (0,5 литра)":  dict(base=18, weekly=0.55, yearly=0.35, peak=190, cv=0.35, promo=0.06),
    "Бананы (1 кг)":     dict(base=14, weekly=0.20, yearly=0.15, peak=60,  cv=0.30, promo=0.05),
    "Персики (1 кг)":    dict(base=9,  weekly=0.20, yearly=0.95, peak=225, cv=0.45, promo=0.08),
    "Мандарины (1 кг)":  dict(base=11, weekly=0.25, yearly=0.90, peak=355, cv=0.45, promo=0.07),
    "Сигареты":          dict(base=26, weekly=0.10, yearly=0.03, peak=200, cv=0.15, promo=0.00),
    "Мороженое":         dict(base=12, weekly=0.35, yearly=0.75, peak=200, cv=0.40, promo=0.05),
}

# множители по дню недели: пн..вс. Пятница-суббота — пик в продуктовой рознице.
DOW_PROFILE = np.array([0.85, 0.85, 0.90, 1.00, 1.25, 1.35, 1.05])

HOLIDAY_BOOST = {(12, 30): 2.2, (12, 31): 2.6, (3, 7): 1.8, (2, 22): 1.5,
                 (4, 30): 1.6, (1, 1): 0.4, (1, 2): 0.6}


def generate(days: int = 730, start: str = "2024-01-01", seed: int = 42,
             stockouts: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []

    for name, p in PRODUCTS.items():
        stock = float(p["base"] * 14)
        trend = rng.uniform(-0.0002, 0.0006)  # лёгкий дрейф спроса
        for i, d in enumerate(dates):
            doy = d.dayofyear
            season_y = 1 + p["yearly"] * np.cos(2 * np.pi * (doy - p["peak"]) / 365.25)
            season_w = 1 + p["weekly"] * (DOW_PROFILE[d.dayofweek] - 1) / 0.35
            hol = HOLIDAY_BOOST.get((d.month, d.day), 1.0)
            promo = 1.8 if rng.random() < p["promo"] else 1.0

            mu = p["base"] * max(season_y, 0.05) * season_w * hol * promo * (1 + trend * i)
            demand = max(0.0, rng.normal(mu, mu * p["cv"]))
            demand = float(np.round(demand))

            sold = min(demand, stock) if stockouts else demand
            stock -= sold

            # поставка: раз в 3-4 дня, если запас ниже недельного спроса
            if stock < p["base"] * 4 and rng.random() < 0.45:
                stock += round(p["base"] * rng.uniform(7, 12))

            rows.append((d, name, float(sold), float(stock)))

    return pd.DataFrame(rows, columns=["Дата", "Товар", "Продано", "Остаток на складе"])


def make_messy(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Портит данные так, как их портит реальная жизнь."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    # pandas 3 запрещает класть строки в float-колонку — переводим в object
    for c in d.columns:
        d[c] = d[c].astype(object)

    # 1. разные написания одного товара
    mask = (d["Товар"] == "Молоко (1 литр)") & (rng.random(len(d)) < 0.25)
    d.loc[mask, "Товар"] = "Молоко(1 л)"
    mask = (d["Товар"] == "Бананы (1 кг)") & (rng.random(len(d)) < 0.15)
    d.loc[mask, "Товар"] = " Бананы  (1 кг) "

    # 2. числа как строки с пробелами и запятыми
    mask = rng.random(len(d)) < 0.15
    d.loc[mask, "Продано"] = d.loc[mask, "Продано"].map(
        lambda v: f"{int(v):,}".replace(",", "\u00a0") + ",0")

    # 3. единицы измерения в ячейке
    mask = rng.random(len(d)) < 0.05
    d.loc[mask, "Продано"] = d.loc[mask, "Продано"].map(lambda v: f"{v} шт")

    # 4. прочерки вместо нулей в остатке
    mask = rng.random(len(d)) < 0.04
    d.loc[mask, "Остаток на складе"] = "—"

    # 5. даты в разных форматах
    mask = rng.random(len(d)) < 0.20
    d.loc[mask, "Дата"] = pd.to_datetime(d.loc[mask, "Дата"]).dt.strftime("%d.%m.%Y")

    # 6. возвраты
    mask = rng.random(len(d)) < 0.01
    d.loc[mask, "Продано"] = -rng.integers(1, 4, mask.sum()).astype(float)

    # 7. дубли строк
    dup = d.sample(frac=0.02, random_state=seed)
    d = pd.concat([d, dup], ignore_index=True)

    # 8. выпавшие дни (касса не выгрузила)
    d = d[rng.random(len(d)) > 0.03].reset_index(drop=True)

    # 9. битые строки
    broken = pd.DataFrame({
        "Дата": ["", "ИТОГО", None],
        "Товар": ["", "", "Молоко (1 литр)"],
        "Продано": ["", 999999, "н/д"],
        "Остаток на складе": ["", "", ""],
    })
    d = pd.concat([d, broken], ignore_index=True)

    # 10. пустые хвостовые строки с форматированием
    d = pd.concat([d, pd.DataFrame([[None] * 4] * 7, columns=d.columns)],
                  ignore_index=True)
    return d


def wrap_with_1c_header(df: pd.DataFrame) -> pd.DataFrame:
    """Оборачивает таблицу шапкой, как это делает выгрузка из 1С."""
    head = pd.DataFrame([
        ["Отчёт о продажах по номенклатуре", None, None, None],
        ["Период: 01.01.2024 - 31.12.2025", None, None, None],
        ["Организация: ООО «Магазин у дома»", None, None, None],
        [None, None, None, None],
    ], columns=df.columns)
    return pd.concat([head, pd.DataFrame([df.columns.tolist()], columns=df.columns),
                      df], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--messy", action="store_true", help="испортить данные")
    ap.add_argument("--header-1c", action="store_true", help="добавить шапку 1С")
    ap.add_argument("--out", default="data/synthetic.csv")
    a = ap.parse_args()

    df = generate(a.days, a.start, a.seed)
    if a.messy:
        df = make_messy(df, a.seed)
    if a.header_1c:
        df = wrap_with_1c_header(df)
        header = False
    else:
        header = True

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".xlsx", ".xlsm"}:
        df.to_excel(out, index=False, header=header)
    else:
        df.to_csv(out, index=False, header=header, encoding="utf-8-sig")
    print(f"Записано {len(df)} строк в {out}")


if __name__ == "__main__":
    main()
