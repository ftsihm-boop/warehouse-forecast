"""
Генератор демонстрационного Excel-файла для проверки сайта.

Делает выгрузку продаж продуктового магазина, максимально похожую на то,
что отдаёт касса: дата, товар, продано, остаток, цены. Три года истории
по 20 товарам.

Товары подобраны так, чтобы в файле оказались ВСЕ типы спроса, которые
умеет различать система, — иначе классификация на сайте не будет видна:

    гладкий       молоко, хлеб, яйца — продаются каждый день ровно
    сезонный      мороженое, мандарины, арбуз — выраженный годовой цикл
    неравномерный пиво, чипсы — продаются часто, но объём скачет
    прерывистый   мангал, шампанское — редко, но предсказуемыми порциями
    комковатый    торт на заказ, икра — редко и хаотично

Плюс реалистичные помехи: недельный профиль (пятница-суббота выше),
всплески в праздники, промо-акции, периоды дефицита (товар кончился
на складе — это те самые «цензурированные» дни), рост цен со временем.

Запуск:
    python scripts/make_demo_xlsx.py
    python scripts/make_demo_xlsx.py --years 3 --products 20 --out demo.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# (название, закупка ₽, наценка, базовый спрос, недельная, годовая,
#  пик года (день), разброс, вероятность промо, тип)
PRODUCTS: list[tuple] = [
    # --- ходовые, продаются каждый день ---------------------------------
    ("Молоко «Домик в деревне» 1 л",      62,  0.28, 34, 0.22, 0.06, 350, 0.18, 0.04, "smooth"),
    ("Хлеб «Бородинский» 400 г",          38,  0.35, 48, 0.28, 0.05, 350, 0.16, 0.03, "smooth"),
    ("Яйцо куриное С1, 10 шт",           110,  0.24, 26, 0.30, 0.12, 100, 0.20, 0.05, "smooth"),
    ("Сахар-песок 1 кг",                  68,  0.18, 18, 0.20, 0.15, 220, 0.22, 0.04, "smooth"),
    ("Масло подсолнечное 1 л",           125,  0.22, 14, 0.18, 0.08, 350, 0.20, 0.05, "smooth"),
    ("Макароны «Макфа» 450 г",            72,  0.30, 16, 0.18, 0.05, 350, 0.19, 0.06, "smooth"),
    ("Вода питьевая 5 л",                 95,  0.32, 22, 0.25, 0.30, 200, 0.21, 0.04, "smooth"),
    ("Сигареты (пачка)",                 185,  0.08, 30, 0.10, 0.03, 200, 0.12, 0.00, "smooth"),
    # --- сезонные -------------------------------------------------------
    ("Мороженое «Пломбир» 100 г",         48,  0.42, 16, 0.35, 0.85, 200, 0.35, 0.07, "seasonal"),
    ("Мандарины 1 кг",                   145,  0.30, 14, 0.25, 0.90, 355, 0.40, 0.08, "seasonal"),
    ("Арбуз 1 кг",                        42,  0.38, 20, 0.30, 1.00, 235, 0.45, 0.06, "seasonal"),
    ("Персики 1 кг",                     190,  0.32, 10, 0.22, 0.95, 225, 0.42, 0.08, "seasonal"),
    # --- неравномерные: часто, но объём скачет --------------------------
    ("Пиво «Жигулёвское» 0,5 л",          58,  0.34, 24, 0.55, 0.35, 195, 0.55, 0.09, "erratic"),
    ("Чипсы «Lay's» 150 г",              105,  0.36, 12, 0.45, 0.20, 195, 0.60, 0.10, "erratic"),
    ("Кока-кола 2 л",                    130,  0.28, 15, 0.40, 0.30, 200, 0.50, 0.08, "erratic"),
    # --- прерывистые: редко, но ровными порциями ------------------------
    ("Мангал одноразовый",               260,  0.45,  4, 0.30, 0.90, 180, 0.25, 0.05, "intermittent"),
    ("Шампанское «Абрау-Дюрсо»",         520,  0.40,  3, 0.35, 0.80, 358, 0.28, 0.06, "intermittent"),
    # --- комковатые: редко и непредсказуемо -----------------------------
    ("Торт на заказ «Медовик» 2 кг",    1200,  0.35,  2, 0.20, 0.25, 350, 0.90, 0.02, "lumpy"),
    ("Икра лососёвая 140 г",            1450,  0.30,  2, 0.25, 0.60, 358, 0.95, 0.03, "lumpy"),
    ("Подарочный набор конфет",           680,  0.38,  2, 0.20, 0.70, 358, 0.92, 0.04, "lumpy"),
]

DOW_PROFILE = np.array([0.82, 0.84, 0.90, 1.00, 1.28, 1.38, 1.06])  # пн..вс

# Праздники, в которые продуктовая розница резко растёт (месяц, день)
HOLIDAY_BOOST = {
    (12, 29): 1.8, (12, 30): 2.4, (12, 31): 2.9,
    (1, 1): 0.35, (1, 2): 0.55, (1, 3): 0.75,
    (2, 22): 1.5, (2, 23): 1.2,
    (3, 6): 1.4, (3, 7): 1.9, (3, 8): 1.3,
    (4, 30): 1.7, (5, 1): 1.4, (5, 8): 1.5, (5, 9): 1.3,
    (6, 11): 1.4, (11, 3): 1.3,
}


def generate(years: int, n_products: int, start: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = int(round(years * 365.25))
    dates = pd.date_range(start, periods=days, freq="D")
    products = PRODUCTS[:n_products]

    rows = []
    for (name, cost, margin, base, weekly, yearly, peak, cv, promo_p, kind) in products:
        price = round(cost / (1 - margin))
        # запас на старте: примерно двухнедельная потребность
        stock = float(base * rng.uniform(10, 18))
        # лёгкий тренд спроса и годовая инфляция цен
        trend = rng.uniform(-0.00015, 0.00045)

        for i, d in enumerate(dates):
            doy = d.dayofyear

            season_y = 1 + yearly * np.cos(2 * np.pi * (doy - peak) / 365.25)
            season_w = 1 + weekly * (DOW_PROFILE[d.dayofweek] - 1) / 0.38
            hol = HOLIDAY_BOOST.get((d.month, d.day), 1.0)
            promo = rng.uniform(1.6, 2.4) if rng.random() < promo_p else 1.0

            mu = base * max(season_y, 0.03) * season_w * hol * promo * (1 + trend * i)

            if kind in ("lumpy", "intermittent"):
                # редкий спрос: сначала решаем, был ли он вообще сегодня
                p_sale = 0.30 if kind == "intermittent" else 0.16
                p_sale *= min(max(season_y, 0.1), 2.0) * hol
                if rng.random() > min(p_sale, 0.95):
                    demand = 0.0
                elif kind == "intermittent":
                    demand = float(max(1, round(rng.normal(base * 1.6, base * cv))))
                else:
                    # комковатый: обычно штучно, иногда крупная партия
                    demand = float(rng.choice([1, 1, 2, 2, 3, 8, 14, 22],
                                              p=[.26, .22, .16, .12, .09, .08, .04, .03]))
            else:
                demand = float(max(0, round(rng.normal(mu, mu * cv))))

            # продать можно только то, что есть на складе
            sold = min(demand, stock)
            stock -= sold

            # поставка: когда запас падает ниже недельной потребности
            weekly_need = max(base * 7, 7)
            if stock < weekly_need * 0.55 and rng.random() < 0.5:
                stock += round(weekly_need * rng.uniform(1.4, 2.6))

            # цены медленно растут (~8% в год), промо снижает цену продажи
            infl = 1 + 0.08 * (i / 365.25)
            day_price = round(price * infl * (0.85 if promo > 1 else 1.0), 2)
            day_cost = round(cost * infl, 2)

            rows.append((d, name, int(sold), int(round(stock)), day_price, day_cost))

    df = pd.DataFrame(rows, columns=[
        "Дата", "Товар", "Продано", "Остаток на складе",
        "Цена продажи", "Закупочная цена"])
    return df.sort_values(["Дата", "Товар"]).reset_index(drop=True)


def write_excel(df: pd.DataFrame, path: Path) -> None:
    """Пишет xlsx с настоящим форматом даты и читаемой шапкой."""
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="DD.MM.YYYY") as xl:
        df.to_excel(xl, index=False, sheet_name="Продажи")
        ws = xl.sheets["Продажи"]

        from openpyxl.styles import Alignment, Font, PatternFill

        head_fill = PatternFill("solid", fgColor="1F5F5B")
        head_font = Font(color="FFFFFF", bold=True, size=11)
        for cell in ws[1]:
            cell.fill = head_fill
            cell.font = head_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        widths = {"A": 13, "B": 34, "C": 11, "D": 19, "E": 14, "F": 17}
        for col, w in widths.items():
            ws.column_dimensions[col].width = w

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for row in ws.iter_rows(min_row=2, min_col=5, max_col=6):
            for cell in row:
                cell.number_format = "#,##0.00"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--products", type=int, default=20)
    ap.add_argument("--start", default=None,
                    help="дата начала; по умолчанию так, чтобы история "
                         "заканчивалась сегодня")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/demo_3years_20products.xlsx")
    a = ap.parse_args()

    if a.start:
        start = a.start
    else:
        days = int(round(a.years * 365.25))
        start = (pd.Timestamp.today().normalize()
                 - pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")

    df = generate(a.years, a.products, start, a.seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False, encoding="utf-8-sig")
    else:
        write_excel(df, out)

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Готово: {out}")
    print(f"  строк:      {len(df):,}")
    print(f"  товаров:    {df['Товар'].nunique()}")
    print(f"  период:     {df['Дата'].min():%d.%m.%Y} — {df['Дата'].max():%d.%m.%Y}")
    print(f"  размер:     {size_mb:.1f} МБ")
    print(f"  дней с дефицитом: "
          f"{int(((df['Продано'] == 0) & (df['Остаток на складе'] == 0)).sum())}")


if __name__ == "__main__":
    main()
