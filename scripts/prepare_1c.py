"""
Конвертер датасета «Predict Future Sales» (компания 1С) в канонический формат.

Источник (регистрация на Kaggle, бесплатно):
    https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data

Нужны файлы sales_train.csv и items.csv.

Исходные колонки sales_train.csv:
    date (дд.мм.гггг), date_block_num, shop_id, item_id, item_price, item_cnt_day

Что делает скрипт:
  * склеивает shop_id + item_id в один SKU (это и есть «товар на складе»)
  * подтягивает русские названия товаров из items.csv
  * отбирает ряды с достаточной историей (иначе обучать не на чем)
  * агрегирует дубли по дню и дозаполняет пропущенные даты нулями
  * складывает результат в parquet для быстрой загрузки

Запуск:
    python scripts/prepare_1c.py --raw data/raw_1c --out data/1c_canonical.parquet \
        --min-days 400 --max-series 500

Почему --max-series: в исходнике 2,9 млн строк и 400+ тыс. пар магазин-товар.
Для учебной модели это избыточно; 300-500 рядов с длинной историей дают
такое же качество и обучаются за минуты, а не за часы.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build(raw_dir: Path, min_days: int, max_series: int,
          min_mean_daily: float) -> pd.DataFrame:
    sales_path = raw_dir / "sales_train.csv"
    if not sales_path.exists():
        raise SystemExit(
            f"Не найден {sales_path}.\n"
            "Скачайте датасет со страницы конкурса и распакуйте в эту папку:\n"
            "  https://www.kaggle.com/competitions/"
            "competitive-data-science-predict-future-sales/data")

    print("Чтение sales_train.csv ...")
    sales = pd.read_csv(sales_path, dtype={
        "shop_id": "int16", "item_id": "int32",
        "item_cnt_day": "float32", "item_price": "float32"})
    sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")
    print(f"  {len(sales):,} строк, {sales['date'].min().date()} — "
          f"{sales['date'].max().date()}")

    # возвраты в исходнике идут отрицательными числами — обнуляем,
    # той же логикой, что и в src/cleaning.py
    returns = int((sales["item_cnt_day"] < 0).sum())
    sales.loc[sales["item_cnt_day"] < 0, "item_cnt_day"] = 0.0
    print(f"  возвратов обнулено: {returns:,}")

    # --- отбор рядов --------------------------------------------------------
    print("Отбор рядов с длинной историей ...")
    grp = sales.groupby(["shop_id", "item_id"])
    stat = grp.agg(days=("date", lambda s: (s.max() - s.min()).days + 1),
                   total=("item_cnt_day", "sum"),
                   n=("item_cnt_day", "size")).reset_index()
    stat["mean_daily"] = stat["total"] / stat["days"].clip(lower=1)
    keep = stat[(stat["days"] >= min_days) & (stat["mean_daily"] >= min_mean_daily)]
    keep = keep.sort_values("total", ascending=False).head(max_series)
    print(f"  отобрано {len(keep)} рядов из {len(stat):,}")

    if keep.empty:
        raise SystemExit("Ни один ряд не прошёл фильтры. Снизьте --min-days "
                         "или --min-mean-daily.")

    sel = sales.merge(keep[["shop_id", "item_id"]], on=["shop_id", "item_id"])

    # --- названия товаров ---------------------------------------------------
    items_path = raw_dir / "items.csv"
    if items_path.exists():
        items = pd.read_csv(items_path)[["item_id", "item_name"]]
        sel = sel.merge(items, on="item_id", how="left")
        sel["item_name"] = sel["item_name"].fillna("").str.slice(0, 60)
        sel["sku"] = ("Магазин " + sel["shop_id"].astype(str) + " / "
                      + sel["item_name"] + " [" + sel["item_id"].astype(str) + "]")
    else:
        sel["sku"] = ("shop" + sel["shop_id"].astype(str)
                      + "_item" + sel["item_id"].astype(str))

    # --- агрегация и дозаполнение -------------------------------------------
    print("Агрегация по дням ...")
    daily = (sel.groupby(["sku", "date"], as_index=False)
             .agg(qty=("item_cnt_day", "sum"), price=("item_price", "mean")))

    print("Дозаполнение пропущенных дат нулями ...")
    parts = []
    for sku, g in daily.groupby("sku", sort=False):
        g = g.sort_values("date")
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full).rename_axis("date").reset_index()
        g["sku"] = sku
        g["qty"] = g["qty"].fillna(0.0)
        g["price"] = g["price"].ffill().bfill()
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)

    # В исходнике нет остатков на складе. Восстанавливать их нельзя —
    # это было бы выдумкой. Ставим NaN: модуль очистки корректно это
    # обрабатывает, а признак цензурирования просто не считается.
    out["stock"] = np.nan
    out["is_censored"] = False
    out["is_filled"] = False
    out["is_winsorized"] = False
    out["is_return"] = False

    out = out[["date", "sku", "qty", "stock", "is_filled", "is_censored",
               "is_winsorized", "is_return"]]
    return out.sort_values(["sku", "date"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_1c", help="папка с sales_train.csv")
    ap.add_argument("--out", default="data/1c_canonical.parquet")
    ap.add_argument("--min-days", type=int, default=400)
    ap.add_argument("--max-series", type=int, default=500)
    ap.add_argument("--min-mean-daily", type=float, default=0.5)
    a = ap.parse_args()

    df = build(Path(a.raw), a.min_days, a.max_series, a.min_mean_daily)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\nГотово: {len(df):,} строк, {df['sku'].nunique()} рядов -> {out}")
    print(f"Период: {df['date'].min().date()} — {df['date'].max().date()}")
    print("\nДальше:")
    print(f"  python -m src.train --data {out} --out models/global --skip-clean")


if __name__ == "__main__":
    main()
