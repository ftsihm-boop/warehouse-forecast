"""
Конвертер датасета Online Retail II в канонический формат.

ЧТО ЭТО ЗА ДАННЫЕ
-----------------
Реальные продажи британского интернет-магазина подарков и товаров для
дома за два года (декабрь 2009 — декабрь 2011), больше миллиона строк.
Каждая строка — позиция в чеке: дата со временем, артикул, название,
количество, цена за единицу.

Чем ценен именно этот набор: он скачивается БЕЗ РЕГИСТРАЦИИ, в отличие
от Kaggle, и в нём есть настоящие цены — значит экономический расчёт
будет считаться по реальным цифрам, а не по допущениям.

    https://archive.ics.uci.edu/dataset/502/online+retail+ii
    файл online_retail_II.xlsx (~44 МБ, два листа)

ЧТО ДЕЛАЕТ СКРИПТ
-----------------
  * склеивает оба листа (2009-2010 и 2010-2011)
  * отбрасывает возвраты, отмены и служебные позиции (доставка, комиссия)
  * агрегирует чеки в дневные продажи по каждому товару
  * дозаполняет дни без продаж нулями
  * переносит цену продажи и оценивает закупочную по типичной для
    этой отрасли наценке
  * отбирает товары с длинной историей и регулярным спросом

Запуск:
    python scripts/prepare_online_retail.py --raw data/online_retail_II.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.demand_classes import split_by_forecastability, summary_text  # noqa: E402

# Позиции, которые не являются товаром на складе
SERVICE_CODES = {"POST", "DOT", "C2", "M", "BANK CHARGES", "PADS", "AMAZONFEE",
                 "CRUK", "S", "B", "D", "GIFT"}
SERVICE_WORDS = ["postage", "carriage", "manual", "fee", "adjust", "sample",
                 "discount", "commission", "bank charges", "cheque",
                 "check", "damages", "found", "lost", "test", "?"]

# В рознице подарков наценка обычно высокая; закупочную цену оцениваем
# от неё, потому что в исходнике её нет. Это допущение, и оно помечается.
ASSUMED_MARGIN = 0.45


def _is_service(code: str, desc: str) -> bool:
    c = str(code).strip().upper()
    if c in SERVICE_CODES or len(c) <= 1:
        return True
    d = str(desc).lower()
    return any(w in d for w in SERVICE_WORDS)


def build(raw: Path, min_days: int, max_series: int, min_mean_daily: float,
          classify: bool = True) -> pd.DataFrame:
    if not raw.exists():
        raise SystemExit(
            f"Не найден {raw}.\n"
            "Скачайте online_retail_II.xlsx (регистрация не нужна):\n"
            "  https://archive.ics.uci.edu/dataset/502/online+retail+ii")

    print(f"Чтение {raw.name} ...")
    if raw.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(raw, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
        print(f"  листов: {len(sheets)}")
    else:
        df = pd.read_csv(raw, encoding="latin-1")

    # названия колонок в разных публикациях отличаются пробелами и регистром
    df.columns = [str(c).strip().lower().replace(" ", "") for c in df.columns]
    ren = {"invoice": "invoice", "invoiceno": "invoice",
           "stockcode": "code", "description": "desc",
           "quantity": "qty", "invoicedate": "date",
           "price": "price", "unitprice": "price"}
    df = df.rename(columns={c: ren[c] for c in df.columns if c in ren})

    need = {"invoice", "code", "qty", "date", "price"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"В файле нет колонок: {', '.join(sorted(missing))}. "
                         f"Найдено: {', '.join(df.columns)}")

    print(f"  {len(df):,} строк")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]

    # --- чистка ------------------------------------------------------------
    before = len(df)
    # отменённые чеки помечены буквой C в номере
    df = df[~df["invoice"].astype(str).str.upper().str.startswith("C")]
    # возвраты и нулевые цены
    df = df[(df["qty"] > 0) & (df["price"] > 0)]
    print(f"  отброшено возвратов и отмен: {before - len(df):,}")

    if "desc" not in df.columns:
        df["desc"] = ""
    before = len(df)
    mask_service = [_is_service(c, d) for c, d in zip(df["code"], df["desc"])]
    df = df[~np.array(mask_service)]
    print(f"  отброшено служебных позиций (доставка, комиссии): "
          f"{before - len(df):,}")

    # --- товар = артикул + название ----------------------------------------
    df["desc"] = df["desc"].astype(str).str.strip().str.slice(0, 55)
    names = (df.groupby("code")["desc"]
             .agg(lambda s: s.value_counts().index[0] if len(s) else ""))
    df["sku"] = df["code"].astype(str).str.strip() + " — " + df["code"].map(names)

    # --- агрегация в дневные продажи ---------------------------------------
    print("Агрегация чеков в дневные продажи ...")
    df["day"] = df["date"].dt.normalize()
    daily = (df.groupby(["sku", "day"], as_index=False)
             .agg(qty=("qty", "sum"), price=("price", "mean")))
    daily = daily.rename(columns={"day": "date"})

    # --- отбор рядов --------------------------------------------------------
    stat = daily.groupby("sku").agg(
        days=("date", lambda s: (s.max() - s.min()).days + 1),
        total=("qty", "sum"), n=("qty", "size")).reset_index()
    stat["mean_daily"] = stat["total"] / stat["days"].clip(lower=1)
    keep = stat[(stat["days"] >= min_days) & (stat["mean_daily"] >= min_mean_daily)]
    keep = keep.sort_values("total", ascending=False).head(max_series)
    print(f"  отобрано {len(keep)} рядов из {len(stat):,}")
    if keep.empty:
        raise SystemExit("Ни один ряд не прошёл фильтры — снизьте --min-days.")

    daily = daily[daily["sku"].isin(set(keep["sku"]))]

    # --- дозаполнение дней без продаж --------------------------------------
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

    # --- цены ---------------------------------------------------------------
    # Цена продажи в данных настоящая, закупочной нет — оцениваем её
    # по отраслевой наценке. Это допущение, и система пометит его как
    # оценку, а не как факт из файла.
    out["cost"] = (out["price"] * (1 - ASSUMED_MARGIN)).round(2)

    # Остатков в исходнике нет, и выдумывать их нельзя: модуль очистки
    # корректно работает с их отсутствием.
    out["stock"] = np.nan
    for f in ("is_filled", "is_censored", "is_winsorized", "is_return"):
        out[f] = False

    out = out[["date", "sku", "qty", "stock", "price", "cost",
               "is_filled", "is_censored", "is_winsorized", "is_return"]]
    out = out.sort_values(["sku", "date"]).reset_index(drop=True)

    if classify:
        print("Классификация по характеру спроса ...")
        keep_df, _drop, profiles = split_by_forecastability(
            out, min_history_days=min_days)
        print(summary_text(profiles))
        if not keep_df.empty:
            dropped = out["sku"].nunique() - keep_df["sku"].nunique()
            print(f"\n  исключено {dropped} товар(ов) с непрогнозируемым спросом")
            out = keep_df.reset_index(drop=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/online_retail_II.xlsx")
    ap.add_argument("--out", default="data/online_retail_canonical.parquet")
    ap.add_argument("--min-days", type=int, default=250)
    ap.add_argument("--max-series", type=int, default=800)
    ap.add_argument("--min-mean-daily", type=float, default=1.0)
    ap.add_argument("--no-classify", action="store_true")
    a = ap.parse_args()

    df = build(Path(a.raw), a.min_days, a.max_series, a.min_mean_daily,
               classify=not a.no_classify)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        try:
            df.to_parquet(out, index=False)
        except Exception:
            out = out.with_suffix(".csv")
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print("  (parquet недоступен — сохранено в CSV)")
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\nГотово: {len(df):,} строк, {df['sku'].nunique()} рядов -> {out}")
    print(f"Период: {df['date'].min():%d.%m.%Y} — {df['date'].max():%d.%m.%Y}")
    print("\nДальше:")
    print(f"  python -m src.train --data {out} --out models/global --skip-clean")


if __name__ == "__main__":
    main()
