"""
Обучение модели сразу на нескольких наборах данных.

ЗАЧЕМ
-----
Модель, обученная на одном магазине, знает ровно его повадки. Приходит
пользователь с другим ассортиментом — и прогноз смещён, потому что таких
рядов модель не видела.

Лечится это не хитрыми настройками, а разнообразием обучающих данных.
Если модель видела и продуктовую розницу с ежедневным ровным спросом,
и магазин подарков с всплесками под праздники, и товары с выраженной
сезонностью — она учится реагировать на ФОРМУ ряда, а не запоминать
конкретные товары. Именно на это рассчитана архитектура: в признаках
нет ни названия товара, ни категории, только поведение продаж.

Скрипт принимает несколько канонических файлов (после prepare_1c.py,
prepare_online_retail.py или make_demo_xlsx.py), склеивает их в одну
обучающую выборку и обучает модель.

ЗАПУСК
------
    python scripts/train_multi.py \\
        --data data/1c_canonical.parquet \\
               data/online_retail_canonical.parquet \\
               data/demo_3years_20products.xlsx \\
        --out models/global

    # с ограничением: не дать одному источнику перевесить остальные
    python scripts/train_multi.py --data a.parquet b.parquet --balance

ГДЕ ВЗЯТЬ ДАННЫЕ
----------------
  Online Retail II — реальный ритейл, БЕЗ регистрации, есть цены:
      https://archive.ics.uci.edu/dataset/502/online+retail+ii
      затем: python scripts/prepare_online_retail.py

  Predict Future Sales (1С) — российские данные, нужна регистрация:
      https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data
      затем: python scripts/prepare_1c.py

  Store Item Demand Forecasting — 10 магазинов × 50 товаров, 5 лет:
      https://www.kaggle.com/c/demand-forecasting-kernels-only/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cleaning import load_and_clean  # noqa: E402
from src.evaluate import evaluate_baselines  # noqa: E402
from src.schema import (  # noqa: E402
    COST, DATE, DEFAULT_HORIZONS, FLAG_CENSORED, FLAG_FILLED, FLAG_RETURN,
    FLAG_WINSORIZED, PRICE, QTY, SKU, STOCK,
)
from src.train import train_global  # noqa: E402

CANONICAL = [DATE, SKU, QTY, STOCK, PRICE, COST,
             FLAG_FILLED, FLAG_CENSORED, FLAG_WINSORIZED, FLAG_RETURN]


def load_one(path: Path) -> pd.DataFrame:
    """Читает файл: канонический — как есть, сырой — через очистку."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        head = pd.read_csv(path, nrows=3) if path.suffix == ".csv" else None
        canonical = head is not None and {DATE, SKU, QTY}.issubset(head.columns)
        if canonical:
            df = pd.read_csv(path, parse_dates=[DATE])
        else:
            df, rep = load_and_clean(path)
            print(f"    очищено: {rep.rows_final:,} строк, "
                  f"{rep.sku_count} товаров")
            return df

    df[DATE] = pd.to_datetime(df[DATE])
    for col in (STOCK, PRICE, COST):
        if col not in df.columns:
            df[col] = float("nan")
    for f in (FLAG_FILLED, FLAG_CENSORED, FLAG_WINSORIZED, FLAG_RETURN):
        if f not in df.columns:
            df[f] = False
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="один или несколько файлов с историей продаж")
    ap.add_argument("--out", default="models/global")
    ap.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--balance", action="store_true",
                    help="уравнять вклад источников: у крупного взять "
                         "столько же товаров, сколько у самого мелкого")
    a = ap.parse_args()

    frames, sources = [], []
    for raw in a.data:
        path = Path(raw)
        if not path.exists():
            print(f"[пропуск] не найден: {path}")
            continue
        print(f"\nИсточник: {path.name}")
        df = load_one(path)
        if df.empty:
            print("    пусто, пропускаю")
            continue

        # Префикс к названию товара: в разных источниках артикулы
        # совпадают, и без него ряды склеились бы в один.
        tag = path.stem[:18]
        df = df.copy()
        df[SKU] = f"[{tag}] " + df[SKU].astype(str)

        print(f"    {len(df):,} строк, {df[SKU].nunique()} товаров, "
              f"{df[DATE].min():%d.%m.%Y} — {df[DATE].max():%d.%m.%Y}")
        frames.append(df)
        sources.append({"file": path.name, "rows": int(len(df)),
                        "skus": int(df[SKU].nunique()),
                        "from": str(df[DATE].min().date()),
                        "to": str(df[DATE].max().date())})

    if not frames:
        raise SystemExit("Не удалось прочитать ни одного файла.")

    if a.balance and len(frames) > 1:
        smallest = min(f[SKU].nunique() for f in frames)
        print(f"\nВыравнивание источников: по {smallest} товаров из каждого")
        balanced = []
        for f in frames:
            order = (f.groupby(SKU)[QTY].sum()
                     .sort_values(ascending=False).head(smallest).index)
            balanced.append(f[f[SKU].isin(order)])
        frames = balanced

    combined = pd.concat([f[CANONICAL] for f in frames], ignore_index=True)
    combined = combined.sort_values([SKU, DATE]).reset_index(drop=True)

    print("\n" + "=" * 62)
    print(f"ИТОГО: {len(combined):,} строк, {combined[SKU].nunique()} товаров "
          f"из {len(frames)} источник(ов)")
    print("=" * 62)

    model, report = train_global(combined, a.horizons, a.test_days)

    if "overall" in report:
        print("\nКачество на отложенной выборке:")
        for k, v in report["overall"].items():
            print(f"  {k:<16} {v}")
        print(f"  Покрытие q90:    {report.get('q90_coverage')} (цель ~0.90)")

    print("\nСравнение с baseline (горизонт 30 дней):")
    bl = evaluate_baselines(combined, horizon=30, test_days=a.test_days)
    if not bl.empty:
        rows = bl.to_dict("records")
        for r in report.get("by_horizon", []):
            if r["Горизонт, дней"] == 30:
                rows.append({"Модель": f"Наша модель ({model.backend})",
                             **{k: v for k, v in r.items()
                                if k not in ("Горизонт, дней", "Наблюдений")}})
        print(pd.DataFrame(rows).sort_values("WAPE, %").to_string(index=False))

    out = Path(a.out)
    model.save(out)
    report["sources"] = sources
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"\nМодель сохранена в {out}/")
    print(f"Обучена на {len(sources)} источник(ах) — это делает её "
          "устойчивее к чужому ассортименту.")


if __name__ == "__main__":
    main()
