"""
Сквозной прогон всего пайплайна одной командой — для демонстрации на защите
и для проверки, что ничего не сломалось после правок.

    python scripts/run_demo.py                      # на синтетике
    python scripts/run_demo.py --data мой_файл.xlsx # на своих данных
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import Economics, backtest  # noqa: E402
from src.cleaning import load_and_clean  # noqa: E402
from src.demand_classes import profile_all, summary_text  # noqa: E402
from src.evaluate import evaluate_baselines  # noqa: E402
from src.forecast import forecast, forecast_curve  # noqa: E402
from src.inventory import build_order_plan  # noqa: E402
from src.model import QuantileModel  # noqa: E402
from src.schema import DATE, FLAG_CENSORED, FLAG_FILLED, FLAG_RETURN, FLAG_WINSORIZED, SKU  # noqa: E402
from src.train import train_global  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def header(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/messy.xlsx")
    ap.add_argument("--model", default="models/global")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--lead-time", type=int, default=7)
    ap.add_argument("--train", action="store_true", help="переобучить модель")
    ap.add_argument("--skip-clean", action="store_true",
                    help="данные уже в каноническом формате (например, после prepare_1c.py)")
    a = ap.parse_args()

    # --- 1. очистка (или загрузка уже готовых канонических данных) ----------
    path = Path(a.data)
    if a.skip_clean or path.suffix == ".parquet":
        header("ШАГ 1. ДАННЫЕ УЖЕ В КАНОНИЧЕСКОМ ФОРМАТЕ — ОЧИСТКА НЕ ТРЕБУЕТСЯ")
        df = (pd.read_parquet(path) if path.suffix == ".parquet"
              else pd.read_csv(path, parse_dates=[DATE]))
        df[DATE] = pd.to_datetime(df[DATE])
        for col in (FLAG_FILLED, FLAG_CENSORED, FLAG_WINSORIZED, FLAG_RETURN):
            if col not in df.columns:
                df[col] = False
        print(f"Загружено {len(df):,} строк, {df[SKU].nunique()} рядов "
              f"(например, магазин+товар после подготовки датасета 1С)")
        print(f"Период: {df[DATE].min().date()} — {df[DATE].max().date()}")
    else:
        header("ШАГ 1. ОЧИСТКА ДАННЫХ")
        df, report = load_and_clean(a.data)
        print(report.to_text())
        header("СОСТОЯНИЕ ПО ТОВАРАМ")
        print(pd.DataFrame(report.per_sku).to_string(index=False))

    # --- 1б. классификация спроса ------------------------------------------
    header("ШАГ 1б. КЛАССИФИКАЦИЯ ТОВАРОВ ПО ХАРАКТЕРУ СПРОСА")
    profiles = profile_all(df)
    print(summary_text(profiles))
    if not profiles.empty:
        print("\nТоп-15 товаров по обороту:")
        cols = ["sku", "class_label", "abc_class", "adi", "cv2",
                "mean_daily", "zero_share", "forecastable"]
        top = profiles.head(15)[cols].copy()
        top["sku"] = top["sku"].str.slice(0, 44)
        print(top.to_string(index=False))

    # --- 2. модель ----------------------------------------------------------
    model_dir = Path(a.model)
    if a.train or not (model_dir / "meta.json").exists():
        header("ШАГ 2. ОБУЧЕНИЕ ГЛОБАЛЬНОЙ МОДЕЛИ")
        model, rep = train_global(df, horizons=[7, 14, 30], test_days=60)
        model.save(model_dir)
        if "overall" in rep:
            print("\nМетрики на отложенной выборке:")
            for k, v in rep["overall"].items():
                print(f"  {k:<16} {v}")
            print(f"  Покрытие q90:    {rep.get('q90_coverage')} (цель ~0.90)")
    else:
        model = QuantileModel.load(model_dir)
        header(f"ШАГ 2. МОДЕЛЬ ЗАГРУЖЕНА ({model.backend})")

    # --- 3. сравнение с baseline -------------------------------------------
    header(f"ШАГ 3. СРАВНЕНИЕ С BASELINE (горизонт {a.horizon} дней)")
    bl = evaluate_baselines(df, horizon=a.horizon, test_days=60)
    print(bl.to_string(index=False) if not bl.empty else "Недостаточно данных.")

    # --- 4. прогноз ---------------------------------------------------------
    header(f"ШАГ 4. ПРОГНОЗ СПРОСА НА {a.horizon} ДНЕЙ")
    fc = forecast(df, horizon=a.horizon, model=model, model_dir=None, verbose=True)
    print(fc[["sku", "q50", "q90", "daily_q50", "mode_label",
              "history_days", "stock"]].to_string(index=False))

    # --- 5. план закупки ----------------------------------------------------
    header("ШАГ 5. ПЛАН ЗАКУПКИ")
    plan = build_order_plan(fc, horizon_days=a.horizon,
                            lead_time_days=a.lead_time, review_period_days=7)
    cols = ["sku", "stock", "forecast_horizon_q50", "safety_stock",
            "reorder_point", "recommended_order", "days_of_supply",
            "stockout_date", "status_label"]
    print(plan[cols].to_string(index=False))

    # --- 6. график ----------------------------------------------------------
    first = fc.iloc[0]["sku"]
    header(f"ШАГ 6. ДНЕВНАЯ РАСКЛАДКА ПРОГНОЗА — «{first}» (первые 10 дней)")
    curve = forecast_curve(df, first, a.horizon, model=model, model_dir=None)
    print(curve.head(10).to_string(index=False) if not curve.empty else "н/д")

    # --- 7. бэктест ---------------------------------------------------------
    header("ШАГ 7. БЭКТЕСТ: РУЧНОЕ ПЛАНИРОВАНИЕ ПРОТИВ ML")
    econ = Economics()
    print(f"Допущения: закупочная цена {econ.unit_cost} ₽, наценка "
          f"{econ.margin:.0%}, хранение {econ.holding_rate_year:.0%} годовых, "
          f"размещение заказа {econ.order_cost} ₽")
    table, summary = backtest(df, model, econ=econ, test_days=180,
                              horizon=a.horizon, lead_time=a.lead_time,
                              review_period=7)
    if table.empty:
        print("\nИстории недостаточно для бэктеста (нужно от 270 дней).")
    else:
        print()
        print(table.to_string(index=False))
        print("\nСВОДКА:")
        for k, v in summary.items():
            print(f"  {k:<36} {v}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
