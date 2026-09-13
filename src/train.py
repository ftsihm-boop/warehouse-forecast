"""
Обучение глобальной модели.

Запуск:
    python -m src.train --data data/clean.csv --out models/global
    python -m src.train --data data/1c_canonical.parquet --out models/global \
                        --horizons 7 14 30 60 90 --test-days 90

Что происходит:
  1. очистка входного файла
  2. сборка обучающей выборки по всем горизонтам сразу
     (горизонт — обычный признак, поэтому модель одна, а не пять)
  3. разбиение ТОЛЬКО по времени
  4. обучение двух моделей: квантиль 0.5 и 0.9
  5. сравнение с baseline и сохранение артефактов

Модель не видит sku — только форму ряда, поэтому её можно применять
к товарам, которых не было в обучении.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import load_and_clean
from .evaluate import all_metrics, evaluate_baselines
from .features import (
    FEATURE_COLUMNS, TARGET_ABS, TARGET_COL, BASE_COL, build_training_set, time_split,
)
from .model import QuantileModel
from .schema import DATE, DEFAULT_HORIZONS, QUANTILE_HIGH, QUANTILE_LOW


def train_global(df: pd.DataFrame, horizons: list[int] | None = None,
                 test_days: int = 60, verbose: bool = True
                 ) -> tuple[QuantileModel, dict]:
    horizons = horizons or DEFAULT_HORIZONS

    if verbose:
        print(f"Сборка обучающей выборки, горизонты: {horizons}")
    ds = build_training_set(df, horizons)
    if verbose:
        print(f"  строк: {len(ds):,}  товаров: {ds['sku'].nunique()}")

    train, test = time_split(ds, test_days=test_days, date_col=DATE)
    feats = [c for c in FEATURE_COLUMNS if c in ds.columns]

    Xtr = train[feats].fillna(0.0)
    ytr = train[TARGET_COL]
    Xte = test[feats].fillna(0.0)
    yte = test[TARGET_COL]

    if verbose:
        print(f"  train: {len(train):,}  test: {len(test):,}")

    model = QuantileModel(alpha_low=QUANTILE_LOW, alpha_high=QUANTILE_HIGH)
    if verbose:
        print(f"  бэкенд: {model.backend}")
    model.fit(Xtr, ytr, eval_set=(Xte, yte) if len(test) else None)

    # --- метрики в АБСОЛЮТНЫХ единицах, а не в множителях -----------------
    report: dict = {"backend": model.backend, "horizons": horizons,
                    "n_train": int(len(train)), "n_test": int(len(test))}

    if len(test):
        q50_ratio, q90_ratio = model.predict(Xte)
        scale = test[BASE_COL].to_numpy() * test["horizon"].to_numpy()
        y_abs = test[TARGET_ABS].to_numpy()
        p_abs = q50_ratio * scale
        report["overall"] = all_metrics(y_abs, p_abs)

        per_h = []
        for h in horizons:
            m = test["horizon"].to_numpy() == h
            if m.sum() < 10:
                continue
            per_h.append({"Горизонт, дней": h, "Наблюдений": int(m.sum()),
                          **all_metrics(y_abs[m], p_abs[m])})
        report["by_horizon"] = per_h

        # проверка калибровки q90: доля фактов ниже верхней границы
        cover = float((y_abs <= q90_ratio * scale).mean())
        report["q90_coverage"] = round(cover, 3)

    imp = model.feature_importance()
    if imp is not None:
        report["top_features"] = imp.head(20).to_dict("records")

    return model, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение глобальной модели прогноза спроса")
    ap.add_argument("--data", required=True, help="CSV/XLSX/Parquet с историей продаж")
    ap.add_argument("--out", default="models/global", help="куда сохранить модель")
    ap.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--skip-clean", action="store_true",
                    help="данные уже в каноническом формате")
    a = ap.parse_args()

    path = Path(a.data)
    if a.skip_clean or path.suffix == ".parquet":
        df = (pd.read_parquet(path) if path.suffix == ".parquet"
              else pd.read_csv(path, parse_dates=["date"]))
        print(f"Загружено {len(df):,} строк (без очистки)")
    else:
        df, rep = load_and_clean(path)
        print(rep.to_text())
        print()

    model, report = train_global(df, a.horizons, a.test_days)

    print("\n" + "=" * 60)
    print("КАЧЕСТВО МОДЕЛИ (отложенная по времени выборка)")
    print("=" * 60)
    if "overall" in report:
        for k, v in report["overall"].items():
            print(f"  {k:<16} {v}")
        print(f"  Покрытие q90:    {report.get('q90_coverage')} (цель ~0.90)")
    if report.get("by_horizon"):
        print()
        print(pd.DataFrame(report["by_horizon"]).to_string(index=False))

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ С BASELINE (горизонт 30 дней)")
    print("=" * 60)
    bl = evaluate_baselines(df, horizon=30, test_days=a.test_days)
    if not bl.empty:
        rows = bl.to_dict("records")
        if "by_horizon" in report:
            for r in report["by_horizon"]:
                if r["Горизонт, дней"] == 30:
                    rows.append({"Модель": f"CatBoost ({model.backend})",
                                 **{k: v for k, v in r.items()
                                    if k not in ("Горизонт, дней", "Наблюдений")}})
        print(pd.DataFrame(rows).sort_values("WAPE, %").to_string(index=False))

    out = Path(a.out)
    model.save(out)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"\nМодель сохранена в {out}/")


if __name__ == "__main__":
    main()
