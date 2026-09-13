"""
Генератор признаков для глобальной модели прогноза.

Два принципа, без которых гибрид (вариант C) не работает:

1. МОДЕЛЬ НЕ ЗНАЕТ, ЧТО ЗА ТОВАР ОНА ПРЕДСКАЗЫВАЕТ.
   Никаких sku / item_id / category в признаках — только форма ряда.
   Модель, обученная на дисках 1С, должна применяться к молоку.

2. ПРЕДСКАЗЫВАЕМ МНОЖИТЕЛЬ, А НЕ ШТУКИ.
       target = (сумма продаж за горизонт) / (среднее за 28 дней * горизонт)
   Товар с 5 шт/день и товар с 500 шт/день после нормировки для модели
   одинаковы. Обратно умножаем при выдаче прогноза. Без этого перенос
   на чужие данные не летит: модель просто выучит масштабы обучающей выборки.

Целевая переменная — СУММА ЗА ГОРИЗОНТ, а не значение конкретного дня.
Точно предсказать, сколько молока продадут 14 ноября, невозможно;
«за следующие 30 дней уйдёт 480-540 литров» — реально, и именно это
нужно для расчёта заказа.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_ru import CALENDAR_FEATURES, add_calendar_features
from .schema import (
    DATE, FLAG_CENSORED, QTY, ROLLING_BASE, SKU,
)

LAGS = [1, 2, 3, 7, 14, 21, 28]
WINDOWS = [7, 14, 28, 56]

BASE_COL = "base_mean"      # знаменатель нормировки
TARGET_COL = "target_ratio"  # целевая переменная (множитель)
TARGET_ABS = "target_abs"    # абсолютная сумма за горизонт (для метрик)

EPS = 1e-6


def _series_features(g: pd.DataFrame) -> pd.DataFrame:
    """Признаки одного временного ряда. g отсортирован по дате."""
    g = g.copy()
    qty = g[QTY].astype(float)

    # цензурированные дни (дефицит) не отражают спрос — для статистик
    # заменяем их на NaN, чтобы они не тянули средние вниз
    if FLAG_CENSORED in g.columns:
        qty_stat = qty.where(~g[FLAG_CENSORED].astype(bool), np.nan)
    else:
        qty_stat = qty

    base = qty_stat.shift(1).rolling(ROLLING_BASE, min_periods=7).mean()
    g[BASE_COL] = base

    for lag in LAGS:
        g[f"lag_{lag}"] = qty_stat.shift(lag) / (base + EPS)

    for w in WINDOWS:
        roll = qty_stat.shift(1).rolling(w, min_periods=max(3, w // 3))
        g[f"mean_{w}"] = roll.mean() / (base + EPS)
        g[f"std_{w}"] = roll.std() / (base + EPS)
        g[f"median_{w}"] = roll.median() / (base + EPS)
        g[f"max_{w}"] = roll.max() / (base + EPS)

    # тренд: отношение коротких средних к длинным
    g["trend_7_28"] = g["mean_7"] / (g["mean_28"] + EPS)
    g["trend_14_56"] = g["mean_14"] / (g["mean_56"] + EPS)

    # профиль ряда
    zero = (qty_stat.shift(1) <= 0).astype(float)
    g["zero_share_28"] = zero.rolling(28, min_periods=7).mean()
    g["cv_28"] = g["std_28"] / (g["mean_28"] + EPS)
    g["days_of_history"] = np.arange(len(g), dtype=float)

    if FLAG_CENSORED in g.columns:
        g["censored_share_28"] = (g[FLAG_CENSORED].astype(float)
                                  .shift(1).rolling(28, min_periods=7).mean())
    else:
        g["censored_share_28"] = 0.0

    # спрос в тот же день недели: средний множитель по dow за 8 недель
    dow_ratio = (qty_stat / (base + EPS))
    g["dow_ratio_8w"] = (dow_ratio.shift(7)
                         .rolling(8 * 7, min_periods=14)
                         .apply(lambda a: np.nanmean(a[::7]) if len(a) else np.nan,
                                raw=True))
    return g


def _add_target(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Целевая: сумма продаж за следующие `horizon` дней, нормированная."""
    g = g.copy()
    qty = g[QTY].astype(float)
    # сумма за окно ВПЕРЁД, не включая текущий день
    fwd = qty.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    g[TARGET_ABS] = fwd
    g[TARGET_COL] = fwd / ((g[BASE_COL] * horizon) + EPS)
    return g


FEATURE_COLUMNS: list[str] = (
    [f"lag_{l}" for l in LAGS]
    + [f"{s}_{w}" for w in WINDOWS for s in ("mean", "std", "median", "max")]
    + ["trend_7_28", "trend_14_56", "zero_share_28", "cv_28",
       "days_of_history", "censored_share_28", "dow_ratio_8w"]
    + CALENDAR_FEATURES
    + ["horizon"]
)


def build_features(df: pd.DataFrame, horizon: int | None = None,
                   with_target: bool = True) -> pd.DataFrame:
    """
    Собирает матрицу признаков.

    horizon=None + with_target=False -> матрица для инференса (последняя
    строка каждого SKU содержит актуальное состояние ряда).
    """
    parts = []
    for sku, g in df.groupby(SKU, sort=False):
        g = g.sort_values(DATE).reset_index(drop=True)
        g = _series_features(g)
        if with_target and horizon is not None:
            g = _add_target(g, horizon)
        g[SKU] = sku
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = add_calendar_features(out, DATE)
    out["horizon"] = float(horizon) if horizon is not None else np.nan
    return out


def build_training_set(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Одна модель на все горизонты: горизонт подаётся признаком."""
    frames = []
    for h in horizons:
        f = build_features(df, horizon=h, with_target=True)
        f = f[f[TARGET_COL].notna() & f[BASE_COL].notna() & (f[BASE_COL] > 0)]
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)

    # выбрасываем строки, где слишком много признаков не посчиталось
    need = [c for c in FEATURE_COLUMNS if c in out.columns]
    out = out[out[need].isna().mean(axis=1) < 0.3]

    # обрезаем безумные множители (target > 10 — это почти всегда артефакт)
    out = out[(out[TARGET_COL] >= 0) & (out[TARGET_COL] <= 10)]
    return out.reset_index(drop=True)


def inference_rows(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Последняя доступная строка по каждому SKU — точка, из которой прогнозируем."""
    f = build_features(df, horizon=horizon, with_target=False)
    last = f.sort_values(DATE).groupby(SKU, as_index=False).tail(1)
    return last.reset_index(drop=True)


def time_split(df: pd.DataFrame, test_days: int = 60,
               date_col: str = DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбиение ТОЛЬКО по времени. Случайное перемешивание на временных рядах —
    грубая ошибка: модель подглядывает в будущее через соседние строки.
    """
    cutoff = df[date_col].max() - pd.Timedelta(days=test_days)
    return df[df[date_col] <= cutoff].copy(), df[df[date_col] > cutoff].copy()
