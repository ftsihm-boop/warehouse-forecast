"""
Производственный календарь РФ без внешних зависимостей.

Библиотека `holidays` сюда не тащится сознательно: нам нужны не столько
официальные выходные, сколько ДНИ ВСПЛЕСКА СПРОСА в рознице, а это
немного другой список (31 декабря важнее 12 июня, а 7 марта — важнее
самого 8 марта, потому что цветы и подарки покупают накануне).

Даёт три признака:
    is_holiday      — нерабочий праздничный день
    is_weekend      — суббота/воскресенье
    days_to_holiday — сколько дней до ближайшего праздника (0..14, иначе 15)
    days_after_holiday — сколько дней прошло после праздника (0..14, иначе 15)
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

# Фиксированные праздники (месяц, день) — нерабочие по ТК РФ
FIXED_HOLIDAYS: list[tuple[int, int]] = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),  # новогодние
    (2, 23),   # День защитника Отечества
    (3, 8),    # Международный женский день
    (5, 1),    # Праздник Весны и Труда
    (5, 9),    # День Победы
    (6, 12),   # День России
    (11, 4),   # День народного единства
]

# «Предпраздничные» даты, которые в рознице дают всплеск не хуже самого
# праздника. Считаем их праздничными для целей прогноза спроса.
RETAIL_PEAK_DAYS: list[tuple[int, int]] = [
    (12, 30), (12, 31),   # закупка к новогоднему столу
    (2, 22),              # канун 23 февраля
    (3, 7),               # канун 8 марта
    (4, 30),              # канун майских
    (12, 29),
]


def _holiday_dates(years: list[int]) -> set[_dt.date]:
    out: set[_dt.date] = set()
    for y in years:
        for m, d in FIXED_HOLIDAYS + RETAIL_PEAK_DAYS:
            try:
                out.add(_dt.date(y, m, d))
            except ValueError:  # 29 февраля и прочая экзотика
                continue
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Добавляет календарные признаки. Не мутирует исходный DataFrame."""
    df = df.copy()
    dates = pd.to_datetime(df[date_col])

    df["dow"] = dates.dt.dayofweek.astype("int16")            # 0=пн .. 6=вс
    df["day_of_month"] = dates.dt.day.astype("int16")
    df["month"] = dates.dt.month.astype("int16")
    df["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    # конец/начало месяца — зарплатные дни, заметный эффект в продуктовой рознице
    df["is_month_start"] = (df["day_of_month"] <= 3).astype("int8")
    df["is_month_end"] = dates.dt.is_month_end.astype("int8")

    years = sorted({int(y) for y in dates.dt.year.dropna().unique()})
    years = years + [min(years) - 1, max(years) + 1] if years else []
    hol = _holiday_dates(years)
    hol_sorted = np.array(sorted(pd.Timestamp(h).value for h in hol), dtype="int64")

    d_vals = dates.values.astype("datetime64[ns]").astype("int64")
    df["is_holiday"] = np.isin(d_vals, hol_sorted).astype("int8")

    day_ns = 86_400_000_000_000
    if len(hol_sorted):
        # ближайший праздник справа и слева от каждой даты
        idx = np.searchsorted(hol_sorted, d_vals, side="left")
        nxt = np.where(idx < len(hol_sorted), hol_sorted[np.clip(idx, 0, len(hol_sorted) - 1)], np.nan)
        prv_idx = np.clip(idx - 1, 0, len(hol_sorted) - 1)
        prv = np.where(idx > 0, hol_sorted[prv_idx], np.nan)
        to_h = (nxt - d_vals) / day_ns
        after_h = (d_vals - prv) / day_ns
    else:
        to_h = np.full(len(df), np.nan)
        after_h = np.full(len(df), np.nan)

    df["days_to_holiday"] = np.clip(np.nan_to_num(to_h, nan=15), 0, 15).astype("int16")
    df["days_after_holiday"] = np.clip(np.nan_to_num(after_h, nan=15), 0, 15).astype("int16")
    return df


CALENDAR_FEATURES = [
    "dow", "day_of_month", "month", "week_of_year",
    "is_weekend", "is_month_start", "is_month_end",
    "is_holiday", "days_to_holiday", "days_after_holiday",
]
