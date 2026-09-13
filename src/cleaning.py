"""
Слои 3, 4 и 5 очистки.

Слой 3 — логика данных:
  * отрицательные продажи (возвраты) -> 0 + флаг
  * дубли «дата+товар» -> суммируем
  * фаззи-склейка похожих наименований товара
  * пропущенные даты -> дозаполняем нулями (для временного ряда пропуск
    строки означает «продали 0», а не «нет данных»)
  * выбросы -> винзоризация по MAD (устойчивее перцентилей), не удаление
  * отрицательный остаток -> 0

Слой 4 — ЦЕНЗУРИРОВАННЫЙ СПРОС:
  день, когда продажи = 0 И остаток = 0, это не «спроса не было»,
  а дефицит: спрос был, товара не было. Если скормить такие дни модели
  как честные нули, она научится систематически занижать спрос — и система
  начнёт усугублять out-of-stock вместо того чтобы его лечить.

Слой 5 — машиночитаемый отчёт об очистке для интерфейса и для курсовой.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .ingest import IngestResult, sku_key
from .schema import (
    DATE, FLAG_CENSORED, FLAG_FILLED, FLAG_RETURN, FLAG_WINSORIZED,
    GLOBAL_HISTORY_DAYS, MIN_HISTORY_DAYS, MODE_FINE_TUNE, MODE_GLOBAL,
    MODE_REJECT, MODE_STATS, QTY, SKU, STATS_HISTORY_DAYS, STOCK,
)

WINSOR_MAD_K = 6.0        # порог выброса: |x - median| > k * MAD
MIN_DAYS_FOR_WINSOR = 21  # на коротких рядах выбросы не ищем


@dataclass
class CleaningReport:
    rows_read: int = 0
    rows_dropped_bad_date: int = 0
    rows_dropped_bad_qty: int = 0
    rows_dropped_empty_sku: int = 0
    duplicates_merged: int = 0
    returns_zeroed: int = 0
    dates_filled: int = 0
    outliers_winsorized: int = 0
    negative_stock_fixed: int = 0
    censored_days: int = 0
    sku_merged: list[tuple[str, str]] = field(default_factory=list)
    sku_count: int = 0
    date_min: str | None = None
    date_max: str | None = None
    rows_final: int = 0
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_sku: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        """Человекочитаемый лог — его же можно отдать кнопкой «скачать»."""
        L = [
            "ОТЧЁТ ОБ ОЧИСТКЕ ДАННЫХ",
            "=" * 52,
            f"Прочитано строк:                {self.rows_read}",
            f"Отброшено (не распознана дата): {self.rows_dropped_bad_date}",
            f"Отброшено (не распознано кол-во): {self.rows_dropped_bad_qty}",
            f"Отброшено (пустое наименование): {self.rows_dropped_empty_sku}",
            f"Дублей объединено:              {self.duplicates_merged}",
            f"Возвратов обнулено:             {self.returns_zeroed}",
            f"Пропущенных дат дозаполнено:    {self.dates_filled}",
            f"Выбросов обрезано:              {self.outliers_winsorized}",
            f"Отрицательных остатков исправлено: {self.negative_stock_fixed}",
            f"Дней дефицита обнаружено:       {self.censored_days}",
            "-" * 52,
            f"Товаров:                        {self.sku_count}",
            f"Период:                         {self.date_min} — {self.date_max}",
            f"Строк в итоговой таблице:       {self.rows_final}",
        ]
        if self.sku_merged:
            L.append("-" * 52)
            L.append("Объединённые наименования:")
            L += [f"  «{a}» -> «{b}»" for a, b in self.sku_merged]
        if self.notes:
            L.append("-" * 52)
            L += [f"  {n}" for n in self.notes]
        if self.warnings:
            L.append("-" * 52)
            L.append("ПРЕДУПРЕЖДЕНИЯ:")
            L += [f"  ! {w}" for w in self.warnings]
        return "\n".join(L)


def _merge_similar_skus(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Схлопывает наименования с одинаковым нормализованным ключом."""
    keys = df[SKU].map(sku_key)
    groups: dict[str, list[str]] = defaultdict(list)
    for key, name in zip(keys, df[SKU]):
        if name not in groups[key]:
            groups[key].append(name)

    rename: dict[str, str] = {}
    for key, names in groups.items():
        if len(names) > 1:
            # канонический вариант — самый частый, при равенстве самый длинный
            counts = df.loc[keys == key, SKU].value_counts()
            canon = max(names, key=lambda n: (counts.get(n, 0), len(n)))
            for n in names:
                if n != canon:
                    rename[n] = canon
                    report.sku_merged.append((n, canon))
    if rename:
        df[SKU] = df[SKU].replace(rename)
    return df


def _winsorize_group(qty: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Обрезка выбросов по MAD. Возвращает (значения, флаг обрезки)."""
    flag = pd.Series(False, index=qty.index)
    if len(qty) < MIN_DAYS_FOR_WINSOR:
        return qty, flag
    med = float(qty.median())
    mad = float((qty - med).abs().median())
    if mad <= 0:
        # Вырожденный случай: больше половины значений совпадают с медианой.
        # Стандартное отклонение тут считать нельзя — сам выброс его и раздует,
        # порог уедет выше выброса и тот не поймается. Берём кратность медианы.
        if med <= 0:
            return qty, flag
        upper = med * WINSOR_MAD_K
    else:
        upper = med + WINSOR_MAD_K * 1.4826 * mad
    # обрезаем только сверху: всплеск продаж искажает среднее гораздо
    # сильнее, чем ноль, а нули для нас содержательны (см. слой 4)
    mask = qty > upper
    if mask.any():
        flag = mask
        qty = qty.where(~mask, upper)
    return qty, flag


def clean(ingested: IngestResult, fill_missing_dates: bool = True,
          winsorize: bool = True) -> tuple[pd.DataFrame, CleaningReport]:
    """Главная функция очистки. Возвращает канонический DataFrame и отчёт."""
    df = ingested.frame.copy()
    rep = CleaningReport(rows_read=len(df), notes=list(ingested.notes))

    # --- слой 3.1: отбраковка непригодных строк ----------------------------
    bad_date = df[DATE].isna()
    rep.rows_dropped_bad_date = int(bad_date.sum())
    df = df[~bad_date]

    empty_sku = df[SKU].astype(str).str.strip().eq("") | df[SKU].isna()
    rep.rows_dropped_empty_sku = int(empty_sku.sum())
    df = df[~empty_sku]

    bad_qty = df[QTY].isna()
    rep.rows_dropped_bad_qty = int(bad_qty.sum())
    df = df[~bad_qty]

    if df.empty:
        rep.warnings.append("После очистки не осталось ни одной пригодной строки.")
        return pd.DataFrame(columns=[DATE, SKU, QTY, STOCK]), rep

    # --- слой 3.2: возвраты -------------------------------------------------
    neg = df[QTY] < 0
    rep.returns_zeroed = int(neg.sum())
    df[FLAG_RETURN] = neg.values
    df.loc[neg, QTY] = 0.0

    # --- слой 3.3: отрицательный остаток -----------------------------------
    if STOCK in df.columns:
        neg_stock = df[STOCK] < 0
        rep.negative_stock_fixed = int(neg_stock.sum())
        df.loc[neg_stock, STOCK] = 0.0

    # --- слой 3.4: склейка наименований ------------------------------------
    df = _merge_similar_skus(df, rep)

    # --- слой 3.5: дубли ----------------------------------------------------
    before = len(df)
    agg = {QTY: "sum", STOCK: "last", FLAG_RETURN: "max"}
    df = (df.groupby([SKU, DATE], as_index=False)
            .agg({k: v for k, v in agg.items() if k in df.columns}))
    rep.duplicates_merged = before - len(df)

    # --- слой 3.6: дозаполнение пропущенных дат ----------------------------
    frames = []
    for sku, g in df.groupby(SKU, sort=False):
        g = g.sort_values(DATE)
        if fill_missing_dates and len(g) > 1:
            full = pd.date_range(g[DATE].min(), g[DATE].max(), freq="D")
            g = (g.set_index(DATE).reindex(full).rename_axis(DATE).reset_index())
            filled = g[QTY].isna()
            rep.dates_filled += int(filled.sum())
            g[FLAG_FILLED] = filled.values
            g[QTY] = g[QTY].fillna(0.0)
            g[SKU] = sku
            if STOCK in g.columns:
                # остаток тянем вперёд: между поставками он меняется плавно
                g[STOCK] = g[STOCK].ffill()
            if FLAG_RETURN in g.columns:
                g[FLAG_RETURN] = g[FLAG_RETURN].fillna(False).astype(bool)
        else:
            g[FLAG_FILLED] = False
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)

    # --- слой 3.7: выбросы --------------------------------------------------
    df[FLAG_WINSORIZED] = False
    if winsorize:
        parts = []
        for _, g in df.groupby(SKU, sort=False):
            g = g.copy()
            vals, flag = _winsorize_group(g[QTY])
            g[QTY] = vals
            g[FLAG_WINSORIZED] = flag.values
            rep.outliers_winsorized += int(flag.sum())
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)

    # --- слой 4: цензурированный спрос -------------------------------------
    if STOCK in df.columns:
        censored = (df[QTY] <= 0) & df[STOCK].notna() & (df[STOCK] <= 0)
    else:
        censored = pd.Series(False, index=df.index)
    df[FLAG_CENSORED] = censored.values
    rep.censored_days = int(censored.sum())

    df = df.sort_values([SKU, DATE]).reset_index(drop=True)

    # --- слой 5: сводка -----------------------------------------------------
    rep.rows_final = len(df)
    rep.sku_count = int(df[SKU].nunique())
    rep.date_min = str(df[DATE].min().date())
    rep.date_max = str(df[DATE].max().date())

    for sku, g in df.groupby(SKU, sort=False):
        days = int((g[DATE].max() - g[DATE].min()).days) + 1
        cens = int(g[FLAG_CENSORED].sum())
        rep.per_sku.append({
            "sku": sku,
            "days": days,
            "rows": len(g),
            "total_qty": float(g[QTY].sum()),
            "mean_daily": float(g[QTY].mean()),
            "zero_share": float((g[QTY] <= 0).mean()),
            "censored_days": cens,
            "mode": pick_mode(days),
        })

    short = [s["sku"] for s in rep.per_sku if s["days"] < MIN_HISTORY_DAYS]
    if short:
        rep.warnings.append(
            f"Слишком короткая история (<{MIN_HISTORY_DAYS} дней) у товаров: "
            + ", ".join(short[:5]) + (" и др." if len(short) > 5 else "")
            + ". Прогноз по ним не строится.")

    if rep.censored_days:
        rep.warnings.append(
            f"Обнаружено {rep.censored_days} дней дефицита (продажи 0 при нулевом "
            "остатке). Реальный спрос в эти дни был выше зафиксированного — "
            "они исключены из обучения, чтобы модель не занижала прогноз.")

    heavy_zero = [s["sku"] for s in rep.per_sku if s["zero_share"] > 0.7]
    if heavy_zero:
        rep.warnings.append(
            "Больше 70% дней без продаж у товаров: " + ", ".join(heavy_zero[:5])
            + ". Для таких рядов точечный прогноз малоинформативен, "
              "используйте агрегат за горизонт.")

    return df, rep


def pick_mode(history_days: int) -> str:
    """Выбор режима гибридной модели (вариант C) по длине истории."""
    if history_days < MIN_HISTORY_DAYS:
        return MODE_REJECT
    if history_days < STATS_HISTORY_DAYS:
        return MODE_STATS
    if history_days < GLOBAL_HISTORY_DAYS:
        return MODE_GLOBAL
    return MODE_FINE_TUNE


def load_and_clean(source, filename: str | None = None, **kwargs):
    """Удобная обёртка: файл -> (чистый DataFrame, отчёт)."""
    from .ingest import read_table
    return clean(read_table(source, filename), **kwargs)
