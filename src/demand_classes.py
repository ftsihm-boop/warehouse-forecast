"""
Автоматическая классификация товаров по ХАРАКТЕРУ СПРОСА.

ЗАЧЕМ ЭТО НУЖНО
---------------
Не всякий товар вообще поддаётся прогнозированию. Молоко продаётся каждый
день примерно одинаково — его спрос предсказуем. А коллекционный брелок
продаётся «0, 0, 0, 0, 4, 0, 0, 0, 0, 3» — тут прогнозировать нечего,
и любая модель, которая попытается, выдаст раздутый страховой запас:
она увидит редкий всплеск в 4 штуки и решит, что нужно держать запас
на такой случай. Деньги замораживаются, эффект уходит в минус.

Ручной порог «брать только товары, которые продаются чаще N раз в день»
эту задачу решает грубо: он смотрит только на ОБЪЁМ и не видит РЕГУЛЯРНОСТЬ.
Товар может продаваться в среднем 2 штуки в день и быть при этом
совершенно непрогнозируемым (одна партия в 60 штук раз в месяц).

МЕТОДИКА
--------
Используется классификация Syntetos-Boylan-Croston (SBC, 2005) —
стандарт в управлении запасами. Она смотрит на два независимых свойства
ряда продаж:

    ADI  (Average Demand Interval) — средний интервал между продажами.
         ADI = число дней / число дней с ненулевыми продажами.
         ADI = 1.0  -> продаётся каждый день
         ADI = 5.0  -> в среднем раз в 5 дней

    CV²  (квадрат коэффициента вариации) — насколько скачет РАЗМЕР продажи
         в те дни, когда она была. CV² = (std / mean)² по ненулевым дням.
         CV² ~ 0    -> всегда продаётся примерно одинаковое количество
         CV² > 0.49 -> размер продажи непредсказуем

Пороги ADI = 1.32 и CV² = 0.49 — канонические из оригинальной работы.
Пересечение двух признаков даёт четыре класса:

                   CV² < 0.49            CV² >= 0.49
    ADI < 1.32     SMOOTH                ERRATIC
                   (гладкий)             (неравномерный)
    ADI >= 1.32    INTERMITTENT          LUMPY
                   (прерывистый)         (комковатый)

    SMOOTH        — продаётся регулярно и ровно. Идеал для ML-прогноза.
    ERRATIC       — продаётся регулярно, но объём скачет. Прогноз возможен,
                    страховой запас будет большим — это честно отражает риск.
    INTERMITTENT  — продаётся редко, но ровными порциями. ML применять
                    нельзя, работает простое правило min/max.
    LUMPY         — редко И непредсказуемо. Не прогнозируется в принципе;
                    для таких товаров в реальной практике применяют
                    закупку «под заказ», а не прогнозную.

ABC-АНАЛИЗ
----------
Дополнительно считается вклад товара в общий оборот (правило Парето):
    A — товары, дающие первые 80% оборота
    B — следующие 15%
    C — последние 5% (длинный хвост из сотен позиций)

Итоговое решение принимается по ОБОИМ признакам: класс C + LUMPY —
это гарантированный кандидат на исключение, а вот A + LUMPY стоит
показать менеджеру отдельно: товар важный, но непрогнозируемый.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .schema import DATE, FLAG_CENSORED, QTY, SKU

# --- Канонические пороги SBC --------------------------------------------------
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

# Минимум ненулевых продаж, чтобы статистика вообще имела смысл
MIN_NONZERO_DAYS = 5

# --- Классы спроса ------------------------------------------------------------
SMOOTH = "smooth"
ERRATIC = "erratic"
INTERMITTENT = "intermittent"
LUMPY = "lumpy"
NO_DATA = "no_data"

CLASS_LABELS_RU = {
    SMOOTH: "Гладкий (регулярный)",
    ERRATIC: "Неравномерный по объёму",
    INTERMITTENT: "Прерывистый (редкий)",
    LUMPY: "Комковатый (непрогнозируемый)",
    NO_DATA: "Недостаточно продаж",
}

CLASS_SHORT_RU = {
    SMOOTH: "гладкий",
    ERRATIC: "неравномерный",
    INTERMITTENT: "прерывистый",
    LUMPY: "комковатый",
    NO_DATA: "нет данных",
}

# Какие классы пригодны для ML-прогноза, а какие нет
ML_SUITABLE = {SMOOTH, ERRATIC}
STATS_ONLY = {INTERMITTENT}
NOT_FORECASTABLE = {LUMPY, NO_DATA}

# Рекомендуемая стратегия закупки по классу — идёт прямо в интерфейс
STRATEGY_RU = {
    SMOOTH: "ML-прогноз спроса, автоматический расчёт заказа",
    ERRATIC: "ML-прогноз с увеличенным страховым запасом",
    INTERMITTENT: "Простое правило min/max, прогноз ненадёжен",
    LUMPY: "Закупка под заказ, автопрогноз не применяется",
    NO_DATA: "Накопить историю продаж (нужно от 5 дней с продажами)",
}

# --- ABC ----------------------------------------------------------------------
ABC_A_SHARE = 0.80
ABC_B_SHARE = 0.95


@dataclass
class DemandProfile:
    """Профиль спроса одного товара."""

    sku: str
    demand_class: str
    abc_class: str
    adi: float                 # средний интервал между продажами, дней
    cv2: float                 # квадрат коэффициента вариации размера продажи
    days: int                  # длина истории в днях
    nonzero_days: int          # в скольких днях были продажи
    zero_share: float          # доля дней без продаж
    total_qty: float           # всего продано за период
    mean_daily: float          # среднедневные продажи
    revenue_share: float       # доля в общем обороте (в штуках)
    forecastable: bool         # можно ли строить ML-прогноз
    recommendation: str        # что делать с этим товаром

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class_label"] = CLASS_LABELS_RU.get(self.demand_class, self.demand_class)
        return d


def classify_series(qty: pd.Series, censored: pd.Series | None = None
                    ) -> tuple[str, float, float, int]:
    """
    Классифицирует один ряд продаж.

    Возвращает (класс, ADI, CV², число ненулевых дней).

    Дни дефицита (censored) исключаются из расчёта: в такой день продажи
    были нулевыми не потому, что спроса не было, а потому что товара
    не было на складе. Если их считать честными нулями, любой товар
    выглядит более «прерывистым», чем он есть.
    """
    q = pd.Series(qty).astype(float)
    if censored is not None:
        q = q[~pd.Series(censored).astype(bool).to_numpy()]

    q = q.dropna()
    n_periods = len(q)
    if n_periods == 0:
        return NO_DATA, float("nan"), float("nan"), 0

    nonzero = q[q > 0]
    n_nonzero = len(nonzero)

    if n_nonzero < MIN_NONZERO_DAYS:
        return NO_DATA, float("nan"), float("nan"), n_nonzero

    # ADI: сколько в среднем дней проходит между продажами
    adi = n_periods / n_nonzero

    # CV²: насколько скачет размер продажи в дни, когда она была
    mean_nz = float(nonzero.mean())
    if mean_nz <= 0:
        return NO_DATA, adi, float("nan"), n_nonzero
    std_nz = float(nonzero.std(ddof=1)) if n_nonzero > 1 else 0.0
    cv2 = (std_nz / mean_nz) ** 2

    # Пересечение двух признаков даёт класс
    if adi < ADI_THRESHOLD:
        cls = SMOOTH if cv2 < CV2_THRESHOLD else ERRATIC
    else:
        cls = INTERMITTENT if cv2 < CV2_THRESHOLD else LUMPY

    return cls, adi, cv2, n_nonzero


def _abc_classes(totals: pd.Series) -> dict[str, str]:
    """ABC-анализ по вкладу в оборот (правило Парето)."""
    if totals.empty or totals.sum() <= 0:
        return {sku: "C" for sku in totals.index}

    ordered = totals.sort_values(ascending=False)
    cumulative = ordered.cumsum() / ordered.sum()

    out: dict[str, str] = {}
    for sku, cum in cumulative.items():
        if cum <= ABC_A_SHARE:
            out[sku] = "A"
        elif cum <= ABC_B_SHARE:
            out[sku] = "B"
        else:
            out[sku] = "C"
    return out


def profile_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит профиль спроса по каждому товару в очищенном DataFrame.

    Возвращает таблицу с колонками DemandProfile, отсортированную
    по обороту (самые значимые товары сверху).
    """
    if df.empty:
        return pd.DataFrame()

    totals = df.groupby(SKU)[QTY].sum()
    abc = _abc_classes(totals)
    grand_total = float(totals.sum()) or 1.0

    rows: list[dict] = []
    for sku, g in df.groupby(SKU, sort=False):
        g = g.sort_values(DATE)
        censored = g[FLAG_CENSORED] if FLAG_CENSORED in g.columns else None
        cls, adi, cv2, n_nonzero = classify_series(g[QTY], censored)

        days = int((g[DATE].max() - g[DATE].min()).days) + 1
        total = float(g[QTY].sum())

        rows.append(DemandProfile(
            sku=sku,
            demand_class=cls,
            abc_class=abc.get(sku, "C"),
            adi=round(adi, 2) if np.isfinite(adi) else float("nan"),
            cv2=round(cv2, 2) if np.isfinite(cv2) else float("nan"),
            days=days,
            nonzero_days=n_nonzero,
            zero_share=round(float((g[QTY] <= 0).mean()), 3),
            total_qty=total,
            mean_daily=round(total / max(days, 1), 2),
            revenue_share=round(total / grand_total, 4),
            forecastable=cls in ML_SUITABLE,
            recommendation=STRATEGY_RU[cls],
        ).to_dict())

    out = pd.DataFrame(rows)
    return out.sort_values("total_qty", ascending=False).reset_index(drop=True)


def split_by_forecastability(
    df: pd.DataFrame,
    keep_classes: set[str] | None = None,
    min_history_days: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Делит данные на пригодные и непригодные для ML-прогноза.

    Возвращает (данные_для_модели, данные_исключённых, профиль_всех_товаров).

    Исключённые товары НЕ выбрасываются из системы: они по-прежнему
    показываются менеджеру, просто с пометкой «прогноз не строится,
    работает правило min/max». Пропадать из интерфейса товар не должен —
    менеджеру всё равно нужно видеть его остаток.
    """
    keep = keep_classes if keep_classes is not None else ML_SUITABLE
    profiles = profile_all(df)
    if profiles.empty:
        return df, df.iloc[0:0], profiles

    ok_mask = (profiles["demand_class"].isin(keep)
               & (profiles["days"] >= min_history_days))
    ok_skus = set(profiles.loc[ok_mask, "sku"])

    keep_df = df[df[SKU].isin(ok_skus)].copy()
    drop_df = df[~df[SKU].isin(ok_skus)].copy()
    return keep_df, drop_df, profiles


def summary_text(profiles: pd.DataFrame) -> str:
    """Человекочитаемая сводка — идёт в консоль и в отчёт курсовой."""
    if profiles.empty:
        return "Нет данных для классификации."

    n = len(profiles)
    lines = [
        "КЛАССИФИКАЦИЯ ТОВАРОВ ПО ХАРАКТЕРУ СПРОСА",
        "(методика Syntetos-Boylan-Croston: ADI + CV²)",
        "=" * 62,
    ]

    counts = profiles["demand_class"].value_counts()
    share = profiles.groupby("demand_class")["revenue_share"].sum()

    for cls in (SMOOTH, ERRATIC, INTERMITTENT, LUMPY, NO_DATA):
        c = int(counts.get(cls, 0))
        if not c:
            continue
        s = float(share.get(cls, 0.0)) * 100
        mark = "прогнозируем" if cls in ML_SUITABLE else "не прогнозируем"
        lines.append(f"  {CLASS_LABELS_RU[cls]:<32} {c:>5} шт "
                     f"({s:>5.1f}% оборота)  — {mark}")

    forecastable = profiles["forecastable"]
    n_ok = int(forecastable.sum())
    share_ok = float(profiles.loc[forecastable, "revenue_share"].sum()) * 100

    lines += [
        "-" * 62,
        f"  Всего товаров:                   {n:>5}",
        f"  Пригодны для ML-прогноза:        {n_ok:>5} "
        f"({n_ok / n * 100:.1f}% позиций, {share_ok:.1f}% оборота)",
        f"  Работают по правилу min/max:     {n - n_ok:>5}",
    ]

    # Важное предупреждение: товар крупный, но непрогнозируемый
    big_lumpy = profiles[(profiles["abc_class"] == "A")
                         & (~profiles["forecastable"])]
    if len(big_lumpy):
        lines += [
            "-" * 62,
            f"  ВНИМАНИЕ: {len(big_lumpy)} товар(ов) из группы A "
            "(верхние 80% оборота)",
            "  имеют непрогнозируемый спрос — их стоит разобрать вручную:",
        ]
        for sku in big_lumpy["sku"].head(5):
            lines.append(f"    · {sku[:58]}")

    return "\n".join(lines)
