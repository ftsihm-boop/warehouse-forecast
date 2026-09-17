"""
Бэктест политики закупки и расчёт экономического эффекта.

Это единственный честный способ защитить цифру «экономия N рублей в месяц».
Прогоняем по истории две политики и сравниваем их результат на одних
и тех же реальных продажах:

    БАЗОВАЯ (as-is)   — так закупают вручную: «средний спрос за месяц × запас
                        прочности», пересмотр раз в R дней.
    НАША (to-be)      — ML-прогноз + страховой запас по квантилям.

Считаем по каждой политике:
    * средний товарный остаток (замороженные оборотные средства)
    * количество дней дефицита и упущенные продажи в штуках
    * стоимость хранения
    * упущенную выручку

Разница между политиками и есть экономический эффект — не выдуманный,
а посчитанный на фактических продажах.

ВАЖНО: в бэктесте нельзя использовать будущее. Прогноз на дату T строится
только по данным до T включительно. За это отвечает walk-forward цикл ниже.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .demand_classes import profile_all
from .forecast import stats_forecast
from .model import QuantileModel
from .features import BASE_COL, FEATURE_COLUMNS, inference_rows
from .schema import DATE, FLAG_CENSORED, QTY, SKU


@dataclass
class Economics:
    """Экономические допущения. Все — параметры, а не магические числа."""
    unit_cost: float = 60.0            # закупочная цена единицы, ₽
    margin: float = 0.25               # наценка (доля от цены продажи)
    holding_rate_year: float = 0.35    # стоимость хранения, % от стоимости в год
    order_cost: float = 300.0          # стоимость размещения одного заказа, ₽

    @property
    def price(self) -> float:
        return self.unit_cost / (1 - self.margin)

    @property
    def profit_per_unit(self) -> float:
        return self.price - self.unit_cost

    def holding_cost_per_unit_day(self) -> float:
        return self.unit_cost * self.holding_rate_year / 365


@dataclass
class PolicyResult:
    name: str
    avg_stock_units: float
    avg_stock_value: float
    stockout_days: int
    lost_units: float
    lost_revenue: float
    holding_cost: float
    orders_placed: int
    ordering_cost: float
    total_cost: float
    service_level: float   # доля дней без дефицита

    def to_dict(self) -> dict:
        return asdict(self)


def _simulate(demand: np.ndarray, dates: pd.DatetimeIndex,
              order_fn, lead_time: int, review_period: int,
              econ: Economics, init_stock: float | None = None) -> PolicyResult:
    """
    Общий движок симуляции. order_fn(t) -> объём заказа, размещаемого в день t
    (приходит через lead_time дней). Заказы размещаются раз в review_period дней.
    """
    n = len(demand)
    stock = float(init_stock if init_stock is not None
                  else demand[:28].mean() * (lead_time + review_period))
    pipeline = np.zeros(n + lead_time + 1)

    stock_hist, lost, stockout_days, orders = [], 0.0, 0, 0

    for t in range(n):
        stock += pipeline[t]
        d = demand[t]
        sold = min(stock, d)
        if d > stock:
            lost += d - stock
            stockout_days += 1
        stock -= sold
        stock_hist.append(stock)

        if t % review_period == 0:
            qty = order_fn(t, stock, float(pipeline[t + 1:t + lead_time + 1].sum()))
            if qty > 0:
                pipeline[min(t + lead_time, n + lead_time)] += qty
                orders += 1

    avg_stock = float(np.mean(stock_hist))
    holding = float(np.sum(stock_hist) * econ.holding_cost_per_unit_day())
    ordering = orders * econ.order_cost
    lost_rev = lost * econ.profit_per_unit

    return PolicyResult(
        name="", avg_stock_units=round(avg_stock, 1),
        avg_stock_value=round(avg_stock * econ.unit_cost, 0),
        stockout_days=stockout_days, lost_units=round(lost, 1),
        lost_revenue=round(lost_rev, 0), holding_cost=round(holding, 0),
        orders_placed=orders, ordering_cost=round(ordering, 0),
        total_cost=round(holding + ordering + lost_rev, 0),
        service_level=round(1 - stockout_days / max(n, 1), 3),
    )


def backtest_sku(
    g: pd.DataFrame,
    model: QuantileModel | None,
    horizon: int = 30,
    lead_time: int = 7,
    review_period: int = 7,
    test_days: int = 180,
    econ: Economics | None = None,
    safety_factor_baseline: float = 1.25,
) -> dict:
    """Бэктест по одному товару: базовая политика против нашей."""
    econ = econ or Economics()
    g = g.sort_values(DATE).reset_index(drop=True)
    if len(g) < test_days + 90:
        return {}

    split = len(g) - test_days
    demand = g[QTY].astype(float).to_numpy()
    test_demand = demand[split:]
    dates = pd.DatetimeIndex(g[DATE].iloc[split:])

    cover = lead_time + review_period

    # --- базовая политика: среднее за 28 дней × запас прочности -----------
    def baseline_order(t, stock, in_transit):
        hist = demand[max(0, split + t - 28):split + t]
        avg = hist.mean() if len(hist) else 0.0
        target = avg * cover * safety_factor_baseline
        return max(target - stock - in_transit, 0.0)

    # --- наша политика: прогноз q90 на срок покрытия -----------------------
    # предрасчёт: прогноз пересматриваем раз в review_period дней,
    # каждый раз только по данным ДО текущего дня (walk-forward)
    ml_cover: dict[int, float] = {}
    for t in range(0, test_days, review_period):
        hist = g.iloc[:split + t]
        if model is not None and len(hist) >= 90:
            rows = inference_rows(hist, cover)
            rows = rows[rows[BASE_COL].notna() & (rows[BASE_COL] > 0)]
            if len(rows):
                feats = [c for c in (model.features or FEATURE_COLUMNS)
                         if c in rows.columns]
                _, hi = model.predict(rows[feats].fillna(0.0))
                ml_cover[t] = float(hi[0] * rows[BASE_COL].iloc[0] * cover)
                continue
        _, hi = stats_forecast(hist, cover)
        ml_cover[t] = hi

    def ml_order(t, stock, in_transit):
        key = (t // review_period) * review_period
        target = ml_cover.get(key, 0.0)
        return max(target - stock - in_transit, 0.0)

    init = float(test_demand[:28].mean() * cover) if len(test_demand) >= 28 else None

    base = _simulate(test_demand, dates, baseline_order, lead_time,
                     review_period, econ, init)
    base.name = "Ручное планирование (as-is)"
    ml = _simulate(test_demand, dates, ml_order, lead_time,
                   review_period, econ, init)
    ml.name = "ML-прогноз (to-be)"

    return {"sku": g[SKU].iloc[0], "baseline": base.to_dict(), "ml": ml.to_dict(),
            "days": int(test_days), "demand_total": float(test_demand.sum())}


def backtest(df: pd.DataFrame, model: QuantileModel | None = None,
             econ: Economics | None = None,
             only_forecastable: bool = True, **kwargs
             ) -> tuple[pd.DataFrame, dict]:
    """
    Бэктест по всем товарам. Возвращает (таблица по SKU, сводка).

    only_forecastable=True (по умолчанию) считает экономический эффект
    только по товарам, для которых ML-прогноз вообще применим. Это не
    «подгонка результата», а корректная постановка: система не обещает
    прогнозировать хаотичный спрос, поэтому и мерить её на таких товарах
    нечестно — они закупаются по правилу min/max, а не по прогнозу.
    Число исключённых позиций попадает в сводку, так что картина остаётся
    прозрачной.
    """
    econ = econ or Economics()
    rows, totals = [], {"baseline": {}, "ml": {}}

    excluded = 0
    if only_forecastable and not df.empty:
        profiles = profile_all(df)
        if not profiles.empty:
            ok = set(profiles.loc[profiles["forecastable"], "sku"])
            excluded = int(df[SKU].nunique() - len(ok))
            df = df[df[SKU].isin(ok)]

    for _, g in df.groupby(SKU, sort=False):
        r = backtest_sku(g, model, econ=econ, **kwargs)
        if not r:
            continue
        b, m = r["baseline"], r["ml"]
        rows.append({
            "Товар": r["sku"],
            "Средний остаток, as-is": b["avg_stock_units"],
            "Средний остаток, to-be": m["avg_stock_units"],
            "Дней дефицита, as-is": b["stockout_days"],
            "Дней дефицита, to-be": m["stockout_days"],
            "Упущено шт, as-is": b["lost_units"],
            "Упущено шт, to-be": m["lost_units"],
            "Итого затрат, as-is ₽": b["total_cost"],
            "Итого затрат, to-be ₽": m["total_cost"],
            "Эффект, ₽": round(b["total_cost"] - m["total_cost"]),
        })
        for key, res in (("baseline", b), ("ml", m)):
            for f in ("avg_stock_units", "avg_stock_value", "stockout_days",
                      "lost_units", "lost_revenue", "holding_cost",
                      "ordering_cost", "total_cost"):
                totals[key][f] = totals[key].get(f, 0.0) + res[f]

    table = pd.DataFrame(rows)
    if table.empty:
        return table, {}

    days = kwargs.get("test_days", 180)
    b, m = totals["baseline"], totals["ml"]
    summary = {
        "Период бэктеста, дней": days,
        "Товаров в расчёте эффекта": len(table),
        "Исключено (спрос непрогнозируем)": excluded,
        "Средний остаток as-is, ₽": round(b["avg_stock_value"]),
        "Средний остаток to-be, ₽": round(m["avg_stock_value"]),
        "Снижение остатка, %": round(
            (1 - m["avg_stock_value"] / max(b["avg_stock_value"], 1)) * 100, 1),
        "Упущенная выручка as-is, ₽": round(b["lost_revenue"]),
        "Упущенная выручка to-be, ₽": round(m["lost_revenue"]),
        # Процент считаем только если базе есть что снижать: при потерях
        # около нуля относительная метрика теряет смысл (0 -> 6 руб. это
        # не «минус 500%», а статистический шум).
        "Снижение потерь от дефицита, %": (
            round((1 - m["lost_revenue"] / b["lost_revenue"]) * 100, 1)
            if b["lost_revenue"] >= 100 else "н/д (потерь почти нет)"),
        "Затраты на хранение as-is, ₽": round(b["holding_cost"]),
        "Затраты на хранение to-be, ₽": round(m["holding_cost"]),
        "Совокупный эффект за период, ₽": round(b["total_cost"] - m["total_cost"]),
        "Эффект в месяц, ₽": round((b["total_cost"] - m["total_cost"]) / days * 30),
    }
    return table, summary
