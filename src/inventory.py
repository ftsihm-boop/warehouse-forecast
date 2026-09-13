"""
Модуль расчёта закупки.

Работает поверх квантильного прогноза: на вход приходят q50 (медиана
спроса за период) и q90 (верхняя граница), на выход — сколько заказать,
когда заказать и какой статус показать менеджеру.

ФОРМУЛЫ

    Страховой запас (safety stock):
        SS = q90(спрос за L) - q50(спрос за L)

    где L — срок поставки (lead time) в днях. Это ключевое отличие от
    учебной формулы SS = z * sigma * sqrt(L): та требует нормального
    распределения спроса, которого у штучных товаров нет. Квантильная
    разность берёт распределение из самих данных.

    Точка перезаказа (reorder point):
        ROP = q50(спрос за L) + SS

    Рекомендуемый объём заказа (модель периодического пополнения):
        Q = q90(спрос за L + R) - остаток - товар_в_пути

    где R — период между заказами (review period). Заказываем так, чтобы
    покрыть спрос до следующей поставки с вероятностью 90%.

    Экономичный размер партии (EOQ, формула Уилсона):
        EOQ = sqrt(2 * D * S / H)

    D — годовой спрос, S — стоимость размещения заказа, H — стоимость
    хранения единицы в год. Применяем как ограничение снизу на размер
    партии: заказывать по 3 штуки каждый день невыгодно из-за логистики.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

STATUS_CRITICAL = "critical"   # остаток уже ниже спроса за срок поставки
STATUS_ALERT = "alert"         # остаток ниже точки перезаказа — пора заказывать
STATUS_OK = "ok"               # запас в норме
STATUS_EXCESS = "excess"       # запаса больше, чем нужно на 2 горизонта

STATUS_LABELS_RU = {
    STATUS_CRITICAL: "Критично",
    STATUS_ALERT: "Пополнить",
    STATUS_OK: "Норма",
    STATUS_EXCESS: "Избыток",
}


@dataclass
class OrderPlan:
    sku: str
    stock: float
    forecast_horizon_q50: float     # прогноз спроса за горизонт (медиана)
    forecast_horizon_q90: float
    demand_lead_q50: float          # спрос за срок поставки
    safety_stock: float
    reorder_point: float
    recommended_order: float
    days_of_supply: float           # на сколько дней хватит текущего остатка
    stockout_date: str | None       # ожидаемая дата исчерпания запаса
    eoq: float | None
    status: str
    mode: str                       # режим модели: fine_tune / global / stats
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status_label"] = STATUS_LABELS_RU.get(self.status, self.status)
        return d


def _scale(value: float, from_days: int, to_days: int) -> float:
    """Пересчёт прогноза с одного горизонта на другой (линейно по дням)."""
    if from_days <= 0:
        return 0.0
    return float(value) * to_days / from_days


def compute_order(
    sku: str,
    q50_horizon: float,
    q90_horizon: float,
    horizon_days: int,
    stock: float | None,
    lead_time_days: int = 7,
    review_period_days: int = 7,
    in_transit: float = 0.0,
    mode: str = "global",
    order_cost: float | None = None,
    holding_cost_per_unit_year: float | None = None,
    min_order_qty: float = 0.0,
    order_multiple: float = 1.0,
) -> OrderPlan:
    """Считает план закупки по одному товару."""
    q50_horizon = max(float(q50_horizon), 0.0)
    q90_horizon = max(float(q90_horizon), q50_horizon)
    has_stock = stock is not None and np.isfinite(stock)
    stock_val = float(stock) if has_stock else 0.0

    cover = lead_time_days + review_period_days
    demand_lead_q50 = _scale(q50_horizon, horizon_days, lead_time_days)
    demand_lead_q90 = _scale(q90_horizon, horizon_days, lead_time_days)
    demand_cover_q90 = _scale(q90_horizon, horizon_days, cover)

    safety_stock = max(demand_lead_q90 - demand_lead_q50, 0.0)
    reorder_point = demand_lead_q50 + safety_stock

    need = demand_cover_q90 - stock_val - float(in_transit)
    recommended = max(need, 0.0)

    # EOQ как нижняя граница партии
    eoq = None
    if order_cost and holding_cost_per_unit_year and holding_cost_per_unit_year > 0:
        annual = _scale(q50_horizon, horizon_days, 365)
        if annual > 0:
            eoq = float(np.sqrt(2 * annual * order_cost / holding_cost_per_unit_year))
            if 0 < recommended < eoq:
                recommended = eoq

    if recommended > 0:
        recommended = max(recommended, min_order_qty)
        if order_multiple > 1:
            recommended = float(np.ceil(recommended / order_multiple) * order_multiple)
    recommended = float(np.ceil(recommended))

    daily = q50_horizon / horizon_days if horizon_days else 0.0
    days_supply = float(stock_val / daily) if daily > 0 and has_stock else float("inf")
    stockout_date = None
    if np.isfinite(days_supply) and days_supply < 365:
        stockout_date = str((pd.Timestamp.today().normalize()
                             + pd.Timedelta(days=int(days_supply))).date())

    if not has_stock:
        status = STATUS_ALERT if recommended > 0 else STATUS_OK
        note = "Остаток неизвестен — показана полная потребность за период."
    elif stock_val < demand_lead_q50:
        status = STATUS_CRITICAL
        note = "Запаса не хватит даже до прихода поставки."
    elif stock_val < reorder_point:
        status = STATUS_ALERT
        note = "Остаток ниже точки перезаказа."
    elif stock_val > 2 * q90_horizon and q90_horizon > 0:
        status = STATUS_EXCESS
        note = "Запас превышает двойной прогноз спроса — замороженные средства."
    else:
        status = STATUS_OK
        note = ""

    return OrderPlan(
        sku=sku,
        stock=round(stock_val, 2) if has_stock else float("nan"),
        forecast_horizon_q50=round(q50_horizon, 1),
        forecast_horizon_q90=round(q90_horizon, 1),
        demand_lead_q50=round(demand_lead_q50, 1),
        safety_stock=round(safety_stock, 1),
        reorder_point=round(reorder_point, 1),
        recommended_order=recommended,
        days_of_supply=round(days_supply, 1) if np.isfinite(days_supply) else -1.0,
        stockout_date=stockout_date,
        eoq=round(eoq, 1) if eoq else None,
        status=status,
        mode=mode,
        note=note,
    )


def build_order_plan(forecasts: pd.DataFrame, stocks: dict[str, float] | None = None,
                     horizon_days: int = 30, **kwargs) -> pd.DataFrame:
    """
    forecasts: DataFrame с колонками sku, q50, q90, mode
    stocks:    текущий остаток по каждому SKU (если None — берётся из forecasts.stock)
    """
    plans = []
    for _, r in forecasts.iterrows():
        sku = r["sku"]
        stock = None
        if stocks is not None:
            stock = stocks.get(sku)
        elif "stock" in forecasts.columns:
            stock = r["stock"]
        if stock is not None and not np.isfinite(stock):
            stock = None
        plans.append(compute_order(
            sku=sku, q50_horizon=r["q50"], q90_horizon=r["q90"],
            horizon_days=horizon_days, stock=stock,
            mode=r.get("mode", "global"), **kwargs).to_dict())

    out = pd.DataFrame(plans)
    if out.empty:
        return out
    order = {STATUS_CRITICAL: 0, STATUS_ALERT: 1, STATUS_EXCESS: 2, STATUS_OK: 3}
    out["_o"] = out["status"].map(order).fillna(9)
    return out.sort_values(["_o", "sku"]).drop(columns="_o").reset_index(drop=True)
