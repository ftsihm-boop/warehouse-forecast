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

ЗАКАЗ В ДНЯХ
------------
У каждого товара свой срок поставки L, поэтому «пора заказывать» наступает
в разные дни. Для менеджера это главный вопрос — не «какой у меня запас»,
а «до какого числа нужно успеть оформить заявку»:

    дней запаса          = остаток / спрос в день
    заказать через       = дней запаса - L - страховой запас в днях
    заказать до (дата)   = сегодня + «заказать через»
    придёт (если сегодня) = сегодня + L
    хватит после заказа  = (остаток + в пути + заказ) / спрос в день

«Сегодня» — это день, следующий за последней датой в файле: именно с него
начинается прогноз. Если выгрузка свежая, это настоящая сегодняшняя дата.
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
    # --- заказ в днях --------------------------------------------------
    lead_time_days: int = 7         # срок поставки именно этого товара
    review_period_days: int = 7
    in_transit: float = 0.0         # уже заказано и едет
    stock_known: bool = True
    daily_demand: float = 0.0       # ожидаемый спрос в день (медиана)
    order_in_days: int | None = None      # через сколько дней нужно заказать
    order_by_date: str | None = None      # до какой даты успеть с заказом
    arrival_date: str | None = None       # когда придёт, если заказать сегодня
    stockout_gap_days: int = 0            # сколько дней не будет товара до прихода
    expected_lost_units: float = 0.0      # сколько не продадим за эти дни
    days_after_order: float = -1.0        # на сколько дней хватит с учётом заказа
    days_after_order_high: float = -1.0   # то же, если спрос окажется высоким
    covered_until: str | None = None      # до какой даты хватит с учётом заказа

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
    service_factor: float = 1.0,
    as_of: pd.Timestamp | None = None,
) -> OrderPlan:
    """
    Считает план закупки по одному товару.

    as_of — «сегодня» для расчёта дат. По умолчанию настоящая сегодняшняя
    дата, но сервер передаёт день после последней даты в файле: прогноз
    строится именно от него, и даты заказа должны лежать на той же оси.
    """
    lead_time_days = int(max(1, round(float(lead_time_days))))
    review_period_days = int(max(1, round(float(review_period_days))))
    today = (pd.Timestamp(as_of).normalize() if as_of is not None
             else pd.Timestamp.today().normalize())
    q50_horizon = max(float(q50_horizon), 0.0)
    q90_horizon = max(float(q90_horizon), q50_horizon)
    has_stock = stock is not None and np.isfinite(stock)
    stock_val = float(stock) if has_stock else 0.0

    cover = lead_time_days + review_period_days
    demand_lead_q50 = _scale(q50_horizon, horizon_days, lead_time_days)
    demand_lead_q90 = _scale(q90_horizon, horizon_days, lead_time_days)
    demand_cover_q90 = _scale(q90_horizon, horizon_days, cover)

    # service_factor масштабирует страховой запас под экономику товара:
    # у товара с высокой маржой и дешёвым хранением выгодно держать
    # больше, у скоропорта с копеечной наценкой — меньше. Подбирается
    # автоматически (см. economics.autotune_service_factor).
    raw_safety = max(demand_lead_q90 - demand_lead_q50, 0.0)
    safety_stock = raw_safety * float(service_factor)
    reorder_point = demand_lead_q50 + safety_stock

    demand_cover_q50 = _scale(q50_horizon, horizon_days, cover)
    cover_safety = max(demand_cover_q90 - demand_cover_q50, 0.0) * float(service_factor)
    need = demand_cover_q50 + cover_safety - stock_val - float(in_transit)
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
    daily_hi = q90_horizon / horizon_days if horizon_days else 0.0
    days_supply = float(stock_val / daily) if daily > 0 and has_stock else float("inf")

    def on_day(n: float) -> str:
        return str((today + pd.Timedelta(days=int(n))).date())

    stockout_date = on_day(days_supply) if (
        np.isfinite(days_supply) and days_supply < 365) else None

    # --- заказ в днях ----------------------------------------------------
    safety_days = safety_stock / daily if daily > 0 else 0.0
    order_in_days = order_by_date = None
    if has_stock and np.isfinite(days_supply):
        # через сколько дней остаток дойдёт до точки перезаказа
        order_in_days = int(max(0, np.floor(days_supply - lead_time_days
                                            - safety_days)))
        if order_in_days < 365:
            order_by_date = on_day(order_in_days)
        else:
            order_in_days = None

    # сколько дней товара не будет, даже если заказать прямо сейчас
    gap_days, lost_units = 0, 0.0
    if has_stock and daily > 0 and days_supply < lead_time_days:
        gap_days = int(np.ceil(lead_time_days - days_supply))
        lost_units = round(float((lead_time_days - days_supply) * daily), 1)

    # на сколько хватит, если докупить рекомендованное
    after_total = stock_val + float(in_transit) + recommended
    days_after = float(after_total / daily) if daily > 0 else float("inf")
    days_after_hi = float(after_total / daily_hi) if daily_hi > 0 else float("inf")
    covered_until = on_day(days_after) if (
        np.isfinite(days_after) and days_after < 730) else None

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
        lead_time_days=lead_time_days,
        review_period_days=review_period_days,
        in_transit=round(float(in_transit), 1),
        stock_known=bool(has_stock),
        daily_demand=round(daily, 2),
        order_in_days=order_in_days,
        order_by_date=order_by_date,
        arrival_date=on_day(lead_time_days),
        stockout_gap_days=gap_days,
        expected_lost_units=lost_units,
        days_after_order=round(days_after, 1) if np.isfinite(days_after) else -1.0,
        days_after_order_high=(round(days_after_hi, 1)
                               if np.isfinite(days_after_hi) else -1.0),
        covered_until=covered_until,
    )


def build_order_plan(forecasts: pd.DataFrame, stocks: dict[str, float] | None = None,
                     horizon_days: int = 30,
                     service_factors: dict[str, float] | None = None,
                     lead_times: dict[str, float] | None = None,
                     review_periods: dict[str, float] | None = None,
                     in_transit: dict[str, float] | None = None,
                     as_of: pd.Timestamp | None = None,
                     **kwargs) -> pd.DataFrame:
    """
    forecasts: DataFrame с колонками sku, q50, q90, mode
    stocks:    текущий остаток по каждому SKU (если None — берётся из forecasts.stock)
    service_factors: множитель страхового запаса по каждому товару.
               Считается автоматически из экономики товара
               (см. economics.derive_sku_economics). Если не передан,
               берётся общий из kwargs или 1.0.
    lead_times / review_periods: срок поставки и период между заказами
               по каждому товару, в днях. Для товара, которого в словаре
               нет, берётся общее значение из lead_time_days /
               review_period_days.
    in_transit: сколько штук уже заказано и едет к нам.
    """
    plans = []
    default_factor = kwargs.pop("service_factor", 1.0)
    default_lead = kwargs.pop("lead_time_days", 7)
    default_review = kwargs.pop("review_period_days", 7)
    lead_times = lead_times or {}
    review_periods = review_periods or {}
    in_transit = in_transit or {}

    for _, r in forecasts.iterrows():
        sku = r["sku"]
        stock = None
        if stocks is not None:
            stock = stocks.get(sku)
        elif "stock" in forecasts.columns:
            stock = r["stock"]
        if stock is not None and not np.isfinite(stock):
            stock = None

        factor = default_factor
        if service_factors:
            factor = service_factors.get(sku, default_factor)

        lead = lead_times.get(sku)
        review = review_periods.get(sku)
        plans.append(compute_order(
            sku=sku, q50_horizon=r["q50"], q90_horizon=r["q90"],
            horizon_days=horizon_days, stock=stock,
            mode=r.get("mode", "global"), service_factor=factor,
            lead_time_days=default_lead if lead is None or not np.isfinite(lead) else lead,
            review_period_days=(default_review if review is None
                                or not np.isfinite(review) else review),
            in_transit=float(in_transit.get(sku, 0.0) or 0.0),
            as_of=as_of, **kwargs).to_dict())

    out = pd.DataFrame(plans)
    if out.empty:
        return out
    order = {STATUS_CRITICAL: 0, STATUS_ALERT: 1, STATUS_EXCESS: 2, STATUS_OK: 3}
    out["_o"] = out["status"].map(order).fillna(9)
    # внутри статуса сначала то, что надо заказывать раньше всего
    out["_d"] = out["order_in_days"].fillna(9999)
    return (out.sort_values(["_o", "_d", "sku"])
               .drop(columns=["_o", "_d"]).reset_index(drop=True))
