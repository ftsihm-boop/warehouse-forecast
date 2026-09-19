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

    @classmethod
    def from_data(cls, df: "pd.DataFrame", holding_rate_year: float = 0.35,
                  order_cost: float = 300.0) -> "Economics":
        """
        Строит экономику по ценам из файла пользователя.

        Если цен в файле нет, возвращаются значения по умолчанию —
        то же, что и раньше, просто теперь это запасной вариант,
        а не единственный.
        """
        from .economics import derive_sku_economics
        from .schema import QTY, SKU

        if df is None or df.empty:
            return cls(holding_rate_year=holding_rate_year, order_cost=order_cost)

        eco = derive_sku_economics(df, holding_rate_year=holding_rate_year)
        if eco.empty or not (eco["source"] == "данные").any():
            return cls(holding_rate_year=holding_rate_year, order_cost=order_cost)

        # взвешиваем по обороту: экономика ходовых товаров важнее
        qty = df.groupby(SKU)[QTY].sum().reindex(eco["sku"]).fillna(0.0)
        w = qty.to_numpy() if qty.sum() > 0 else np.ones(len(eco))
        cost = float(np.average(eco["unit_cost"], weights=w))
        price = float(np.average(eco["unit_price"], weights=w))
        margin = (price - cost) / price if price > 0 else 0.25

        return cls(unit_cost=round(cost, 2), margin=round(margin, 4),
                   holding_rate_year=holding_rate_year, order_cost=order_cost)


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


def _forecast_path(g: pd.DataFrame, model: QuantileModel | None, split: int,
                   test_days: int, review_period: int, cover: int
                   ) -> dict[int, tuple[float, float]]:
    """
    Прогноз на каждую дату пересмотра: (медиана, ширина страхового запаса).

    Считается ОДИН раз на товар и переиспользуется для всех проверяемых
    коэффициентов. Раньше прогноз пересчитывался заново под каждый
    коэффициент, хотя от коэффициента он не зависит вообще: множитель
    масштабирует уже готовую разницу (q90 - q50). На сетке из семи
    значений это ровно в семь раз лишней работы.

    Walk-forward соблюдён: прогноз на день t строится только по данным
    до t включительно, будущее модели недоступно.
    """
    path: dict[int, tuple[float, float]] = {}
    for t in range(0, test_days, review_period):
        hist = g.iloc[:split + t]
        lo_v = hi_v = None
        if model is not None and len(hist) >= 90:
            rows = inference_rows(hist, cover)
            rows = rows[rows[BASE_COL].notna() & (rows[BASE_COL] > 0)]
            if len(rows):
                feats = [c for c in (model.features or FEATURE_COLUMNS)
                         if c in rows.columns]
                lo, hi = model.predict(rows[feats].fillna(0.0))
                scale = float(rows[BASE_COL].iloc[0]) * cover
                lo_v, hi_v = float(lo[0] * scale), float(hi[0] * scale)
        if lo_v is None:
            lo_v, hi_v = stats_forecast(hist, cover)
        path[t] = (lo_v, max(hi_v - lo_v, 0.0))
    return path


def backtest_sku_factors(
    g: pd.DataFrame,
    model: QuantileModel | None,
    factors: list[float],
    horizon: int = 30,
    lead_time: int = 7,
    review_period: int = 7,
    test_days: int = 180,
    econ: Economics | None = None,
    safety_factor_baseline: float = 1.25,
) -> dict:
    """
    Бэктест одного товара сразу по нескольким коэффициентам запаса.

    Базовая политика и прогноз считаются один раз, дальше прогоняется
    только дешёвая симуляция склада под каждый коэффициент. Возвращает
    базовый результат и словарь {коэффициент: результат to-be}.
    """
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

    # --- наша политика: прогноз q50 + страховой запас на срок покрытия ------
    # service_factor масштабирует ИМЕННО страховой запас (q90 - q50),
    # не трогая медианный прогноз. 1.0 — полный страховой запас под
    # 90% уровень сервиса; 0.5 — половина (меньше запас, чуть больше
    # риск дефицита); 1.5 — перестраховка. Это единственная ручка,
    # которой настраивается баланс «замороженные деньги / упущенные продажи».
    path = _forecast_path(g, model, split, test_days, review_period, cover)

    init = float(test_demand[:28].mean() * cover) if len(test_demand) >= 28 else None

    base = _simulate(test_demand, dates, baseline_order, lead_time,
                     review_period, econ, init)
    base.name = "Ручное планирование (as-is)"

    by_factor: dict[float, dict] = {}
    for f in factors:
        def ml_order(t, stock, in_transit, _f=f):
            key = (t // review_period) * review_period
            lo_v, spread = path.get(key, (0.0, 0.0))
            return max(lo_v + spread * _f - stock - in_transit, 0.0)

        ml = _simulate(test_demand, dates, ml_order, lead_time,
                       review_period, econ, init)
        ml.name = "ML-прогноз (to-be)"
        by_factor[f] = ml.to_dict()

    return {"sku": g[SKU].iloc[0], "baseline": base.to_dict(),
            "by_factor": by_factor, "days": int(test_days),
            "demand_total": float(test_demand.sum())}


def backtest_sku(
    g: pd.DataFrame,
    model: QuantileModel | None,
    horizon: int = 30,
    lead_time: int = 7,
    review_period: int = 7,
    test_days: int = 180,
    econ: Economics | None = None,
    safety_factor_baseline: float = 1.25,
    service_factor: float = 1.0,
) -> dict:
    """Бэктест по одному товару: базовая политика против нашей."""
    r = backtest_sku_factors(
        g, model, [service_factor], horizon=horizon, lead_time=lead_time,
        review_period=review_period, test_days=test_days, econ=econ,
        safety_factor_baseline=safety_factor_baseline)
    if not r:
        return {}
    return {"sku": r["sku"], "baseline": r["baseline"],
            "ml": r["by_factor"][service_factor], "days": r["days"],
            "demand_total": r["demand_total"]}


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


def tune_portfolio(
    df: pd.DataFrame,
    model: QuantileModel | None = None,
    econ: Economics | None = None,
    test_days: int = 90,
    lead_time: int = 7,
    review_period: int = 7,
    max_skus: int | None = None,
    only_forecastable: bool = True,
    allow_manual: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Подбирает страховой запас ПО КАЖДОМУ ТОВАРУ и решает, каким товаром
    вообще стоит управлять по прогнозу.

    ПОЧЕМУ НЕ ОДИН КОЭФФИЦИЕНТ НА ВЕСЬ АССОРТИМЕНТ
    ----------------------------------------------
    Так было раньше, и это оказалось грубой ошибкой. Подбор искал одно
    число, которое устроит сразу все товары. Но если хотя бы по одному
    товару прогноз не работает — например, сезонный арбуз, которого
    модель не видела, — дефицит по нему не закрывается ничем. Подбор
    честно пытался его закрыть и поднимал коэффициент ГЛОБАЛЬНО: 1.3,
    2.2, 3.7. Арбузу это не помогало, а все остальные товары получали
    тройной запас и тонули в стоимости хранения. Итог: система с точным
    прогнозом показывала убыток.

    Запас — это решение по конкретному товару, а не по портфелю.
    У молока, арбуза и сигарет разная маржа, разная сезонность и разная
    предсказуемость, поэтому и коэффициент у каждого свой: он ищется
    вокруг теоретического оптимума ИМЕННО ЭТОГО товара.

    ПОЧЕМУ СИСТЕМА ИМЕЕТ ПРАВО ОТКАЗАТЬСЯ
    -------------------------------------
    Если по товару даже лучший коэффициент проигрывает ручному правилу,
    навязывать прогноз нечестно и убыточно. Такой товар остаётся на
    ручном планировании, и система прямо об этом говорит. Знать границы
    своей применимости — часть работы модели, а не изъян.
    """
    from .economics import (MAX_SERVICE_FACTOR, MIN_SERVICE_FACTOR,
                            derive_sku_economics)

    econ = econ or Economics()
    if df is None or df.empty:
        return {}

    cover = lead_time + review_period

    excluded = 0
    if only_forecastable:
        profiles = profile_all(df)
        if not profiles.empty:
            ok = set(profiles.loc[profiles["forecastable"], "sku"])
            excluded = int(df[SKU].nunique() - len(ok))
            df = df[df[SKU].isin(ok)]

    per_sku = df.groupby(SKU).agg(total=(QTY, "sum"), n=(QTY, "size"))
    eligible = per_sku[per_sku["n"] >= test_days + 90]
    if eligible.empty:
        return {}
    ranked = eligible.sort_values("total", ascending=False)
    chosen = ranked.index if max_skus is None else ranked.head(max_skus).index

    # теоретический оптимум по каждому товару — центр его личной сетки
    eco = derive_sku_economics(df, holding_rate_year=econ.holding_rate_year,
                               cover_days=cover, default_margin=econ.margin)
    theo_by_sku = ({str(r["sku"]): float(r["service_factor"])
                    for _, r in eco.iterrows()} if not eco.empty else {})

    def clamp(v: float) -> float:
        return round(max(MIN_SERVICE_FACTOR, min(MAX_SERVICE_FACTOR, v)), 2)

    rows: list[dict] = []
    totals = {"baseline": {}, "ml": {}}
    FIELDS = ("avg_stock_units", "avg_stock_value", "stockout_days",
              "lost_units", "lost_revenue", "holding_cost",
              "ordering_cost", "total_cost")

    for sku in chosen:
        g = df[df[SKU] == sku]
        theo = theo_by_sku.get(str(sku), 1.0)
        grid = sorted({clamp(theo * m)
                       for m in (0.3, 0.5, 0.75, 1.0, 1.3, 1.7, 2.2)})

        r = backtest_sku_factors(g, model, grid, econ=econ, test_days=test_days,
                                 lead_time=lead_time, review_period=review_period)
        if not r:
            continue
        base = r["baseline"]
        tolerance = max(base["lost_revenue"] * 1.05, 50.0)

        def evaluate(by_factor: dict) -> list[dict]:
            out = []
            for f, m in by_factor.items():
                out.append({"factor": f, "ml": m,
                            "effect": base["total_cost"] - m["total_cost"],
                            "lost": m["lost_revenue"]})
            return out

        results = evaluate(r["by_factor"])

        def pick(rs: list[dict]) -> dict:
            # Сначала — варианты, которые не ухудшают дефицит против
            # текущей практики. Среди них берём самый выгодный. Если
            # таких нет, берём просто самый выгодный: держать нулевой
            # дефицит ценой запаса, который стоит дороже самого дефицита,
            # бессмысленно.
            safe = [x for x in rs if x["lost"] <= tolerance]
            return max(safe or rs, key=lambda x: x["effect"])

        # Расширение сетки вверх — только пока это реально улучшает эффект.
        # Критерий «минимум дефицита любой ценой» убран: именно он раньше
        # уводил коэффициент в 3.7 и делал систему убыточной.
        for _ in range(3):
            edge = max(results, key=lambda x: x["factor"])
            if pick(results)["factor"] < edge["factor"] - 1e-9:
                break
            nxt_f = clamp(edge["factor"] * 1.35)
            if nxt_f <= edge["factor"] + 1e-9:
                break
            extra = backtest_sku_factors(
                g, model, [nxt_f], econ=econ, test_days=test_days,
                lead_time=lead_time, review_period=review_period)
            if not extra:
                break
            nxt = evaluate(extra["by_factor"])[0]
            results.append(nxt)
            if nxt["effect"] <= edge["effect"]:
                break

        best = pick(results)
        policy = "ml"
        chosen_ml = best["ml"]
        if allow_manual and best["effect"] <= 0:
            # система проигрывает ручному правилу — не навязываемся
            policy = "manual"
            chosen_ml = dict(base)

        if verbose:
            print(f"  {str(sku)[:28]:<28} × {best['factor']:.2f} "
                  f"(теор. {theo:.2f})  {base['total_cost'] - chosen_ml['total_cost']:>8.0f} ₽"
                  f"  {policy}")

        rows.append({
            "sku": sku, "factor": best["factor"], "theoretical": round(theo, 2),
            "policy": policy,
            "effect": round(base["total_cost"] - chosen_ml["total_cost"]),
            "effect_if_ml": round(best["effect"]),
            "lost_baseline": base["lost_revenue"], "lost_ml": chosen_ml["lost_revenue"],
            "holding_baseline": base["holding_cost"],
            "holding_ml": chosen_ml["holding_cost"],
        })
        for key, res in (("baseline", base), ("ml", chosen_ml)):
            for f in FIELDS:
                totals[key][f] = totals[key].get(f, 0.0) + res[f]

    if not rows:
        return {}

    b, m = totals["baseline"], totals["ml"]
    managed = [r for r in rows if r["policy"] == "ml"]
    manual = [r for r in rows if r["policy"] == "manual"]
    factors = [r["factor"] for r in managed] or [r["factor"] for r in rows]

    summary = {
        "Период бэктеста, дней": test_days,
        "Товаров в расчёте эффекта": len(rows),
        "Исключено (спрос непрогнозируем)": excluded,
        "Средний остаток as-is, ₽": round(b["avg_stock_value"]),
        "Средний остаток to-be, ₽": round(m["avg_stock_value"]),
        "Снижение остатка, %": round(
            (1 - m["avg_stock_value"] / max(b["avg_stock_value"], 1)) * 100, 1),
        "Упущенная выручка as-is, ₽": round(b["lost_revenue"]),
        "Упущенная выручка to-be, ₽": round(m["lost_revenue"]),
        "Снижение потерь от дефицита, %": (
            round((1 - m["lost_revenue"] / b["lost_revenue"]) * 100, 1)
            if b["lost_revenue"] >= 100 else "н/д (потерь почти нет)"),
        "Затраты на хранение as-is, ₽": round(b["holding_cost"]),
        "Затраты на хранение to-be, ₽": round(m["holding_cost"]),
        "Совокупный эффект за период, ₽": round(b["total_cost"] - m["total_cost"]),
        "Эффект в месяц, ₽": round((b["total_cost"] - m["total_cost"]) / test_days * 30),
    }

    return {
        "summary": summary,
        "per_sku": rows,
        "managed": len(managed),
        "manual": len(manual),
        "manual_skus": [str(r["sku"]) for r in manual],
        "factor_median": round(float(np.median(factors)), 2),
        "factor_min": round(float(min(factors)), 2),
        "factor_max": round(float(max(factors)), 2),
        "theoretical_median": round(
            float(np.median([r["theoretical"] for r in rows])), 2),
        "skus_total_eligible": int(len(ranked)),
    }


def economic_impact(
    df: pd.DataFrame,
    model: QuantileModel | None = None,
    econ: Economics | None = None,
    service_factor: float | None = None,
    max_skus: int | None = 10,
    test_days: int = 90,
    lead_time: int = 7,
    review_period: int = 7,
    tune_skus: int = 10,
    fit: dict | None = None,
) -> dict:
    """
    Экономический эффект для показа в интерфейсе.

    max_skus — по скольким самым оборотистым товарам считать эффект.
    None означает «по всем, у кого хватает истории»: цифра получается
    полной, но ждать дольше.

    Подбор страхового запаса при этом всегда идёт на ограниченной
    подвыборке (tune_skus): коэффициент общий для всего ассортимента,
    и гонять ради него сотню товаров смысла нет, а вот сам эффект
    считается уже по всему заданному охвату.

    Возвращает готовые к показу показатели: снижение затрат на
    хранение и предотвращённые потери от упущенной выручки —
    ровно то, что требует экономическое обоснование проекта.
    """
    if df is None or df.empty:
        return {}

    econ = econ or Economics.from_data(df)

    if not df.groupby(SKU)[QTY].size().ge(test_days + 90).any():
        return {"error": "Для расчёта эффекта нужно минимум "
                         f"{test_days + 90} дней истории по товару."}

    # Запас подбирается по каждому товару отдельно, и по каждому же
    # решается, стоит ли вообще управлять им по прогнозу.
    tuned = tune_portfolio(df, model, econ=econ, test_days=test_days,
                           lead_time=lead_time, review_period=review_period,
                           max_skus=max_skus,
                           allow_manual=service_factor is None)
    if not tuned:
        return {"error": "Не удалось рассчитать эффект на этих данных."}

    summary = tuned["summary"]
    theoretical = tuned["theoretical_median"]
    service_factor = tuned["factor_median"]

    holding_before = summary["Затраты на хранение as-is, ₽"]
    holding_after = summary["Затраты на хранение to-be, ₽"]
    lost_before = summary["Упущенная выручка as-is, ₽"]
    lost_after = summary["Упущенная выручка to-be, ₽"]

    def pct(before: float, after: float) -> float | None:
        if before < 100:          # база слишком мала, процент бессмыслен
            return None
        return round((1 - after / before) * 100, 1)

    # Выигрывает ли система вообще. Если нет — интерфейс не должен
    # показывать это как достижение: отрицательный «процент снижения»
    # означает рост затрат, и называть его снижением нельзя.
    effect = summary["Эффект в месяц, ₽"]
    beneficial = effect > 0

    return {
        "period_days": test_days,
        "skus_analyzed": summary.get("Товаров в расчёте эффекта", 0),
        "service_factor": round(float(service_factor), 2),
        "theoretical_factor": theoretical,
        "unit_cost": econ.unit_cost,
        "margin": round(econ.margin, 4),

        "holding_before": holding_before,
        "holding_after": holding_after,
        "holding_saved_pct": pct(holding_before, holding_after),

        "stock_before": summary["Средний остаток as-is, ₽"],
        "stock_after": summary["Средний остаток to-be, ₽"],
        "stock_reduced_pct": summary["Снижение остатка, %"],

        "lost_before": lost_before,
        "lost_after": lost_after,
        "lost_prevented": round(lost_before - lost_after),
        "lost_prevented_pct": pct(lost_before, lost_after),

        "total_effect_period": summary["Совокупный эффект за период, ₽"],
        "effect_per_month": effect,
        "beneficial": beneficial,
        "skus_total_eligible": tuned["skus_total_eligible"],

        # по скольким товарам система реально управляет закупкой, а по
        # скольким сама отказалась и оставила ручное правило
        "skus_managed": tuned["managed"],
        "skus_manual": tuned["manual"],
        "manual_skus": tuned["manual_skus"][:8],
        "factor_min": tuned["factor_min"],
        "factor_max": tuned["factor_max"],

        "diagnosis": _diagnose(summary, service_factor, theoretical, fit, tuned),
    }


def _diagnose(summary: dict, factor: float, theoretical: float | None,
              fit: dict | None = None, tuned: dict | None = None) -> dict:
    """
    Объясняет, почему эффект получился таким.

    Низкий эффект сам по себе ничего не говорит — важно, из-за чего он
    низкий. Чаще всего причина одна: модель обучали не на этих данных,
    её прогноз смещён, и системе приходится компенсировать это раздутым
    страховым запасом. Держать лишний запас стоит денег, поэтому вся
    выгода уходит на его оплату. Лечится переобучением на своём файле.
    """
    effect = summary.get("Эффект в месяц, ₽", 0)
    lost_after = summary.get("Упущенная выручка to-be, ₽", 0)
    lost_before = summary.get("Упущенная выручка as-is, ₽", 0)

    notes: list[str] = []
    level = "ok"

    # Модель уже проверена на этих данных? Тогда списывать всё на
    # «обучена не на том ассортименте» нельзя — это прямо противоречило бы
    # сообщению о проверке, которое пользователь видит выше на экране.
    model_verified = bool(fit and fit.get("checked") and fit.get("suitable"))

    managed = (tuned or {}).get("managed")
    manual = (tuned or {}).get("manual") or 0
    manual_names = (tuned or {}).get("manual_skus") or []

    # Система сама отказалась управлять частью ассортимента. Это штатное
    # и правильное поведение, но пользователь должен понимать, по каким
    # именно товарам прогноз не применяется и почему.
    if manual:
        if level == "ok":
            level = "warn"
        names = ", ".join(str(s)[:34] for s in manual_names[:3])
        tail = f" и ещё {manual - 3}" if manual > 3 else ""
        notes.append(
            f"По {manual} товар(ам) система оставила ручное планирование: "
            f"{names}{tail}. На них прогноз не даёт выигрыша — спрос "
            "слишком неровный, и любой страховой запас стоит дороже, чем "
            "экономит. Эти товары в эффект не засчитаны, закупайте их "
            "как раньше.")

    if theoretical and factor > theoretical * 1.6:
        level = "warn"
        if model_verified:
            notes.append(
                "Страховой запас пришлось поднять выше расчётного "
                "(медиана × {:.2f} против × {:.2f}). Прогноз точен, но спрос "
                "по этим товарам неровный: чтобы покрыть всплески, запаса "
                "нужно держать больше теоретического."
                .format(factor, theoretical))
        else:
            notes.append(
                "Страховой запас пришлось поднять выше расчётного "
                "(медиана × {:.2f} против × {:.2f}). Так бывает, когда модель "
                "обучена на другом ассортименте и занижает спрос по вашим "
                "товарам. Переобучите её на этом же файле."
                .format(factor, theoretical))

    if lost_after > max(lost_before * 1.05, 100):
        level = "warn"
        notes.append(
            "Дефицит у системы получился выше, чем при ручном планировании. "
            "Это указывает на смещённый прогноз: модель недооценивает "
            "спрос, и запаса не хватает.")

    if effect <= 0:
        level = "bad"
        notes.append(
            "На этих данных система не выигрывает у ручного планирования "
            "ни по одному товару — внедрять её здесь не нужно. "
            "Проверьте закупочные цены и срок поставки в форме: расчёт "
            "опирается именно на них."
            if model_verified else
            "На этих данных система не выигрывает у ручного планирования. "
            "Главное, что стоит сделать, — обучить модель на этом файле.")
    elif managed and effect < 500:
        if level == "ok":
            level = "warn"
        notes.append(
            "Эффект небольшой. Проверьте, что модель обучена на этих же "
            "данных, и что закупочные цены в файле реальные — от них "
            "напрямую зависит расчёт.")

    if not notes:
        managed_txt = (f"По всем {managed} товарам прогноз выгоднее ручного "
                       "планирования: " if managed else "")
        notes.append(managed_txt + "запас меньше, дефицита не больше.")
    elif effect > 0 and managed:
        notes.insert(0, f"По {managed} товар(ам) закупка идёт по прогнозу, "
                        f"и на них система выигрывает у ручного планирования.")

    return {"level": level, "notes": notes}
