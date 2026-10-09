"""
План закупки со всеми настройками менеджера в одном месте.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ
----------------------
Раньше сервер сам собирал план: брал прогноз, экономику, вызывал расчёт
заказа. С появлением сроков поставки по каждому товару, ручных правок и
«уже в пути» этих шагов стало слишком много, чтобы держать их в обработчике
запроса. Здесь они собраны так, чтобы и экран, и скачиваемая заявка, и
рекомендации считались из одного места и не могли разойтись.

ГЛАВНОЕ УСКОРЕНИЕ
-----------------
Прогноз зависит только от горизонта, а срок поставки, период заказов,
ручные правки и «в пути» на него не влияют — они меняют лишь арифметику
вокруг уже готового прогноза. Поэтому прогноз считается один раз на
горизонт и кэшируется, а смена срока поставки пересчитывает только
дешёвую часть и занимает доли секунды. Раньше каждое изменение срока
запускало прогноз заново вместе с дообучением модели — отсюда и
ощущение, что сайт «завис».
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .economics import derive_sku_economics
from .inventory import build_order_plan
from .schema import (
    DATE, DEFAULT_HORIZONS, DEFAULT_LEAD_DAYS, DEFAULT_REVIEW_DAYS, LEAD,
    MAX_LEAD_DAYS, MIN_LEAD_DAYS, SKU,
)

SOURCE_FILE = "file"        # срок взят из колонки в файле
SOURCE_MANUAL = "manual"    # менеджер поменял вручную
SOURCE_DEFAULT = "default"  # общий срок из настроек

SOURCE_LABELS_RU = {
    SOURCE_FILE: "из файла",
    SOURCE_MANUAL: "задан вручную",
    SOURCE_DEFAULT: "общий срок",
}


def clamp_days(value, lo: int = MIN_LEAD_DAYS, hi: int = MAX_LEAD_DAYS) -> int:
    return int(min(max(round(float(value)), lo), hi))


def as_of_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    «Сегодня» для расчёта дат — день после последней даты в файле.

    Прогноз строится именно от этого дня, поэтому и даты заказа должны
    лежать на той же оси. Для свежей выгрузки это настоящая сегодняшняя
    дата; для старого файла — «сегодня» из его времени, а не из нашего,
    иначе все даты оказались бы в прошлом.
    """
    return pd.Timestamp(df[DATE].max()).normalize() + pd.Timedelta(days=1)


def file_lead_times(df: pd.DataFrame) -> dict[str, int]:
    """Сроки поставки, найденные в файле: по товару берётся самое частое значение."""
    if LEAD not in df.columns or not df[LEAD].notna().any():
        return {}
    out: dict[str, int] = {}
    for sku, g in df.groupby(SKU, sort=False):
        vals = g[LEAD].dropna()
        vals = vals[vals > 0]
        if vals.empty:
            continue
        # самое частое значение, а не среднее: срок поставки — это число,
        # которое не усредняют, и один сбой в учёте не должен его сдвигать
        out[str(sku)] = clamp_days(vals.round().mode().iloc[0])
    return out


def resolve_lead_times(df: pd.DataFrame, default_lead: int,
                       overrides: dict | None = None
                       ) -> tuple[dict[str, int], dict[str, str]]:
    """
    Срок поставки каждого товара и откуда он взят.

    Приоритет: правка менеджера → значение из файла → общий срок.
    """
    from_file = file_lead_times(df)
    overrides = overrides or {}
    default_lead = clamp_days(default_lead)

    used: dict[str, int] = {}
    source: dict[str, str] = {}
    for sku in df[SKU].unique():
        key = str(sku)
        ov = overrides.get(key)
        if ov is not None and np.isfinite(float(ov)):
            used[key], source[key] = clamp_days(ov), SOURCE_MANUAL
        elif key in from_file:
            used[key], source[key] = from_file[key], SOURCE_FILE
        else:
            used[key], source[key] = default_lead, SOURCE_DEFAULT
    return used, source


def effective_horizon(horizon: int, lead_times: dict[str, int],
                      review_period: int) -> tuple[int, str | None]:
    """
    Горизонт прогноза не может быть короче срока покрытия самого
    долгого товара: прогноз «на 14 дней» нельзя честно растянуть на
    35. Если такое случилось, берём ближайший большой стандартный горизонт.
    """
    need = (max(lead_times.values()) if lead_times else 0) + review_period
    if need <= horizon:
        return horizon, None
    for h in sorted(DEFAULT_HORIZONS):
        if h >= need:
            return h, (f"Горизонт прогноза увеличен до {h} дн.: срок покрытия "
                       f"самых долгих товаров — {need} дн. "
                       "(поставка плюс период между заказами).")
    top = max(DEFAULT_HORIZONS)
    return top, (f"Срок покрытия самых долгих товаров ({need} дн.) больше "
                 f"максимального горизонта прогноза ({top} дн.) — по ним "
                 "расчёт приблизительный.")


# Что делать с товаром — одно слово для таблицы и цвет для экрана
ACT_NOW, ACT_SOON, ACT_LATER, ACT_NONE, ACT_EXCESS, ACT_UNKNOWN = (
    "now", "soon", "later", "none", "excess", "unknown")
SOON_DAYS = 3

STATUS_PLAIN_RU = {
    "critical": "Срочно", "alert": "Пора заказать",
    "ok": "В порядке", "excess": "Слишком много",
}


def _fmt_ru(d) -> str:
    return pd.Timestamp(d).strftime("%d.%m.%Y") if d else ""


def _short_ru(d) -> str:
    months = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
              "августа", "сентября", "октября", "ноября", "декабря"]
    t = pd.Timestamp(d)
    return f"{t.day} {months[t.month - 1]}"


def add_actions(plan: pd.DataFrame) -> pd.DataFrame:
    """Одна понятная фраза на товар: что делать и когда."""
    kinds, texts = [], []
    for r in plan.itertuples():
        if not r.stock_known:
            k, t = ACT_UNKNOWN, "Укажите остаток"
        elif r.status == "excess":
            k, t = ACT_EXCESS, "Не заказывать — запаса много"
        elif r.recommended_order <= 0:
            k, t = ACT_NONE, "Заказ не нужен"
        elif r.order_in_days is not None and r.order_in_days <= 0:
            k, t = ACT_NOW, "Заказать сегодня"
        elif r.order_in_days is not None and r.order_in_days <= SOON_DAYS:
            k, t = ACT_SOON, f"Заказать до {_short_ru(r.order_by_date)}"
        else:
            k, t = ACT_LATER, f"Заказать до {_short_ru(r.order_by_date)}"
        kinds.append(k)
        texts.append(t)
    plan = plan.copy()
    plan["action_kind"], plan["action"] = kinds, texts
    plan["status_plain"] = plan["status"].map(STATUS_PLAIN_RU).fillna("")
    return plan


def summarize_plan(plan: pd.DataFrame) -> dict:
    """Цифры для верхней панели экрана «Что заказать»."""
    if plan.empty:
        return {"total": 0}
    kinds = plan["action_kind"].value_counts().to_dict()
    to_order = plan[plan["recommended_order"] > 0]
    now = plan[plan["action_kind"] == ACT_NOW]
    return {
        "total": int(len(plan)),
        "now": int(kinds.get(ACT_NOW, 0)),
        "soon": int(kinds.get(ACT_SOON, 0)),
        "later": int(kinds.get(ACT_LATER, 0)),
        "none": int(kinds.get(ACT_NONE, 0)),
        "excess": int(kinds.get(ACT_EXCESS, 0)),
        "unknown": int(kinds.get(ACT_UNKNOWN, 0)),
        "order_value_now": float(now["order_value"].fillna(0).sum()),
        "order_value_total": float(to_order["order_value"].fillna(0).sum()),
        "order_lines_total": int(len(to_order)),
        "shortage_skus": int((plan["stockout_gap_days"] > 0).sum()),
        "lost_units": float(plan["expected_lost_units"].sum()),
    }


def make_plan(df: pd.DataFrame, fc: pd.DataFrame, horizon: int,
              default_lead: int = DEFAULT_LEAD_DAYS,
              review_period: int = DEFAULT_REVIEW_DAYS,
              lead_overrides: dict | None = None,
              in_transit: dict | None = None,
              stock_overrides: dict | None = None) -> dict:
    """
    Собирает план закупки из готового прогноза.

    fc — результат forecast() на этом горизонте; он не пересчитывается.
    stock_overrides — остатки, которые менеджер ввёл руками (например,
    когда в файле остатков нет или они устарели).
    """
    review_period = clamp_days(review_period, 1, MAX_LEAD_DAYS)
    leads, sources = resolve_lead_times(df, default_lead, lead_overrides)
    as_of = as_of_date(df)

    # страховой запас каждого товара — из его собственной экономики
    # и его собственного срока покрытия
    covers = {k: v + review_period for k, v in leads.items()}
    eco = derive_sku_economics(
        df, cover_days=clamp_days(default_lead) + review_period,
        cover_days_by_sku=covers)
    factors, costs, prices, eco_info = None, {}, {}, {}
    if not eco.empty:
        factors = dict(zip(eco["sku"], eco["service_factor"]))
        costs = dict(zip(eco["sku"], eco["unit_cost"]))
        prices = dict(zip(eco["sku"], eco["unit_price"]))
        from_data = int((eco["source"] == "данные").sum())
        eco_info = {"prices_from_file": from_data,
                    "prices_estimated": int(len(eco) - from_data),
                    "avg_service_level": round(float(eco["service_level"].mean()), 3)}

    stocks = None
    if stock_overrides:
        stocks = {str(r.sku): (float(r.stock) if np.isfinite(r.stock) else None)
                  for r in fc.itertuples()}
        for k, v in stock_overrides.items():
            if v is not None and np.isfinite(float(v)):
                stocks[str(k)] = max(float(v), 0.0)

    plan = build_order_plan(
        fc, stocks=stocks, horizon_days=horizon, service_factors=factors,
        lead_times=leads, review_periods=None, in_transit=in_transit,
        as_of=as_of, lead_time_days=clamp_days(default_lead),
        review_period_days=review_period)

    if not plan.empty:
        plan["lead_source"] = plan["sku"].map(sources).fillna(SOURCE_DEFAULT)
        plan["unit_cost"] = plan["sku"].map(costs)
        plan["unit_price"] = plan["sku"].map(prices)
        plan["order_value"] = (plan["recommended_order"]
                               * plan["unit_cost"]).round(0)
        plan = add_actions(plan)

    return {"plan": plan, "summary": summarize_plan(plan), "costs": costs, "prices": prices, "eco_info": eco_info,
            "lead_times": leads, "lead_sources": sources, "as_of": as_of,
            "review_period": review_period}
