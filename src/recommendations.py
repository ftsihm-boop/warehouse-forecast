"""
Рекомендации менеджеру: «что сделать», а не «что посчитано».

Собирает выводы из четырёх источников — плана закупки, каталога аномалий,
оценки точности и показателей по товарам — в один список понятных действий,
отсортированный по срочности. Где возможно, рядом стоит сумма в рублях:
менеджер реагирует на «вы потеряете около 40 000 ₽», а не на «дефицит 92 шт».

Группы:
    urgent    — сделать сегодня, иначе товар кончится до прихода поставки
    soon      — заказать в ближайшие дни
    optimize  — сэкономить: избытки, замороженные деньги, тренды
    data      — поправить данные, чтобы рекомендации стали точнее
    info      — к сведению
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = [
    ("urgent", "Сделайте сегодня"),
    ("soon", "В ближайшие дни"),
    ("optimize", "Как сэкономить"),
    ("data", "Улучшите данные"),
    ("info", "К сведению"),
]
SOON_DAYS = 3
MAX_NAMES = 6


def _names(skus: list[str], limit: int = MAX_NAMES) -> str:
    s = [f"«{x}»" for x in skus[:limit]]
    more = len(skus) - limit
    return ", ".join(s) + (f" и ещё {more}" if more > 0 else "")


_MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
           "августа", "сентября", "октября", "ноября", "декабря"]


def _ru_date(v) -> str:
    t = pd.Timestamp(v)
    return f"{t.day} {_MONTHS[t.month - 1]}"


def _plural(n: int, one: str, few: str, many: str) -> str:
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return one
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return few
    return many


def rub(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") + " ₽"


def _rec(group: str, title: str, text: str, skus: list[str] | None = None,
         money: float | None = None, money_label: str | None = None,
         tab: str | None = None, key: str = "") -> dict:
    return {"group": group, "key": key or title, "title": title, "text": text,
            "skus": skus or [], "money": None if money is None else round(float(money)),
            "money_label": money_label, "tab": tab}


def build_recommendations(plan: pd.DataFrame, anomalies: dict | None = None,
                          accuracy: dict | None = None,
                          features: list[dict] | None = None,
                          prices: dict | None = None, costs: dict | None = None,
                          lead_from_file: bool = False) -> dict:
    recs: list[dict] = []
    prices, costs = prices or {}, costs or {}

    def margin(sku: str) -> float:
        p, c = prices.get(sku), costs.get(sku)
        return max(float(p) - float(c), 0.0) if p is not None and c is not None \
            and np.isfinite(p) and np.isfinite(c) else 0.0

    if plan is not None and not plan.empty:
        known = plan[plan["stock_known"].astype(bool)] if "stock_known" in plan else plan

        # --- срочно: не хватит до прихода поставки ---------------------------
        gap = known[known["stockout_gap_days"] > 0].sort_values(
            "expected_lost_units", ascending=False)
        if not gap.empty:
            lost = float(sum(r.expected_lost_units * margin(r.sku)
                             for r in gap.itertuples()))
            worst = gap.iloc[0]
            n = len(gap)
            recs.append(_rec(
                "urgent",
                f"{n} {_plural(n, 'товар кончится', 'товара кончатся', 'товаров кончатся')} "
                "раньше, чем придёт поставка",
                f"Закажите сегодня: {_names(gap['sku'].tolist())}. Даже если заказать "
                f"сегодня, до прихода товара пройдёт время без остатка — например, "
                f"«{worst['sku']}» будет отсутствовать около {int(worst['stockout_gap_days'])} дн. "
                "Ускорить поставку или заказать у поставщика с коротким сроком — "
                "единственный способ сократить потери.",
                gap["sku"].tolist(), lost if lost > 0 else None,
                "упущенная прибыль, если ничего не делать" if lost > 0 else None,
                "plan", "gap"))

        # --- срочно/скоро: пора заказывать ------------------------------------
        due = known[(known["stockout_gap_days"] <= 0) & (known["recommended_order"] > 0)
                    & (known["order_in_days"].fillna(99) <= 0)]
        if not due.empty:
            total = float(due["order_value"].fillna(0).sum()) if "order_value" in due else None
            recs.append(_rec(
                "urgent", f"Пора заказывать: {len(due)} "
                          f"{_plural(len(due), 'товар', 'товара', 'товаров')}",
                f"Остатка хватит ровно до прихода поставки — без запаса. Если отложить "
                f"заказ, появится дефицит. {_names(due['sku'].tolist())}.",
                due["sku"].tolist(), total if total else None, "сумма заказа", "plan", "due"))

        soon = known[(known["recommended_order"] > 0)
                     & (known["order_in_days"].fillna(99) > 0)
                     & (known["order_in_days"].fillna(99) <= SOON_DAYS)]
        if not soon.empty:
            soon = soon.sort_values("order_in_days")
            recs.append(_rec(
                "soon", f"В ближайшие {SOON_DAYS} дня нужно заказать "
                        f"{len(soon)} {_plural(len(soon), 'товар', 'товара', 'товаров')}",
                "Можно собрать всё в одну заявку. Ближайший: «"
                f"{soon.iloc[0]['sku']}» — до {_ru_date(soon.iloc[0]['order_by_date'])}.",
                soon["sku"].tolist(), None, None, "plan", "soon"))

        # --- избытки ----------------------------------------------------------
        exc = known[known["status"] == "excess"]
        if not exc.empty:
            frozen = float(sum(max(r.stock - r.forecast_horizon_q50, 0)
                               * (costs.get(r.sku) or 0) for r in exc.itertuples()))
            recs.append(_rec(
                "optimize", f"Лишний запас: {len(exc)} "
                            f"{_plural(len(exc), 'товар', 'товара', 'товаров')}",
                f"Остатка хватит надолго — {_names(exc['sku'].tolist())}. "
                "Не заказывайте их, пока запас не уменьшится; подумайте об акции, "
                "чтобы освободить деньги.",
                exc["sku"].tolist(), frozen if frozen > 0 else None,
                "деньги, замороженные в излишке" if frozen > 0 else None, "plan", "excess"))

        # --- нет остатка в файле ---------------------------------------------
        if "stock_known" in plan:
            unk = plan[~plan["stock_known"].astype(bool)]
            if not unk.empty:
                recs.append(_rec(
                    "data", f"Не знаем остаток по {len(unk)} "
                            f"{_plural(len(unk), 'товару', 'товарам', 'товарам')}",
                    "В файле нет остатков, поэтому нельзя сказать, на сколько дней "
                    "хватит товара. Введите остаток вручную в таблице «Что заказать».",
                    unk["sku"].tolist(), None, None, "plan", "nostock"))

        # --- сроки поставки по умолчанию -------------------------------------
        if "lead_source" in plan and not lead_from_file:
            dflt = plan[plan["lead_source"] == "default"]
            if len(dflt) == len(plan) and len(plan) > 1:
                recs.append(_rec(
                    "data", "Укажите реальные сроки поставки",
                    "Сейчас у всех товаров один и тот же срок. У разных поставщиков "
                    "он разный — введите срок для каждого товара в таблице или "
                    "добавьте колонку «Срок поставки, дн.» в файл. Тогда дата "
                    "заказа будет точной для каждого товара.", [], None, None,
                    "plan", "defaultlead"))

    # --- аномалии в данных ---------------------------------------------------
    if anomalies:
        for g in anomalies.get("groups", []):
            if g["severity"] == "attention":
                recs.append(_rec(
                    "data", g["title"] + f" ({g['count']})",
                    f"{g['what']} {g['action']}", g["skus"],
                    None, None, "data", "anom_" + g["code"]))

    # --- точность -----------------------------------------------------------
    if accuracy:
        low = [i["sku"] for i in accuracy.get("items", []) if i["level"] == "low"]
        if low:
            recs.append(_rec(
                "info", f"По {len(low)} "
                        f"{_plural(len(low), 'товару', 'товарам', 'товарам')} "
                        "прогноз неточный",
                f"{_names(low)}. Прогноз по ним часто промахивается — прежде чем "
                "размещать крупный заказ, сверьтесь со своим опытом.",
                low, None, None, "data", "lowacc"))
        none_ = [i["sku"] for i in accuracy.get("items", []) if i["level"] == "none"]
        if none_:
            recs.append(_rec(
                "info", f"{len(none_)} "
                        f"{_plural(len(none_), 'товар', 'товара', 'товаров')} "
                        "нельзя прогнозировать",
                f"{_names(none_)}. Продажи слишком нерегулярные (редкие крупные "
                "партии). Для них используется простое правило «минимум–максимум», "
                "а заказывать лучше под конкретный спрос.", none_, None, None,
                "data", "noforecast"))

    # --- тренды по товарам --------------------------------------------------
    if features:
        # тренд осмыслен только у товаров с регулярным спросом: у редких
        # он скачет от одной крупной продажи и вводит в заблуждение
        reg = [f for f in features if f.get("demand_class") in ("smooth", "erratic")]
        up = [f for f in reg if (f.get("trend_28") or 0) >= 0.3]
        dn = [f for f in reg if (f.get("trend_28") or 0) <= -0.3]
        if up:
            recs.append(_rec(
                "optimize", f"Спрос растёт: {len(up)} "
                            f"{_plural(len(up), 'товар', 'товара', 'товаров')}",
                f"{_names([f['sku'] for f in up])} — продажи за последний месяц выросли "
                "на треть и больше. Проверьте, не пора ли увеличить закупку и сроки.",
                [f["sku"] for f in up], None, None, "data", "trend_up"))
        if dn:
            recs.append(_rec(
                "optimize", f"Спрос падает: {len(dn)} "
                            f"{_plural(len(dn), 'товар', 'товара', 'товаров')}",
                f"{_names([f['sku'] for f in dn])} — продажи за последний месяц упали "
                "на треть и больше. Не закупайте «по привычке».",
                [f["sku"] for f in dn], None, None, "data", "trend_down"))
        dd = [f for f in features if (f.get("stockout_share") or 0) >= 0.1]
        if dd:
            recs.append(_rec(
                "optimize", f"Часто нет в наличии: {len(dd)} "
                            f"{_plural(len(dd), 'товар', 'товара', 'товаров')}",
                f"{_names([f['sku'] for f in dd])} — товара не было на складе больше "
                "10 % дней. Это упущенные продажи: увеличьте страховой запас или "
                "сократите срок поставки.", [f["sku"] for f in dd], None, None,
                "data", "stockouts"))

    order = {g: i for i, (g, _t) in enumerate(GROUPS)}
    recs.sort(key=lambda r: (order[r["group"]], -(r["money"] or 0)))

    n_urgent = sum(1 for r in recs if r["group"] == "urgent")
    if n_urgent:
        headline = (f"Нужно действовать сегодня: {n_urgent} "
                    f"{_plural(n_urgent, 'срочная задача', 'срочные задачи', 'срочных задач')}.")
    elif any(r["group"] == "soon" for r in recs):
        headline = "Срочного нет, но в ближайшие дни нужно сделать заказ."
    elif recs:
        headline = "Срочных дел нет — есть рекомендации по улучшению."
    else:
        headline = "Всё в порядке — рекомендаций нет."
    return {"headline": headline, "groups": [{"code": c, "title": t} for c, t in GROUPS],
            "items": recs}
