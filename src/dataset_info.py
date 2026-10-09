"""
Описание набора данных человеческим языком: что в файле, какие колонки
распознаны, насколько они заполнены и что по каждому товару.

Это то, что менеджер видит на вкладке «Данные» первым: прежде чем
доверять рекомендациям, нужно убедиться, что система правильно поняла файл.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import CleaningReport
from .demand_classes import CLASS_LABELS_RU, profile_all
from .schema import COST, DATE, LEAD, PRICE, QTY, SKU, STOCK

COLUMN_INFO = {
    DATE: ("Дата", "День продажи. Из неё система узнаёт про дни недели, "
                   "сезон и праздники.", True),
    SKU: ("Товар", "Название или артикул. Один товар — один прогноз.", True),
    QTY: ("Продано, шт", "Сколько единиц продано за день. Главное, что мы "
                         "предсказываем.", True),
    STOCK: ("Остаток, шт", "Сколько было на складе на конец дня. Нужен, чтобы "
                           "отличить «не купили» от «нечего было купить» и "
                           "чтобы посчитать, на сколько дней хватит.", False),
    PRICE: ("Цена продажи", "Нужна, чтобы посчитать выручку и цену нехватки "
                            "товара.", False),
    COST: ("Закупочная цена", "Нужна, чтобы посчитать стоимость заказа и "
                              "цену излишка на складе.", False),
    LEAD: ("Срок поставки, дн.", "За сколько дней привозят товар. Нужен, чтобы "
                                 "назвать дату, когда пора заказывать.", False),
}

MISSING_HINT = {
    STOCK: "Без остатка нельзя сказать, на сколько дней хватит товара; "
           "укажите его вручную в плане заказа.",
    PRICE: "Без цен деньги в расчёте — примерные (по типичной наценке).",
    COST: "Без закупочной цены стоимость заказа примерная.",
    LEAD: "Без срока поставки используется общий срок из настроек.",
}


def _example(series: pd.Series, col: str) -> str:
    s = series.dropna()
    if s.empty:
        return "—"
    if col == DATE:
        return f"{s.min():%d.%m.%Y} – {s.max():%d.%m.%Y}"
    if col == SKU:
        return ", ".join(map(str, s.astype(str).unique()[:2])) + "…"
    return f"{s.min():g} – {s.max():g}"


def describe_columns(df: pd.DataFrame) -> list[dict]:
    out = []
    for col, (title, desc, required) in COLUMN_INFO.items():
        present = col in df.columns and df[col].notna().any()
        fill = float(df[col].notna().mean()) if col in df.columns else 0.0
        item = {"column": col, "title": title, "description": desc,
                "required": required, "present": bool(present),
                "fill_pct": round(fill * 100), "range": _example(df[col], col)
                if col in df.columns else "—"}
        if not present:
            item["hint"] = MISSING_HINT.get(col, "")
        out.append(item)
    return out


def describe_dataset(df: pd.DataFrame, rep: CleaningReport) -> dict:
    n_days = int((df[DATE].max() - df[DATE].min()).days) + 1
    overview = {
        "rows_read": int(rep.rows_read),
        "rows_final": int(len(df)),
        "sku_count": int(df[SKU].nunique()),
        "date_min": f"{df[DATE].min():%d.%m.%Y}",
        "date_max": f"{df[DATE].max():%d.%m.%Y}",
        "days": n_days,
        "total_qty": float(df[QTY].sum()),
    }
    if PRICE in df.columns and df[PRICE].notna().any():
        overview["revenue"] = float((df[QTY] * df[PRICE].fillna(0)).sum())

    prof = profile_all(df)
    last_stock = {}
    if STOCK in df.columns:
        last_stock = (df.sort_values(DATE).groupby(SKU)[STOCK].last()).to_dict()
    span = df.groupby(SKU)[DATE].agg(["min", "max"])

    skus = []
    for _, r in prof.iterrows():
        sku = r["sku"]
        st = last_stock.get(sku, np.nan)
        skus.append({
            "sku": sku, "from": f"{span.at[sku, 'min']:%d.%m.%Y}",
            "to": f"{span.at[sku, 'max']:%d.%m.%Y}", "days": int(r["days"]),
            "total": float(r["total_qty"]), "per_day": float(r["mean_daily"]),
            "stock": None if not np.isfinite(st) else float(st),
            "abc": r["abc_class"], "demand_class": r["demand_class"],
            "demand_label": CLASS_LABELS_RU.get(r["demand_class"], ""),
        })

    # короткая оценка пригодности: достаточно ли истории для сезонности
    if n_days >= 730:
        depth = ("Истории хватает на сезонность: больше двух лет, "
                 "модель видит каждый сезон дважды.")
    elif n_days >= 365:
        depth = "Истории хватает на годовую сезонность, но лучше иметь два года."
    elif n_days >= 90:
        depth = ("Истории меньше года: сезонность и праздники модель выучить "
                 "не сможет, прогноз опирается на недавние продажи.")
    else:
        depth = "Истории очень мало — прогноз будет грубым."

    return {"overview": overview, "columns": describe_columns(df),
            "history_note": depth, "skus": skus, "notes": list(rep.notes)}
