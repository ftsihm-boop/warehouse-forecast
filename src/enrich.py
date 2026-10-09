"""
Признаки: что модель видит про каждый товар и какие дополнительные
показатели мы считаем поверх исходных колонок файла.

ДВА УРОВНЯ
----------
1. ПРИЗНАКИ МОДЕЛИ (каталог) — то, по чему обучается прогноз: вчерашние
   и недельные продажи, средние, разброс, тренд, календарь, праздники.
   Здесь только описание на человеческом языке; сами признаки строит
   features.py. Они помечены «используется моделью».

2. БИЗНЕС-ПОКАЗАТЕЛИ ПО ТОВАРУ — производные от базы показатели для
   менеджера: ABC/XYZ, тренд, сезонность по дням недели, доля дней
   дефицита, оборачиваемость, маржа, изменение цены. Модель на них не
   обучается — они объясняют ассортимент и подсвечивают проблемы. Помечены
   «для анализа»: честно разделять одно и другое важно, чтобы не
   приписывать модели то, чего она не использует.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_ru import CALENDAR_FEATURES
from .demand_classes import CLASS_LABELS_RU, profile_all
from .features import FEATURE_COLUMNS, LAGS, WINDOWS
from .schema import COST, DATE, FLAG_CENSORED, PRICE, QTY, SKU, STOCK

WEEKDAYS_RU = ["понедельник", "вторник", "среда", "четверг", "пятница",
               "суббота", "воскресенье"]
WEEKDAYS_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]

# --- каталог признаков модели --------------------------------------------------

_MODEL_GROUPS = [
    ("lags", "Недавние продажи",
     "Сколько продали вчера, позавчера, неделю, две, четыре недели назад. "
     "Главный сигнал: то, что было недавно, скорее всего повторится.",
     [f"lag_{l}" for l in LAGS]),
    ("means", "Средние продажи",
     "Средние и медианные продажи за 1–8 недель. Сглаживают случайные "
     "скачки и показывают «нормальный» уровень.",
     [f"{s}_{w}" for w in WINDOWS for s in ("mean", "median")]),
    ("spread", "Разброс продаж",
     "Насколько продажи скачут и каков максимум за период. Чем сильнее "
     "разброс, тем больше страховой запас.",
     [f"{s}_{w}" for w in WINDOWS for s in ("std", "max")] + ["cv_28"]),
    ("trend", "Тренд",
     "Растут или падают продажи: сравнение последней недели/двух недель "
     "с месяцем/двумя.", ["trend_7_28", "trend_14_56"]),
    ("profile", "Характер спроса",
     "Как часто бывают дни без продаж, сколько дней дефицита, сколько "
     "истории накоплено, как товар ведёт себя в этот день недели.",
     ["zero_share_28", "censored_share_28", "days_of_history", "dow_ratio_8w"]),
    ("calendar", "Календарь",
     "День недели, число месяца, месяц, неделя года, выходные, начало и "
     "конец месяца.",
     [c for c in CALENDAR_FEATURES if not c.endswith("holiday")
      and "holiday" not in c]),
    ("holidays", "Праздники",
     "Праздничный ли день, сколько дней до ближайшего праздника и после "
     "него (по российскому календарю).",
     [c for c in CALENDAR_FEATURES if "holiday" in c]),
    ("horizon", "Горизонт", "На сколько дней вперёд считаем прогноз.", ["horizon"]),
]


def model_feature_catalog() -> list[dict]:
    known = set(FEATURE_COLUMNS)
    out = []
    for code, title, desc, cols in _MODEL_GROUPS:
        cols = [c for c in cols if c in known]
        out.append({"code": code, "title": title, "description": desc,
                    "count": len(cols), "features": cols,
                    "used_by_model": True})
    return out


# --- бизнес-показатели по товару -----------------------------------------------

def _xyz(cv: float) -> str:
    if not np.isfinite(cv):
        return "—"
    return "X" if cv < 0.5 else "Y" if cv < 1.0 else "Z"


XYZ_RU = {"X": "стабильный спрос", "Y": "заметные колебания",
          "Z": "спрос трудно предсказать", "—": "мало данных"}


def enrich_by_sku(df: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на товар: расширенный профиль для менеджера."""
    prof = profile_all(df).set_index("sku")
    last_day = df[DATE].max()
    has_stock = STOCK in df.columns and df[STOCK].notna().any()
    has_price = PRICE in df.columns and df[PRICE].notna().any()
    has_cost = COST in df.columns and df[COST].notna().any()

    # ABC по выручке, если есть цены, иначе по штукам (profile_all уже по штукам)
    abc_money = None
    if has_price:
        rev = (df[QTY] * df[PRICE].fillna(0)).groupby(df[SKU]).sum()
        from .demand_classes import _abc_classes
        abc_money = _abc_classes(rev)

    rows = []
    for sku, g in df.groupby(SKU, sort=False):
        g = g.sort_values(DATE)
        q = g[QTY].astype(float)
        if FLAG_CENSORED in g.columns:
            cens = g[FLAG_CENSORED].astype(bool)
            q_ok = q[~cens]
        else:
            cens, q_ok = pd.Series(False, index=g.index), q

        weekly = q.groupby(g[DATE].dt.to_period("W")).sum()
        weekly = weekly.iloc[1:-1] if len(weekly) > 4 else weekly
        cv = float(weekly.std() / weekly.mean()) if len(weekly) >= 4 and weekly.mean() > 0 else np.nan

        last28 = q_ok[g[DATE] > last_day - pd.Timedelta(days=28)].mean()
        prev28 = q_ok[(g[DATE] <= last_day - pd.Timedelta(days=28))
                      & (g[DATE] > last_day - pd.Timedelta(days=56))].mean()
        trend = float(last28 / prev28 - 1) if prev28 and prev28 > 0 and np.isfinite(last28) else np.nan

        # год к году: те же 28 дней год назад
        yoy = np.nan
        ly = q_ok[(g[DATE] > last_day - pd.Timedelta(days=28 + 365))
                  & (g[DATE] <= last_day - pd.Timedelta(days=365))].mean()
        if np.isfinite(ly) and ly > 0 and np.isfinite(last28):
            yoy = float(last28 / ly - 1)

        dow = q_ok.groupby(g.loc[q_ok.index, DATE].dt.dayofweek).mean()
        dow_strength, peak, low = np.nan, None, None
        if len(dow) == 7 and dow.mean() > 0:
            dow_strength = float((dow.max() - dow.min()) / dow.mean())
            peak, low = WEEKDAYS_SHORT[int(dow.idxmax())], WEEKDAYS_SHORT[int(dow.idxmin())]

        mon = q_ok.groupby(g.loc[q_ok.index, DATE].dt.month).mean()
        season_strength = float((mon.max() - mon.min()) / mon.mean()) \
            if len(mon) >= 12 and mon.mean() > 0 else np.nan
        peak_month = int(mon.idxmax()) if len(mon) >= 12 else None

        row = {"sku": str(sku),
               "abc_units": prof.at[sku, "abc_class"],
               "abc_money": abc_money.get(sku) if abc_money else None,
               "xyz": _xyz(cv), "cv_weekly": cv,
               "demand_class": prof.at[sku, "demand_class"],
               "demand_label": CLASS_LABELS_RU.get(prof.at[sku, "demand_class"], ""),
               "per_day": float(prof.at[sku, "mean_daily"]),
               "per_day_recent": float(last28) if np.isfinite(last28) else np.nan,
               "trend_28": trend, "yoy_28": yoy,
               "weekday_strength": dow_strength, "peak_weekday": peak,
               "low_weekday": low,
               "season_strength": season_strength, "peak_month": peak_month,
               "stockout_share": float(cens.mean()),
               "zero_share": float((q <= 0).mean())}

        if has_stock:
            st = g[STOCK].astype(float)
            last = st.dropna().iloc[-1] if st.notna().any() else np.nan
            row["stock"] = float(last) if np.isfinite(last) else np.nan
            base = last28 if np.isfinite(last28) and last28 > 0 else prof.at[sku, "mean_daily"]
            row["stock_days"] = float(last / base) if np.isfinite(last) and base > 0 else np.nan
            avg_stock = float(st.mean())
            row["turnover_year"] = float(q.mean() * 365 / avg_stock) if avg_stock > 0 else np.nan
        if has_price:
            p = g[PRICE].astype(float)
            pos = p[p > 0]
            row["price"] = float(pos.iloc[-1]) if len(pos) else np.nan
            if len(pos) >= 60:
                early, late = pos.iloc[:30].median(), pos.iloc[-30:].median()
                row["price_change"] = float(late / early - 1) if early > 0 else np.nan
        if has_price and has_cost:
            c = g[COST].astype(float)
            pr, co = row.get("price", np.nan), c[c > 0].iloc[-1] if (c > 0).any() else np.nan
            row["margin_pct"] = float((pr - co) / pr) if np.isfinite(pr) and pr > 0 and np.isfinite(co) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def explain_sku(r: dict) -> list[str]:
    """Пара фраз на человеческом языке по строке enrich_by_sku()."""
    out = []
    def ok(v): return v is not None and np.isfinite(v)
    if ok(r.get("trend_28")) and abs(r["trend_28"]) >= 0.15:
        out.append(f"Продажи {'растут' if r['trend_28'] > 0 else 'падают'}: "
                   f"{r['trend_28'] * 100:+.0f} % к предыдущему месяцу.")
    if ok(r.get("weekday_strength")) and r["weekday_strength"] >= 0.3:
        out.append(f"Сильнее всего покупают в {r['peak_weekday']}, слабее всего — в {r['low_weekday']}.")
    if ok(r.get("stockout_share")) and r["stockout_share"] >= 0.05:
        out.append(f"Товара не было {r['stockout_share'] * 100:.0f} % дней — часть спроса потеряна.")
    if ok(r.get("season_strength")) and r["season_strength"] >= 0.5:
        out.append("Выраженная сезонность: продажи сильно зависят от времени года.")
    return out


def feature_overview(df: pd.DataFrame) -> dict:
    """Всё для вкладки «Признаки»: каталог модели + таблица показателей."""
    en = enrich_by_sku(df)
    table = []
    for r in en.to_dict("records"):
        r = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
             for k, v in r.items()}
        r["notes"] = explain_sku({k: (np.nan if v is None else v) for k, v in r.items()})
        table.append(r)

    derived = [
        {"code": "abc", "title": "ABC-класс", "used_by_model": False,
         "description": "A — товары, дающие 80 % оборота; B — следующие 15 %; "
                        "C — остальные 5 %. Показывает, на что смотреть в первую очередь."},
        {"code": "xyz", "title": "XYZ-класс", "used_by_model": False,
         "description": "Насколько стабильны недельные продажи: X — стабильно, "
                        "Y — колеблются, Z — трудно предсказать."},
        {"code": "trend", "title": "Тренд и год к году", "used_by_model": False,
         "description": "Последние 28 дней против предыдущих 28 и против того же "
                        "периода год назад."},
        {"code": "weekday", "title": "Дни недели", "used_by_model": False,
         "description": "В какой день товар покупают лучше всего и хуже всего."},
        {"code": "season", "title": "Сезонность", "used_by_model": False,
         "description": "Разница между лучшим и худшим месяцем года (нужно 12+ месяцев истории)."},
        {"code": "stockout", "title": "Доля дней дефицита", "used_by_model": False,
         "description": "Сколько дней товара не было на складе."},
        {"code": "turnover", "title": "Оборачиваемость и запас в днях", "used_by_model": False,
         "description": "Сколько раз за год склад «перекручивается» и на сколько дней хватит остатка."},
        {"code": "margin", "title": "Маржа и динамика цены", "used_by_model": False,
         "description": "Доля прибыли в цене и как изменилась цена за период."},
    ]
    return {"model_features": model_feature_catalog(),
            "model_feature_total": len(FEATURE_COLUMNS),
            "derived": derived, "xyz_labels": XYZ_RU, "table": table}
