"""
Каталог аномалий в данных: что нашли, чем это грозит и что с этим сделали.

ДВА РОДА НАХОДОК
----------------
1. ИСПРАВЛЕНО АВТОМАТИЧЕСКИ — то, что очистка уже поправила: выбросы,
   возвраты, дубли, пропущенные даты, дни дефицита. Тут важно показать
   пользователю, ЧТО БЫЛО в файле и ВО ЧТО это превратилось: слепое
   «мы всё почистили» вызывает недоверие к результату.

2. ТРЕБУЕТ ВНИМАНИЯ — то, что исправить без человека нельзя, потому что
   причина известна только ему: товар не продаётся уже два месяца (снят с
   продажи или закончился?), остаток падает быстрее продаж (кража? брак?
   недосдача?), закупочная цена выше цены продажи (опечатка или убыток?).
   Эти строки не меняются — они выносятся на экран с вопросом.

Каждая находка описана одинаково: что это, чем мешает расчёту, что сделали
или что стоит проверить. Тот же текст показывается в интерфейсе и попадает
в отчёт для записки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import CleaningReport
from .schema import (
    COST, DATE, FLAG_CENSORED, FLAG_FILLED, FLAG_RETURN, FLAG_WINSORIZED,
    MIN_HISTORY_DAYS, PRICE, QTY, SKU, STOCK,
)

FIXED = "fixed"            # поправлено автоматически
ATTENTION = "attention"    # нужен взгляд человека
INFO = "info"              # просто к сведению

SEVERITY_LABELS_RU = {
    FIXED: "Исправлено автоматически",
    ATTENTION: "Проверьте",
    INFO: "К сведению",
}

MAX_EXAMPLES = 8
IDLE_MIN_DAYS = 30          # столько дней без продаж уже подозрительно
STUCK_STOCK_DAYS = 14       # остаток не меняется при продажах — столько дней подряд
STALE_END_DAYS = 14         # данные по товару закончились раньше остальных


def _fmt_date(d) -> str | None:
    return pd.Timestamp(d).strftime("%d.%m.%Y") if pd.notna(d) else None


def _num(v: float | None, digits: int = 1):
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), digits)


def _group(code: str, severity: str, title: str, count: int, what: str,
           why: str, action: str, skus: list[str] | None = None,
           examples: list[dict] | None = None, unit: str = "случаев") -> dict:
    return {
        "code": code, "severity": severity,
        "severity_label": SEVERITY_LABELS_RU[severity],
        "title": title, "count": int(count), "unit": unit,
        "what": what, "why": why, "action": action,
        "skus": (skus or [])[:10], "skus_total": len(skus or []),
        "examples": (examples or [])[:MAX_EXAMPLES],
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Непрерывные отрезки True: [(начало, конец включительно), ...]."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


# --- находки, найденные ВО ВРЕМЯ очистки -----------------------------------------

def _from_cleaning(rep: CleaningReport) -> list[dict]:
    by_kind: dict[str, list[dict]] = {}
    for c in rep.changes:
        by_kind.setdefault(c["kind"], []).append(c)

    def examples(kind: str, fmt) -> list[dict]:
        return [{"sku": c["sku"], "date": _fmt_date(c["date"]), "detail": fmt(c)}
                for c in by_kind.get(kind, [])]

    out: list[dict] = []

    if rep.outliers_winsorized:
        sk = list(dict.fromkeys(c["sku"] for c in by_kind.get("outlier", [])))
        out.append(_group(
            "outlier", FIXED, "Резкие всплески продаж", rep.outliers_winsorized,
            "Продажи за день в разы выше обычных для этого товара: крупная "
            "разовая отгрузка, ошибка ввода или распродажа.",
            "Один такой день раздувает среднее и заставляет систему закладывать "
            "лишний запас «на всякий случай», хотя повторения не будет.",
            "Значение ограничено порогом «медиана + 6 отклонений» — день остаётся "
            "в данных, но не тянет прогноз вверх.",
            sk, examples("outlier", lambda c: f"было {c['before']:g} шт → стало "
                                              f"{c['after']:g} шт"),
            "дней"))

    if rep.returns_zeroed:
        out.append(_group(
            "return", FIXED, "Возвраты (отрицательные продажи)", rep.returns_zeroed,
            "В выгрузке есть строки с отрицательным количеством — покупатель "
            "вернул товар.",
            "Возврат — не «антиспрос». Если оставить минус, система решит, "
            "что товар продаётся хуже, чем на самом деле.",
            "Заменено на ноль, строка помечена как возврат.",
            list(dict.fromkeys(c["sku"] for c in by_kind.get("return", []))),
            examples("return", lambda c: f"было {c['before']:g} шт → 0"), "строк"))

    if rep.negative_stock_fixed:
        out.append(_group(
            "negative_stock", FIXED, "Отрицательный остаток", rep.negative_stock_fixed,
            "Остаток на складе в файле меньше нуля — так бывает, когда продажи "
            "проведены раньше прихода товара в учёте.",
            "Физически остаток не может быть отрицательным; с таким числом "
            "нельзя считать, на сколько дней хватит товара.",
            "Заменено на ноль.",
            list(dict.fromkeys(c["sku"] for c in by_kind.get("negative_stock", []))),
            examples("negative_stock", lambda c: f"было {c['before']:g} → 0"), "строк"))

    if rep.duplicates_merged:
        out.append(_group(
            "duplicate", FIXED, "Дубли записей", rep.duplicates_merged,
            "По одному товару за один день в файле несколько строк.",
            "Если не сложить их, продажи дня окажутся разбитыми на части и "
            "прогноз по дням будет неверным.",
            "Строки объединены: продажи сложены, остаток взят последний за день.",
            list(dict.fromkeys(c["sku"] for c in by_kind.get("duplicate", []))),
            examples("duplicate", lambda c: f"{c['before']} в одном дне → в сумме {c['after']:g} шт"),
            "строк"))

    if rep.dates_filled:
        out.append(_group(
            "gap", FIXED, "Пропущенные даты", rep.dates_filled,
            "Для части дней в файле нет строк — например, в выходные или в дни, "
            "когда товар не продавался.",
            "Для прогноза отсутствие строки означает «продано 0», а не «нет "
            "данных». Пропуски без заполнения сломали бы расчёт средних и лагов.",
            "Недостающие дни добавлены с нулевыми продажами, остаток перенесён "
            "с предыдущего дня.", [], [], "дней"))

    if rep.sku_merged:
        out.append(_group(
            "merged_names", FIXED, "Похожие названия товаров", len(rep.sku_merged),
            "Один и тот же товар записан по-разному: «Молоко 1 л» и «Молоко (1 литр)».",
            "Система сочла бы их двумя разными товарами, у каждого история "
            "оказалась бы вдвое короче.",
            "Названия объединены под самым частым вариантом.",
            [b for _a, b in rep.sku_merged],
            [{"sku": b, "date": None, "detail": f"«{a}» → «{b}»"}
             for a, b in rep.sku_merged], "названий"))

    dropped = (rep.rows_dropped_bad_date + rep.rows_dropped_bad_qty
               + rep.rows_dropped_empty_sku)
    if dropped:
        parts = []
        if rep.rows_dropped_bad_date:
            parts.append(f"{rep.rows_dropped_bad_date} без понятной даты")
        if rep.rows_dropped_bad_qty:
            parts.append(f"{rep.rows_dropped_bad_qty} без количества")
        if rep.rows_dropped_empty_sku:
            parts.append(f"{rep.rows_dropped_empty_sku} без названия товара")
        out.append(_group(
            "dropped", FIXED, "Строки, которые нельзя использовать", dropped,
            "Строки, где не распознана дата, количество или название товара: "
            + ", ".join(parts) + ".",
            "Без любого из этих полей строку невозможно привязать ни к дню, "
            "ни к товару.",
            "Строки пропущены. Если их много — проверьте формат дат в файле.",
            [], [], "строк"))

    return out


# --- находки, найденные в ОЧИЩЕННЫХ данных --------------------------------------

def _censored_group(df: pd.DataFrame, rep: CleaningReport) -> dict | None:
    if FLAG_CENSORED not in df.columns or not df[FLAG_CENSORED].any():
        return None
    ex, per_sku = [], {}
    for sku, g in df.groupby(SKU, sort=False):
        c = g[FLAG_CENSORED].to_numpy(bool)
        if not c.any():
            continue
        per_sku[str(sku)] = int(c.sum())
        dates = g[DATE].to_numpy()
        for a, b in _runs(c)[:3]:
            n = b - a + 1
            ex.append({"sku": str(sku), "date": _fmt_date(dates[a]),
                       "detail": (f"{n} дн. подряд без товара" if n > 1
                                  else "1 день без товара")})
    top = sorted(per_sku, key=per_sku.get, reverse=True)
    total = int(df[FLAG_CENSORED].sum())
    ex.sort(key=lambda e: -int(e["detail"].split()[0]))
    return _group(
        "censored", FIXED, "Дни дефицита (товара не было)", total,
        "Продажи равны нулю, и остаток тоже ноль: спроса не было не потому, что "
        "никто не хотел купить, а потому, что покупать было нечего.",
        "Если считать такие дни честными нулями, система научится занижать "
        "спрос — и будет закладывать всё меньше запаса, усугубляя дефицит.",
        "Эти дни исключены из обучения и из расчёта среднего спроса.",
        top, ex, "дней")


def _idle_group(df: pd.DataFrame) -> dict | None:
    last_day = df[DATE].max()
    rows, skus = [], []
    for sku, g in df.groupby(SKU, sort=False):
        g = g.sort_values(DATE)
        sold = g[g[QTY] > 0]
        if sold.empty or len(g) < 60:
            continue
        idle = int((last_day - sold[DATE].max()).days)
        before = g[g[DATE] <= sold[DATE].max()]
        mean_daily = float(before[QTY].mean())
        # интервал, который для этого товара нормален: у редкого товара
        # месяц без продаж — обычное дело, у ходового — нет
        gaps = sold[DATE].diff().dt.days.dropna()
        usual = float(gaps.quantile(0.95)) if len(gaps) >= 5 else 7.0
        if idle >= max(IDLE_MIN_DAYS, 3 * usual) and mean_daily > 0:
            stock = g[STOCK].iloc[-1] if STOCK in g.columns else np.nan
            skus.append(str(sku))
            tail = (f"остаток {stock:g} шт лежит без движения"
                    if np.isfinite(stock) and stock > 0 else "остаток нулевой")
            rows.append({"sku": str(sku), "date": _fmt_date(sold[DATE].max()),
                         "detail": f"последняя продажа {idle} дн. назад; {tail}",
                         "_idle": idle})
    if not rows:
        return None
    rows.sort(key=lambda r: -r["_idle"])
    for r in rows:
        r.pop("_idle")
    return _group(
        "idle", ATTENTION, "Товар давно не продаётся", len(rows),
        "По товару нет ни одной продажи уже давно — заметно дольше, чем "
        "обычно между его продажами.",
        "Либо товар снят с продажи (тогда заказывать его не нужно), либо он "
        "закончился и не завозится (тогда вы теряете выручку). Система не "
        "может отличить одно от другого — это знаете вы.",
        "Данные не менялись. Проверьте, продаётся ли товар; если снят — "
        "уберите его из выгрузки.", skus, rows, "товаров")


def _stale_end_group(df: pd.DataFrame) -> dict | None:
    last_day = df[DATE].max()
    rows, skus = [], []
    for sku, g in df.groupby(SKU, sort=False):
        end = g[DATE].max()
        lag = int((last_day - end).days)
        if lag >= STALE_END_DAYS:
            skus.append(str(sku))
            rows.append({"sku": str(sku), "date": _fmt_date(end),
                         "detail": f"данные заканчиваются на {lag} дн. раньше остальных"})
    if not rows:
        return None
    return _group(
        "stale_end", ATTENTION, "Данные по товару закончились раньше остальных",
        len(rows),
        "Последняя строка по этим товарам намного старше, чем по остальным.",
        "Остаток «на сегодня» по такому товару взят с давней даты и, скорее "
        "всего, неверен — рекомендация по нему будет ненадёжной.",
        "Данные не менялись. Обновите выгрузку по этим товарам.",
        skus, rows, "товаров")


def _shrink_group(df: pd.DataFrame) -> dict | None:
    """Остаток упал сильнее, чем продали: недостача, списание, брак, ошибка учёта."""
    if STOCK not in df.columns or df[STOCK].notna().sum() < 50:
        return None
    rows, total_units, skus = [], 0.0, {}
    for sku, g in df.groupby(SKU, sort=False):
        g = g.sort_values(DATE).reset_index(drop=True)
        stock = g[STOCK].astype(float)
        if stock.notna().sum() < 30:
            continue
        # что должно было остаться: вчерашний остаток минус сегодняшние продажи
        # (поставки только увеличивают, поэтому разница не может быть отрицательной)
        gap = stock.shift(1) - g[QTY].astype(float) - stock
        skip = pd.Series(False, index=g.index)
        for flag in (FLAG_WINSORIZED, FLAG_FILLED, FLAG_RETURN):
            if flag in g.columns:
                skip |= g[flag].fillna(False).astype(bool)
        # день после дня с нулевым остатком — это начало дефицита, не недостача
        skip |= stock.shift(1).fillna(0) <= 0
        tol = max(2.0, 0.25 * float(g[QTY].mean()))
        bad = (gap > tol) & ~skip & stock.notna() & stock.shift(1).notna()
        if bad.any():
            n = int(bad.sum())
            units = float(gap[bad].sum())
            skus[str(sku)] = units
            total_units += units
            worst = gap[bad].idxmax()
            rows.append({"sku": str(sku), "date": _fmt_date(g.at[worst, DATE]),
                         "detail": (f"{n} раз; всего не хватает {units:g} шт; "
                                    f"крупнейшее расхождение {gap[worst]:g} шт"),
                         "_u": units})
    if not rows:
        return None
    rows.sort(key=lambda r: -r["_u"])
    for r in rows:
        r.pop("_u")
    return _group(
        "shrink", ATTENTION, "Остаток падает быстрее продаж", len(rows),
        "Остаток за день уменьшился сильнее, чем продано. Разница ушла «в никуда».",
        "Это может быть недостача, списание, брак, кража или неучтённая "
        "отгрузка. Для прогноза спроса это не страшно, но деньги теряются, а "
        "реальный остаток ниже, чем думает система.",
        "Данные не менялись. Сверьте склад с учётом по этим товарам.",
        sorted(skus, key=skus.get, reverse=True), rows, "товаров")


def _stuck_stock_group(df: pd.DataFrame) -> dict | None:
    if STOCK not in df.columns or df[STOCK].notna().sum() < 50:
        return None
    rows, skus = [], []
    for sku, g in df.groupby(SKU, sort=False):
        g = g.sort_values(DATE).reset_index(drop=True)
        stock = g[STOCK].astype(float)
        stuck = ((stock.diff() == 0) & (g[QTY] > 0) & (stock > 0)).to_numpy()
        runs = [r for r in _runs(stuck) if r[1] - r[0] + 1 >= STUCK_STOCK_DAYS]
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            skus.append(str(sku))
            rows.append({"sku": str(sku), "date": _fmt_date(g.at[a, DATE]),
                         "detail": f"остаток {stock[a]:g} шт не менялся "
                                   f"{b - a + 1} дн., хотя товар продавался"})
    if not rows:
        return None
    return _group(
        "stuck_stock", ATTENTION, "Остаток «застрял»", len(rows),
        "Остаток неделями не меняется, хотя продажи идут.",
        "Похоже, остатки по этому товару не обновлялись. Расчёт «на сколько "
        "хватит» опирается на последний остаток, и он может быть далёк от правды.",
        "Данные не менялись. Проверьте, что остатки выгружаются на каждую дату.",
        skus, rows, "товаров")


def _price_group(df: pd.DataFrame) -> dict | None:
    if PRICE not in df.columns or not df[PRICE].notna().any():
        return None
    rows, skus = [], []
    for sku, g in df.groupby(SKU, sort=False):
        price = g[PRICE].astype(float)
        p = price[price > 0]
        if len(p) < 10:
            continue
        med = float(p.median())
        issues = []
        wild = ((price > 3 * med) | ((price > 0) & (price < 0.3 * med))).sum()
        if wild:
            issues.append(f"{int(wild)} дн. цена отличается от обычной ({med:.2f} ₽) "
                          "в 3 и более раза")
        if COST in g.columns and g[COST].notna().any():
            under = ((g[PRICE] > 0) & (g[COST] > 0) & (g[PRICE] <= g[COST])).sum()
            if under:
                issues.append(f"{int(under)} дн. цена продажи не выше закупочной")
        if issues:
            skus.append(str(sku))
            rows.append({"sku": str(sku), "date": None, "detail": "; ".join(issues)})
    if not rows:
        return None
    return _group(
        "price", ATTENTION, "Подозрительные цены", len(rows),
        "Цена за день сильно выходит за привычный диапазон товара или "
        "оказывается не выше закупочной.",
        "От цены зависит, сколько стоит нехватка товара и сколько — его "
        "избыток; ошибочная цена сдвигает оптимальный запас.",
        "Данные не менялись. Проверьте цены — это опечатка или реальная "
        "акция с убытком.", skus, rows, "товаров")


def _future_group(df: pd.DataFrame) -> dict | None:
    limit = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)
    bad = df[df[DATE] > limit]
    if bad.empty:
        return None
    return _group(
        "future", ATTENTION, "Даты из будущего", int(len(bad)),
        "В файле есть строки с датами, которые ещё не наступили.",
        "Скорее всего, ошибка формата даты (день и месяц перепутаны). "
        "Из-за неё система считает, что данные заканчиваются позже, чем есть на самом деле.",
        "Данные не менялись. Проверьте формат дат.",
        list(bad[SKU].astype(str).unique()),
        [{"sku": str(r[SKU]), "date": _fmt_date(r[DATE]), "detail": "дата в будущем"}
         for _, r in bad.head(MAX_EXAMPLES).iterrows()], "строк")


def _short_group(df: pd.DataFrame) -> dict | None:
    rows, skus = [], []
    for sku, g in df.groupby(SKU, sort=False):
        days = int((g[DATE].max() - g[DATE].min()).days) + 1
        if days < MIN_HISTORY_DAYS:
            skus.append(str(sku))
            rows.append({"sku": str(sku), "date": None,
                         "detail": f"история {days} дн., нужно не меньше {MIN_HISTORY_DAYS}"})
    if not rows:
        return None
    return _group(
        "short", ATTENTION, "Слишком короткая история", len(rows),
        f"По товару меньше {MIN_HISTORY_DAYS} дней данных.",
        "На таком отрезке нельзя отличить закономерность от случайности — "
        "прогноз был бы выдумкой.",
        "Прогноз по этим товарам не строится; закупайте их вручную, пока не "
        "накопится история.", skus, rows, "товаров")


def _zero_group(df: pd.DataFrame) -> dict | None:
    rows, skus = [], []
    for sku, g in df.groupby(SKU, sort=False):
        z = float((g[QTY] <= 0).mean())
        if z > 0.7 and len(g) >= 30:
            skus.append(str(sku))
            rows.append({"sku": str(sku), "date": None,
                         "detail": f"{z * 100:.0f} % дней без продаж"})
    if not rows:
        return None
    return _group(
        "sparse", INFO, "Редкие продажи", len(rows),
        "Больше 70 % дней у товара — без единой продажи.",
        "Продажи такого товара случайны: «сколько продадим завтра» "
        "предсказать нельзя, можно оценить лишь объём за месяц.",
        "Для таких товаров используется упрощённое правило мин/макс "
        "вместо прогноза модели.", skus, rows, "товаров")


def build_anomaly_report(df: pd.DataFrame, rep: CleaningReport) -> dict:
    """
    Полный каталог аномалий для экрана «Данные».

    Возвращает группы находок (исправленные и требующие внимания),
    итоговую оценку здоровья данных и короткую сводку одной строкой.
    """
    groups = _from_cleaning(rep)
    for fn in (_censored_group,):
        g = fn(df, rep)
        if g:
            groups.append(g)
    for fn in (_idle_group, _stale_end_group, _shrink_group, _stuck_stock_group,
               _price_group, _future_group, _short_group, _zero_group):
        g = fn(df)
        if g:
            groups.append(g)

    order = {ATTENTION: 0, FIXED: 1, INFO: 2}
    groups.sort(key=lambda g: (order[g["severity"]], -g["count"]))

    n_fixed = sum(1 for g in groups if g["severity"] == FIXED)
    n_attn = sum(1 for g in groups if g["severity"] == ATTENTION)
    rows = max(rep.rows_final, 1)
    touched = (rep.outliers_winsorized + rep.returns_zeroed
               + rep.negative_stock_fixed + rep.censored_days)
    share = touched / rows

    if n_attn == 0 and share < 0.03:
        health, health_label = "good", "Данные в хорошем состоянии"
    elif n_attn <= 4 and share < 0.10:
        health, health_label = "fair", "Данные пригодны, есть что проверить"
    else:
        health, health_label = "poor", "В данных много проблем — проверьте найденное"

    if not groups:
        headline = "Аномалий не найдено — данные чистые."
    else:
        bits = []
        if n_fixed:
            bits.append(f"исправлено автоматически — {n_fixed}")
        if n_attn:
            bits.append(f"нужно проверить вам — {n_attn}")
        headline = "Типов находок: " + "; ".join(bits) + "."

    return {
        "health": health, "health_label": health_label, "headline": headline,
        "groups": groups,
        "totals": {"fixed_kinds": n_fixed, "attention_kinds": n_attn,
                   "rows_touched": int(touched), "rows_touched_pct": round(share * 100, 1)},
    }
