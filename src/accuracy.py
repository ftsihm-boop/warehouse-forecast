"""
Точность прогноза — для человека, а не для математика.

Менеджеру проценты ничего не говорят: «WAPE 23 %» — хорошо это или плохо?
Поэтому наружу выходит оценка из трёх слов — «отличная», «хорошая»,
«низкая» — и короткое объяснение; цифры спрятаны в «Подробнее».

КАК ИЗМЕРЯЕМ
------------
Проверка на прошлом: система «забывает», что было после выбранной даты,
строит прогноз так, как построила бы тогда, и сравнивает с тем, что
случилось на самом деле. Так повторяется на нескольких датах в последние
месяцы. Будущее модели при этом недоступно (walk-forward), поэтому
оценка честная: это не подгонка под известный ответ.

Три измерения:
    ошибка (WAPE)  — насколько прогноз промахивается в среднем;
    смещение       — систематически занижает или завышает;
    запас          — как часто факт уложился в верхнюю границу (q90), ведь
                     именно на неё опирается страховой запас.

И сравнение с простейшим правилом «завтра как в среднем за последние 28
дней»: если сложная модель не лучше простого правила, пользы от неё нет, и
об этом нужно говорить прямо.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .demand_classes import NOT_FORECASTABLE, classify_series
from .features import (
    BASE_COL, FEATURE_COLUMNS, TARGET_ABS, build_features,
)
from .forecast import stats_forecast
from .model import QuantileModel
from .schema import DATE, FLAG_CENSORED, QTY, SKU

EXCELLENT, GOOD, LOW, UNKNOWN, NONE = "excellent", "good", "low", "unknown", "none"

LEVEL_LABELS_RU = {
    EXCELLENT: "Отличная", GOOD: "Хорошая", LOW: "Низкая",
    UNKNOWN: "Недостаточно данных", NONE: "Не прогнозируется",
}

# Пороги по ошибке суммы продаж за горизонт, %
EXCELLENT_WAPE = 15.0
GOOD_WAPE = 30.0
BIAS_LIMIT = 15.0          # систематический перекос, % — понижает оценку на ступень
MIN_COVERAGE = 0.75
MIN_HISTORY = 150          # меньше — оценить честно нельзя
MAX_ORIGINS = 10


def _level_from_wape(w: float) -> str:
    return EXCELLENT if w <= EXCELLENT_WAPE else GOOD if w <= GOOD_WAPE else LOW


def _down(level: str) -> str:
    return {EXCELLENT: GOOD, GOOD: LOW}.get(level, level)


def _sku_check(g: pd.DataFrame, model: QuantileModel | None, horizon: int,
               test_days: int) -> dict | None:
    """Скользящая проверка одного товара. None — данных мало."""
    g = g.sort_values(DATE).reset_index(drop=True)
    n = len(g)
    if n < MIN_HISTORY:
        return None
    step = max(7, horizon // 2)
    last_origin = n - 1 - horizon
    first_origin = max(MIN_HISTORY - 30, last_origin - test_days)
    if last_origin <= first_origin:
        return None
    origins = list(range(last_origin, first_origin, -step))[:MAX_ORIGINS]

    qty = g[QTY].astype(float).to_numpy()
    cens = g[FLAG_CENSORED].astype(bool).to_numpy() if FLAG_CENSORED in g.columns \
        else np.zeros(n, bool)

    feats = None
    if model is not None:
        f = build_features(g.assign(**{SKU: g[SKU].iloc[0]}), horizon=horizon,
                           with_target=False)
        f = f.sort_values(DATE).reset_index(drop=True)
        cols = [c for c in (model.features or FEATURE_COLUMNS) if c in f.columns]
        feats = (f, cols)

    actual, pred50, pred90, naive = [], [], [], []
    for o in origins:
        window = slice(o + 1, o + 1 + horizon)
        # дни дефицита в проверяемом окне: фактические продажи там занижены
        # нехваткой товара, и сравнивать с ними прогноз спроса нечестно
        if cens[window].mean() > 0.25:
            continue
        act = float(qty[window].sum())
        hist = qty[:o + 1]
        hc = cens[:o + 1]
        base = np.nanmean(np.where(hc, np.nan, hist)[-28:])
        if not np.isfinite(base):
            continue
        p50 = p90 = None
        if feats is not None:
            f, cols = feats
            row = f.iloc[[o]]
            b = float(row[BASE_COL].iloc[0]) if BASE_COL in row else np.nan
            if np.isfinite(b) and b > 0:
                lo, hi = model.predict(row[cols].fillna(0.0))
                p50, p90 = float(lo[0] * b * horizon), float(hi[0] * b * horizon)
        if p50 is None:
            p50, p90 = stats_forecast(g.iloc[:o + 1], horizon)
        actual.append(act)
        pred50.append(p50)
        pred90.append(max(p90, p50))
        naive.append(base * horizon)

    if len(actual) < 3:
        return None
    a, p, h, nv = map(np.array, (actual, pred50, pred90, naive))
    tot = a.sum()
    if tot <= 0:
        return None
    wape = float(np.abs(a - p).sum() / tot * 100)
    wape_naive = float(np.abs(a - nv).sum() / tot * 100)
    bias = float((p.sum() - tot) / tot * 100)
    cover = float((a <= h + 1e-9).mean())
    return {"checks": len(a), "wape": wape, "wape_naive": wape_naive,
            "bias": bias, "q90_coverage": cover, "actual_avg": float(a.mean()),
            "pred_avg": float(p.mean())}


def _explain(level: str, m: dict) -> tuple[str, list[str]]:
    """Короткая фраза и список пояснений для «Подробнее»."""
    w = m["wape"]
    short = {
        EXCELLENT: "Прогноз обычно почти совпадает с фактом.",
        GOOD: "Прогноз в целом верный, но возможны заметные отклонения.",
        LOW: "Прогноз часто промахивается — опирайтесь на него осторожно.",
    }[level]
    notes = [f"За выбранный период прогноз в среднем отличался от фактических "
             f"продаж на {w:.0f} %.".replace(",", ".")]
    if m["bias"] >= BIAS_LIMIT:
        notes.append(f"Прогноз систематически завышает продажи (в среднем на "
                     f"{m['bias']:.0f} %) — есть риск излишка на складе.")
    elif m["bias"] <= -BIAS_LIMIT:
        notes.append(f"Прогноз систематически занижает продажи (в среднем на "
                     f"{-m['bias']:.0f} %) — есть риск нехватки товара.")
    else:
        notes.append("Систематического перекоса нет: прогноз не завышает и не занижает.")
    notes.append(f"Страховой запас хватало в {m['q90_coverage'] * 100:.0f} % проверок "
                 "(цель — около 90 %).")
    gain = m["wape_naive"] - w
    if m["wape_naive"] > 0 and gain >= 3:
        notes.append(f"Это на {gain:.0f} п.п. точнее простого правила «продажи как за "
                     "последние 4 недели».")
    elif gain <= -3:
        notes.append("Простое правило «как за последние 4 недели» здесь работало "
                     "точнее модели — пользы от неё мало.")
    else:
        notes.append("По точности близко к простому правилу «как за последние 4 недели».")
    return short, notes


def assess_accuracy(df: pd.DataFrame, horizon: int = 14,
                    model: QuantileModel | None = None,
                    model_dir: str | Path | None = "models/global",
                    test_days: int = 120) -> dict:
    """
    Оценка точности по каждому товару и по всему ассортименту.

    horizon — на сколько дней вперёд проверяем (берётся тот же горизонт,
    что и в плане заказа: именно на нём пользователь принимает решения).
    """
    if model is None and model_dir is not None:
        p = Path(model_dir)
        if (p / "meta.json").exists():
            try:
                model = QuantileModel.load(p)
            except Exception:  # noqa: BLE001
                model = None

    items, weights = [], []
    for sku, g in df.groupby(SKU, sort=False):
        cls, *_ = classify_series(
            g[QTY], g[FLAG_CENSORED] if FLAG_CENSORED in g.columns else None)
        if cls in NOT_FORECASTABLE:
            items.append({"sku": str(sku), "level": NONE,
                          "level_label": LEVEL_LABELS_RU[NONE],
                          "short": "Спрос слишком нерегулярный: прогноз не строится, "
                                   "используется правило мин/макс.",
                          "notes": [], "metrics": None})
            continue
        m = _sku_check(g, model, horizon, test_days)
        if m is None:
            items.append({"sku": str(sku), "level": UNKNOWN,
                          "level_label": LEVEL_LABELS_RU[UNKNOWN],
                          "short": "Слишком мало истории, чтобы честно проверить прогноз.",
                          "notes": [], "metrics": None})
            continue
        level = _level_from_wape(m["wape"])
        if abs(m["bias"]) > BIAS_LIMIT or m["q90_coverage"] < MIN_COVERAGE:
            level = _down(level)
        short, notes = _explain(level, m)
        items.append({"sku": str(sku), "level": level,
                      "level_label": LEVEL_LABELS_RU[level], "short": short,
                      "notes": notes,
                      "metrics": {k: round(v, 1) if isinstance(v, float) else v
                                  for k, v in m.items()}})
        weights.append((level, float(g[QTY].sum())))

    if weights:
        score = {EXCELLENT: 2, GOOD: 1, LOW: 0}
        tw = sum(w for _, w in weights) or 1.0
        avg = sum(score[l] * w for l, w in weights) / tw
        overall = EXCELLENT if avg >= 1.6 else GOOD if avg >= 0.8 else LOW
    else:
        overall = UNKNOWN

    counts = {k: sum(1 for i in items if i["level"] == k)
              for k in (EXCELLENT, GOOD, LOW, UNKNOWN, NONE)}
    short = {
        EXCELLENT: "В целом прогноз точный — на него можно опираться.",
        GOOD: "В целом прогноз надёжный; по отдельным товарам возможны отклонения.",
        LOW: "Точность невысокая: используйте рекомендации как ориентир и "
             "перепроверяйте крупные заказы.",
        UNKNOWN: "Данных слишком мало, чтобы оценить точность.",
    }[overall]

    overall_notes = [
        "Оценка получена проверкой на прошлом: система делала прогноз так, будто "
        "не знает последних месяцев, и сравнивала с тем, что случилось на самом деле.",
        f"Проверялся прогноз на {horizon} дн. вперёд по нескольким датам в последние "
        f"{test_days} дней.",
        "«Отличная» — ошибка до 15 %, «хорошая» — до 30 %, «низкая» — больше. "
        "Систематический перекос или нехватка страхового запаса снижают оценку "
        "на ступень.",
    ]
    return {"overall": overall, "overall_label": LEVEL_LABELS_RU[overall],
            "short": short, "notes": overall_notes, "counts": counts,
            "horizon": horizon, "items": items}
