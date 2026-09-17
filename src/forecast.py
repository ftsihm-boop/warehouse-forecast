"""
Гибридный прогноз — вариант C.

Режим выбирается ПО КАЖДОМУ ТОВАРУ ОТДЕЛЬНО, а не по файлу целиком:

    >= 180 дней истории  -> FINE_TUNE  дообучаем глобальную модель на данных
                                       пользователя (CatBoost init_model)
    60..179 дней         -> GLOBAL     предобученная модель как есть
    14..59 дней          -> STATS      скользящее среднее + профиль дня недели,
                                       в интерфейсе честная плашка
    < 14 дней            -> REJECT     отказ с объяснением

Возвращает по каждому SKU медианный прогноз (q50) и верхнюю границу (q90)
за выбранный горизонт — ровно то, что нужно модулю закупки.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import pick_mode
from .features import (
    BASE_COL, FEATURE_COLUMNS, TARGET_COL, build_features, build_training_set,
    inference_rows,
)
from .model import QuantileModel
from .demand_classes import (
    CLASS_LABELS_RU, NOT_FORECASTABLE, STATS_ONLY, STRATEGY_RU, classify_series,
)
from .schema import (
    DATE, FLAG_CENSORED, MODE_FINE_TUNE, MODE_GLOBAL, MODE_LABELS_RU,
    MODE_MINMAX, MODE_REJECT, MODE_STATS, QTY, SKU, STOCK,
)

# сколько дней истории минимум нужно, чтобы дообучение имело смысл
MIN_ROWS_FOR_FINE_TUNE = 150


def _history_days(g: pd.DataFrame) -> int:
    return int((g[DATE].max() - g[DATE].min()).days) + 1


def stats_forecast(g: pd.DataFrame, horizon: int) -> tuple[float, float]:
    """
    Статистический фолбэк для коротких рядов.

    Медиана — среднедневное за доступное окно с поправкой на профиль дня
    недели; верхняя граница — 90-й перцентиль дневных продаж, пересчитанный
    на горизонт. Никакого нормального распределения не предполагаем.
    """
    g = g.sort_values(DATE)
    qty = g[QTY].astype(float)
    if FLAG_CENSORED in g.columns:
        qty = qty.where(~g[FLAG_CENSORED].astype(bool), np.nan)
    qty = qty.dropna()
    if qty.empty:
        return 0.0, 0.0

    window = min(len(qty), 56)
    recent = qty.tail(window)
    daily_med = float(recent.mean())
    daily_hi = float(np.nanpercentile(recent, 90)) if len(recent) >= 7 else daily_med * 1.5

    q50 = daily_med * horizon
    # разброс суммы за H дней растёт как sqrt(H), а не линейно
    spread = (daily_hi - daily_med) * np.sqrt(horizon)
    q90 = q50 + max(spread, 0.0)
    return q50, q90


def minmax_levels(g: pd.DataFrame, horizon: int) -> tuple[float, float]:
    """
    Расчёт уровней для товара с НЕПРОГНОЗИРУЕМЫМ спросом (класс LUMPY).

    Здесь сознательно НЕ применяется ни модель, ни квантиль высокого
    порядка. Для редкого хаотичного спроса 90-й перцентиль — это почти
    всегда редкий крупный всплеск, и закладывать его в запас означает
    заморозить деньги ради события, которое может не повториться.

    Вместо прогноза используется консервативное правило min/max:
    медианная потребность считается по фактическому среднему, а верхняя
    граница ограничена типичным (медианным) размером одной продажи,
    а не максимальным. Такой товар закупается «под заказ», и задача
    системы — не угадать спрос, а не дать остатку уйти в ноль незаметно.
    """
    g = g.sort_values(DATE)
    qty = g[QTY].astype(float)
    if FLAG_CENSORED in g.columns:
        qty = qty.where(~g[FLAG_CENSORED].astype(bool), np.nan)
    qty = qty.dropna()
    if qty.empty:
        return 0.0, 0.0

    q50 = float(qty.mean()) * horizon

    nonzero = qty[qty > 0]
    if nonzero.empty:
        return 0.0, 0.0

    # верхняя граница = средний спрос + ОДНА типичная партия,
    # а не редкий пиковый всплеск
    typical_order = float(nonzero.median())
    q90 = q50 + typical_order
    return q50, q90


def forecast(
    df: pd.DataFrame,
    horizon: int = 30,
    model: QuantileModel | None = None,
    model_dir: str | Path | None = "models/global",
    allow_fine_tune: bool = True,
    respect_demand_class: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    df — очищенный канонический DataFrame.
    Возвращает: sku, q50, q90, mode, mode_label, history_days, stock, daily_q50.
    """
    if model is None and model_dir is not None:
        p = Path(model_dir)
        if (p / "meta.json").exists():
            try:
                model = QuantileModel.load(p)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"Модель не загрузилась ({e}), работаем в режиме STATS")
                model = None

    groups = {sku: g.sort_values(DATE).reset_index(drop=True)
              for sku, g in df.groupby(SKU, sort=False)}

    # --- раскладываем товары по режимам ------------------------------------
    # Порядок проверок важен: сначала смотрим на ХАРАКТЕР спроса, и только
    # потом на длину истории. Товар может иметь три года истории и всё равно
    # быть непрогнозируемым (продаётся раз в месяц случайными партиями) —
    # для такого ML-прогноз даст раздутый страховой запас и заморозит деньги.
    demand_classes: dict[str, str] = {}
    modes: dict[str, str] = {}
    for sku, g in groups.items():
        cls, _adi, _cv2, _nz = classify_series(
            g[QTY], g[FLAG_CENSORED] if FLAG_CENSORED in g.columns else None)
        demand_classes[sku] = cls

        if respect_demand_class and cls in NOT_FORECASTABLE:
            modes[sku] = MODE_MINMAX
            continue

        m = pick_mode(_history_days(g))
        if respect_demand_class and cls in STATS_ONLY and m in (
                MODE_GLOBAL, MODE_FINE_TUNE):
            # прерывистый спрос: история есть, но ML на ней ненадёжен
            m = MODE_STATS
        if m in (MODE_GLOBAL, MODE_FINE_TUNE) and model is None:
            m = MODE_STATS
        if m == MODE_FINE_TUNE and not allow_fine_tune:
            m = MODE_GLOBAL
        modes[sku] = m

    # --- дообучение: одна общая дообученная модель на все «длинные» SKU ----
    tuned: QuantileModel | None = None
    ft_skus = [s for s, m in modes.items() if m == MODE_FINE_TUNE]
    if ft_skus and model is not None:
        sub = df[df[SKU].isin(ft_skus)]
        ds = build_training_set(sub, [horizon])
        if len(ds) >= MIN_ROWS_FOR_FINE_TUNE:
            feats = [c for c in FEATURE_COLUMNS if c in ds.columns]
            try:
                tuned = model.fine_tune(ds[feats].fillna(0.0), ds[TARGET_COL])
                if verbose:
                    print(f"Дообучение выполнено на {len(ds):,} строках "
                          f"({len(ft_skus)} товаров)")
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"Дообучение не удалось ({e}), используем глобальную модель")
        else:
            for s in ft_skus:
                modes[s] = MODE_GLOBAL

    # --- инференс -----------------------------------------------------------
    ml_skus = [s for s, m in modes.items() if m in (MODE_GLOBAL, MODE_FINE_TUNE)]
    ml_pred: dict[str, tuple[float, float]] = {}

    if ml_skus and model is not None:
        rows = inference_rows(df[df[SKU].isin(ml_skus)], horizon)
        rows = rows[rows[BASE_COL].notna() & (rows[BASE_COL] > 0)]
        if len(rows):
            feats = [c for c in (model.features or FEATURE_COLUMNS) if c in rows.columns]
            X = rows[feats].fillna(0.0)
            scale = rows[BASE_COL].to_numpy() * horizon

            for use_tuned in (True, False):
                m = tuned if use_tuned else model
                if m is None:
                    continue
                target = [s for s in rows[SKU]
                          if (modes[s] == MODE_FINE_TUNE) == use_tuned]
                mask = rows[SKU].isin(target).to_numpy()
                if not mask.any():
                    continue
                lo, hi = m.predict(X[mask])
                for sku, a, b, sc in zip(rows.loc[mask, SKU], lo, hi, scale[mask]):
                    ml_pred[sku] = (float(a * sc), float(b * sc))

        # SKU, для которых признаки не посчитались, откатываем на статистику
        for s in ml_skus:
            if s not in ml_pred:
                modes[s] = MODE_STATS

    # --- сборка результата --------------------------------------------------
    out = []
    for sku, g in groups.items():
        mode = modes[sku]
        hist_days = _history_days(g)
        stock = float(g[STOCK].iloc[-1]) if STOCK in g.columns and pd.notna(
            g[STOCK].iloc[-1]) else np.nan

        if mode == MODE_REJECT:
            q50 = q90 = np.nan
        elif mode == MODE_MINMAX:
            q50, q90 = minmax_levels(g, horizon)
        elif mode == MODE_STATS:
            q50, q90 = stats_forecast(g, horizon)
        else:
            q50, q90 = ml_pred[sku]

        cls = demand_classes.get(sku, "")
        out.append({
            "sku": sku,
            "q50": round(q50, 1) if np.isfinite(q50) else np.nan,
            "q90": round(q90, 1) if np.isfinite(q90) else np.nan,
            "daily_q50": round(q50 / horizon, 2) if np.isfinite(q50) else np.nan,
            "mode": mode,
            "mode_label": MODE_LABELS_RU[mode],
            "demand_class": cls,
            "demand_class_label": CLASS_LABELS_RU.get(cls, ""),
            "recommendation": STRATEGY_RU.get(cls, ""),
            "history_days": hist_days,
            "stock": stock,
            "horizon": horizon,
        })

    return pd.DataFrame(out).sort_values("sku").reset_index(drop=True)


def forecast_curve(df: pd.DataFrame, sku: str, horizon: int,
                   model: QuantileModel | None = None,
                   model_dir: str | Path | None = "models/global") -> pd.DataFrame:
    """
    Дневная раскладка прогноза для графика.

    Модель предсказывает СУММУ за горизонт (так надёжнее), но пользователю
    нужен график по дням. Раскладываем сумму по профилю дня недели,
    посчитанному на истории этого товара.
    """
    g = df[df[SKU] == sku].sort_values(DATE)
    if g.empty:
        return pd.DataFrame()

    fc = forecast(g, horizon=horizon, model=model, model_dir=model_dir)
    if fc.empty or not np.isfinite(fc.iloc[0]["q50"]):
        return pd.DataFrame()
    q50, q90 = float(fc.iloc[0]["q50"]), float(fc.iloc[0]["q90"])

    qty = g[QTY].astype(float)
    if FLAG_CENSORED in g.columns:
        qty = qty.where(~g[FLAG_CENSORED].astype(bool), np.nan)
    prof = (pd.DataFrame({"dow": g[DATE].dt.dayofweek, "q": qty})
            .dropna().groupby("dow")["q"].mean())
    if prof.empty or prof.sum() <= 0:
        weights = pd.Series(1.0, index=range(7))
    else:
        weights = prof / prof.mean()

    start = g[DATE].max() + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=horizon, freq="D")
    w = np.array([weights.get(d.dayofweek, 1.0) for d in dates], dtype=float)
    w = w / w.sum()

    return pd.DataFrame({
        "date": dates,
        "q50": np.round(q50 * w, 2),
        "q90": np.round(q90 * w, 2),
    })
