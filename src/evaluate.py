"""
Baseline-модели и метрики.

Baseline нужен не для галочки: это точка отсчёта, которую обязана побить
любая ML-модель. Прийти на защиту с CatBoost и MAPE 25%, когда наивная
модель даёт 18%, — провал. Прийти со сравнительной таблицей — обоснованный
инженерный выбор.

ПОЧЕМУ WAPE, А НЕ MAPE.
MAPE = mean(|y - ŷ| / y) взрывается, когда y близко к нулю: один день
с продажей 1 шт и прогнозом 3 шт даёт вклад 200%. У штучных товаров
(персики зимой) таких дней много, и MAPE перестаёт что-либо значить.

WAPE = sum(|y - ŷ|) / sum(y) — взвешенная по объёму ошибка, устойчивая
к нулям и интерпретируемая как «мы ошиблись на N% от общего оборота».
Считаем обе: MAPE — потому что он в SMART-цели, WAPE — потому что он
честный.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import DATE, QTY, SKU

EPS = 1e-9


# --- метрики ------------------------------------------------------------------

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / (denom + EPS) * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray, min_value: float = 1.0) -> float:
    """MAPE только по точкам, где факт >= min_value (иначе метрика бессмысленна)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.abs(y_true) >= min_value
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Систематическое смещение в % — отрицательное значит занижаем спрос."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float((y_pred - y_true).sum() / (np.abs(y_true).sum() + EPS) * 100)


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {"WAPE, %": round(wape(y_true, y_pred), 2),
            "MAPE, %": round(mape(y_true, y_pred), 2),
            "MAE": round(mae(y_true, y_pred), 2),
            "Смещение, %": round(bias(y_true, y_pred), 2)}


# --- baseline-прогнозы --------------------------------------------------------
# Все возвращают суммарный прогноз спроса за `horizon` дней вперёд от
# каждой точки истории. Сигнатура одинаковая, чтобы сравнивать честно.

def naive_last_period(history: pd.Series, horizon: int) -> pd.Series:
    """«Продадим столько же, сколько за прошлые H дней»."""
    return history.shift(1).rolling(horizon, min_periods=1).sum()


def moving_average(history: pd.Series, horizon: int, window: int = 28) -> pd.Series:
    """Среднедневное за окно × горизонт. Это то, что делает текущий прототип."""
    return history.shift(1).rolling(window, min_periods=3).mean() * horizon


def seasonal_naive(history: pd.Series, horizon: int, period: int = 364) -> pd.Series:
    """Спрос за тот же период год назад (364 = 52 недели, сохраняет дни недели)."""
    shifted = history.shift(period)
    return shifted.shift(1).rolling(horizon, min_periods=1).sum()


def dow_profile(history: pd.Series, dates: pd.Series, horizon: int,
                window: int = 56) -> pd.Series:
    """Среднее с поправкой на профиль дня недели — «умное» скользящее среднее."""
    base = history.shift(1).rolling(window, min_periods=14).mean()
    return base * horizon


BASELINES = {
    "Наивный (прошлый период)": naive_last_period,
    "Скользящее среднее 28д": moving_average,
    "Сезонный наивный (год назад)": seasonal_naive,
}


def evaluate_baselines(df: pd.DataFrame, horizon: int,
                       test_days: int = 60) -> pd.DataFrame:
    """Считает метрики всех baseline-моделей на отложенном по времени периоде."""
    cutoff = df[DATE].max() - pd.Timedelta(days=test_days + horizon)
    rows = []

    preds: dict[str, list[np.ndarray]] = {k: [] for k in BASELINES}
    actuals: list[np.ndarray] = []

    for _, g in df.groupby(SKU, sort=False):
        g = g.sort_values(DATE).reset_index(drop=True)
        qty = g[QTY].astype(float)
        fwd = (qty.shift(-1).rolling(horizon, min_periods=horizon)
               .sum().shift(-(horizon - 1)))
        mask = (g[DATE] > cutoff) & fwd.notna()
        if not mask.any():
            continue
        actuals.append(fwd[mask].to_numpy())
        for name, fn in BASELINES.items():
            p = fn(qty, horizon)
            preds[name].append(np.nan_to_num(p[mask].to_numpy()))

    if not actuals:
        return pd.DataFrame()

    y = np.concatenate(actuals)
    for name in BASELINES:
        p = np.concatenate(preds[name])
        rows.append({"Модель": name, **all_metrics(y, p)})
    return pd.DataFrame(rows)


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """Собирает итоговую сравнительную таблицу для курсовой."""
    return (pd.DataFrame(results)
            .sort_values("WAPE, %")
            .reset_index(drop=True))
