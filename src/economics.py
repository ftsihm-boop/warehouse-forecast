"""
Экономика, выведенная ИЗ ДАННЫХ, и автоматический подбор страхового запаса.

ДВЕ ЗАДАЧИ ЭТОГО МОДУЛЯ
-----------------------

1. НЕ СПРАШИВАТЬ У ПОЛЬЗОВАТЕЛЯ ТО, ЧТО УЖЕ ЕСТЬ В ЕГО ФАЙЛЕ.
   Если в выгрузке есть цена продажи и закупочная цена — маржа, стоимость
   хранения и упущенная прибыль считаются по реальным цифрам магазина,
   а не по усреднённым допущениям. Каждый товар получает свою экономику:
   у молока маржа одна, у сигарет совсем другая.

2. НЕ ЗАСТАВЛЯТЬ ПОДБИРАТЬ ПАРАМЕТР ВРУЧНУЮ.
   Сколько держать страхового запаса — не вопрос вкуса, у него есть
   точный ответ, и он выводится из экономики товара.

МОДЕЛЬ ГАЗЕТЧИКА (newsvendor)
-----------------------------
Классическая задача: продавец газет каждое утро решает, сколько закупить.
Закупит мало — потеряет прибыль с непроданных клиентов. Закупит много —
останется с пачкой вчерашних газет.

У ошибки в каждую сторону своя цена:

    Cu (underage) — цена нехватки: упущенная прибыль с единицы
                    Cu = цена продажи - закупочная цена

    Co (overage)  — цена излишка: сколько стоит продержать
                    лишнюю единицу на складе до следующего заказа
                    Co = закупочная цена × ставка хранения × дни / 365

Оптимальный уровень сервиса — это отношение, которое уравнивает риски:

    critical ratio = Cu / (Cu + Co)

Смысл: если упущенная прибыль в 20 раз дороже хранения, надо закрывать
спрос с вероятностью 20/21 ≈ 95%. Если наоборот (скоропорт, дорогой
склад, копеечная маржа) — оптимум может опуститься до 60%, и держать
большой запас будет убыточно.

Дальше critical ratio переводится в множитель страхового запаса: модель
даёт квантили 0.5 и 0.9, а нам нужен произвольный — пересчитываем через
нормальную аппроксимацию хвоста.

ЧТО ЭТО ДАЁТ НА ПРАКТИКЕ
------------------------
Товар с высокой маржой и дешёвым хранением автоматически получит больший
страховой запас, товар с низкой маржой — меньший. Никаких ручных настроек:
система смотрит на цифры и решает сама, причём по каждому товару отдельно.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .schema import COST, PRICE, QTY, SKU

# --- значения по умолчанию ----------------------------------------------------
# Используются, только если в файле пользователя нет цен. Каждое такое
# допущение честно помечается в отчёте как оценка, а не факт.
DEFAULT_MARGIN = 0.25            # наценка, доля от цены продажи
DEFAULT_HOLDING_RATE = 0.35      # стоимость хранения, % от стоимости в год
DEFAULT_ORDER_COST = 300.0       # стоимость размещения одного заказа, ₽
DEFAULT_UNIT_COST = 60.0         # закупочная цена, если цен нет вообще

# Квантиль, на который обучена модель (см. schema.QUANTILE_HIGH).
# От него пересчитывается любой другой уровень сервиса.
TRAINED_QUANTILE = 0.9

# Разумные границы множителя: без ограничения при экстремальной экономике
# формула может потребовать запас на годы вперёд или вовсе его обнулить.
MIN_SERVICE_FACTOR = 0.3
MAX_SERVICE_FACTOR = 2.5

# Граничные уровни сервиса — тоже страховка от вырожденных случаев
MIN_SERVICE_LEVEL = 0.55
MAX_SERVICE_LEVEL = 0.995


@dataclass
class SkuEconomics:
    """Экономика одного товара."""

    sku: str
    unit_cost: float          # закупочная цена
    unit_price: float         # цена продажи
    margin_abs: float         # прибыль с единицы, ₽
    margin_rate: float        # маржа, доля от цены продажи
    holding_per_unit_day: float
    service_level: float      # оптимальный уровень сервиса (critical ratio)
    service_factor: float     # множитель страхового запаса
    source: str               # откуда взяты цены: "данные" / "оценка"

    def to_dict(self) -> dict:
        return asdict(self)


def _z_score(p: float) -> float:
    """
    Квантиль стандартного нормального распределения.

    Аппроксимация Acklam — точности более чем достаточно, а главное,
    не тянет scipy ради одной функции.
    """
    p = min(max(p, 1e-6), 1 - 1e-6)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def optimal_service_level(margin_abs: float, holding_cost_period: float) -> float:
    """
    Модель газетчика: оптимальный уровень сервиса.

        critical ratio = Cu / (Cu + Co)

    Cu — упущенная прибыль с единицы, Co — стоимость хранения лишней
    единицы за период покрытия.
    """
    cu = max(float(margin_abs), 0.0)
    co = max(float(holding_cost_period), 1e-9)
    if cu <= 0:
        # товар не приносит прибыли — держать запас ради него незачем
        return MIN_SERVICE_LEVEL
    ratio = cu / (cu + co)
    return float(min(max(ratio, MIN_SERVICE_LEVEL), MAX_SERVICE_LEVEL))


def service_level_to_factor(service_level: float) -> float:
    """
    Переводит уровень сервиса в множитель страхового запаса.

    Модель обучена на квантиле 0.9, то есть её (q90 - q50) — это запас
    под уровень сервиса 90%, что соответствует z = 1.2816.
    Для другого уровня нужен свой z, отсюда и множитель.
    """
    z_target = _z_score(service_level)
    z_trained = _z_score(TRAINED_QUANTILE)
    factor = z_target / z_trained if z_trained else 1.0
    return float(min(max(factor, MIN_SERVICE_FACTOR), MAX_SERVICE_FACTOR))


def derive_sku_economics(
    df: pd.DataFrame,
    holding_rate_year: float = DEFAULT_HOLDING_RATE,
    cover_days: int = 14,
    default_margin: float = DEFAULT_MARGIN,
) -> pd.DataFrame:
    """
    Выводит экономику по каждому товару из его данных.

    Логика по убыванию точности:
      1. есть цена продажи и закупки  -> всё считается по факту
      2. есть только цена продажи     -> закупка оценивается через наценку
      3. есть только закупочная цена  -> продажа оценивается через наценку
      4. цен нет                      -> значения по умолчанию
    """
    if df.empty:
        return pd.DataFrame()

    has_price = PRICE in df.columns and df[PRICE].notna().any()
    has_cost = COST in df.columns and df[COST].notna().any()

    rows: list[dict] = []
    for sku, g in df.groupby(SKU, sort=False):
        # взвешиваем по количеству: цена в дни крупных продаж важнее
        w = g[QTY].astype(float).clip(lower=0)
        if w.sum() <= 0:
            w = pd.Series(1.0, index=g.index)

        def wmean(col: str) -> float:
            if col not in g.columns:
                return float("nan")
            vals = g[col].astype(float)
            mask = vals.notna() & (vals > 0)
            if not mask.any():
                return float("nan")
            return float(np.average(vals[mask], weights=w[mask]))

        price = wmean(PRICE) if has_price else float("nan")
        cost = wmean(COST) if has_cost else float("nan")

        if np.isfinite(price) and np.isfinite(cost) and price > cost > 0:
            source = "данные"
        elif np.isfinite(price) and price > 0:
            cost = price * (1 - default_margin)
            source = "цена продажи из данных, закупка оценена"
        elif np.isfinite(cost) and cost > 0:
            price = cost / (1 - default_margin)
            source = "закупка из данных, цена продажи оценена"
        else:
            cost = DEFAULT_UNIT_COST
            price = cost / (1 - default_margin)
            source = "оценка (цен в файле нет)"

        margin_abs = max(price - cost, 0.0)
        margin_rate = margin_abs / price if price > 0 else 0.0
        holding_day = cost * holding_rate_year / 365
        holding_period = holding_day * cover_days

        level = optimal_service_level(margin_abs, holding_period)
        rows.append(SkuEconomics(
            sku=sku,
            unit_cost=round(cost, 2),
            unit_price=round(price, 2),
            margin_abs=round(margin_abs, 2),
            margin_rate=round(margin_rate, 3),
            holding_per_unit_day=round(holding_day, 4),
            service_level=round(level, 3),
            service_factor=round(service_level_to_factor(level), 3),
            source=source,
        ).to_dict())

    return pd.DataFrame(rows)


def portfolio_economics(df: pd.DataFrame, **kwargs) -> dict:
    """Сводная экономика по всему загруженному файлу — для отчёта."""
    eco = derive_sku_economics(df, **kwargs)
    if eco.empty:
        return {}

    qty = df.groupby(SKU)[QTY].sum()
    eco = eco.set_index("sku")
    weights = qty.reindex(eco.index).fillna(0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=eco.index)

    from_data = int((eco["source"] == "данные").sum())
    return {
        "Товаров": len(eco),
        "Цены взяты из файла": from_data,
        "Цены оценены": len(eco) - from_data,
        "Средняя закупочная цена, ₽": round(
            float(np.average(eco["unit_cost"], weights=weights)), 2),
        "Средняя цена продажи, ₽": round(
            float(np.average(eco["unit_price"], weights=weights)), 2),
        "Средняя маржа, %": round(
            float(np.average(eco["margin_rate"], weights=weights)) * 100, 1),
        "Средний уровень сервиса, %": round(
            float(np.average(eco["service_level"], weights=weights)) * 100, 1),
        "Средний множитель запаса": round(
            float(np.average(eco["service_factor"], weights=weights)), 2),
    }


def autotune_service_factor(
    df: pd.DataFrame,
    model=None,
    econ=None,
    max_skus: int = 8,
    test_days: int = 60,
    lead_time: int = 7,
    review_period: int = 7,
    verbose: bool = False,
) -> dict:
    """
    Автоматический подбор множителя страхового запаса.

    ПОЧЕМУ НЕ ЧИСТАЯ ФОРМУЛА.
    Модель газетчика даёт теоретический оптимум, предполагая нормальное
    распределение спроса с бесконечными хвостами. В жизни спрос ограничен:
    начиная с некоторого уровня запаса дефицит исчезает полностью, и
    дальнейшее наращивание — чистые потери на хранении. Формула этого
    не видит и систематически завышает ответ.

    ПОЧЕМУ НЕ ЧИСТЫЙ ПЕРЕБОР.
    Перебор вслепую по всей сетке — это долго и не объясняет, почему
    выбрано именно это значение.

    Поэтому: теория задаёт точку отсчёта и границы поиска, а короткий
    бэктест на реальной истории пользователя уточняет её. Проверяется
    несколько значений вокруг теоретического оптимума на подвыборке
    самых оборотистых товаров — этого достаточно, чтобы поймать
    оптимум, и быстро.
    """
    from .backtest import Economics, backtest  # локально: избегаем цикла

    econ = econ or Economics()
    if df.empty:
        return {"service_factor": 1.0, "source": "нет данных"}

    # --- теоретическая точка отсчёта ---------------------------------------
    eco = derive_sku_economics(df, holding_rate_year=econ.holding_rate_year,
                               cover_days=lead_time + review_period,
                               default_margin=econ.margin)
    if eco.empty:
        theoretical = 1.0
    else:
        qty = df.groupby(SKU)[QTY].sum().reindex(eco["sku"]).fillna(0.0)
        w = qty.to_numpy() if qty.sum() > 0 else np.ones(len(eco))
        theoretical = float(np.average(eco["service_factor"], weights=w))

    # --- сетка вокруг теоретической точки ----------------------------------
    grid = sorted({round(max(MIN_SERVICE_FACTOR, min(MAX_SERVICE_FACTOR, v)), 2)
                   for v in (theoretical * m for m in (0.3, 0.45, 0.6, 0.8, 1.0, 1.25))})

    # --- подвыборка: самые оборотистые товары с достаточной историей -------
    per_sku = df.groupby(SKU).agg(total=(QTY, "sum"), n=(QTY, "size"))
    eligible = per_sku[per_sku["n"] >= test_days + 90]
    if eligible.empty:
        return {"service_factor": round(theoretical, 2),
                "theoretical": round(theoretical, 2),
                "source": "теория (истории мало для проверки)",
                "grid": []}

    top = eligible.sort_values("total", ascending=False).head(max_skus).index
    sample = df[df[SKU].isin(top)]

    # --- короткий бэктест по сетке -----------------------------------------
    results = []
    for factor in grid:
        _table, summary = backtest(
            sample, model, econ=econ, test_days=test_days, horizon=30,
            lead_time=lead_time, review_period=review_period,
            service_factor=factor, only_forecastable=True)
        if summary:
            results.append((factor, summary["Эффект в месяц, ₽"]))
            if verbose:
                print(f"    × {factor:.2f} -> {summary['Эффект в месяц, ₽']:>8} ₽/мес")

    if not results:
        return {"service_factor": round(theoretical, 2),
                "theoretical": round(theoretical, 2),
                "source": "теория (бэктест не дал результата)",
                "grid": []}

    best_factor, best_effect = max(results, key=lambda r: r[1])
    return {
        "service_factor": round(best_factor, 2),
        "theoretical": round(theoretical, 2),
        "best_effect_per_month": best_effect,
        "source": "теория + проверка на ваших данных",
        "grid": [{"factor": f, "effect": e} for f, e in results],
        "skus_tested": int(len(top)),
        "test_days": test_days,
    }


def summary_text(df: pd.DataFrame, **kwargs) -> str:
    """Человекочитаемая сводка для консоли и отчёта."""
    eco = derive_sku_economics(df, **kwargs)
    if eco.empty:
        return "Нет данных для расчёта экономики."

    stats = portfolio_economics(df, **kwargs)
    lines = [
        "ЭКОНОМИКА ПО ДАННЫМ ПОЛЬЗОВАТЕЛЯ",
        "(уровень сервиса подобран автоматически — модель газетчика)",
        "=" * 64,
    ]
    for k, v in stats.items():
        lines.append(f"  {k:<32} {v}")

    if stats.get("Цены оценены", 0):
        lines += [
            "-" * 64,
            f"  Для {stats['Цены оценены']} товар(ов) цены в файле не нашлись,",
            "  экономика по ним рассчитана оценочно. Добавьте в выгрузку",
            "  колонки «Цена продажи» и «Цена закупки» — расчёт станет точным.",
        ]

    # самые показательные примеры: где автонастройка дала разные ответы
    spread = eco.sort_values("service_factor")
    if len(spread) >= 2:
        lo, hi = spread.iloc[0], spread.iloc[-1]
        if abs(hi["service_factor"] - lo["service_factor"]) > 0.05:
            lines += [
                "-" * 64,
                "  Автонастройка запаса различается по товарам:",
                f"    · {str(lo['sku'])[:40]}",
                f"      маржа {lo['margin_rate'] * 100:.0f}% -> "
                f"сервис {lo['service_level'] * 100:.0f}%, "
                f"запас × {lo['service_factor']:.2f}",
                f"    · {str(hi['sku'])[:40]}",
                f"      маржа {hi['margin_rate'] * 100:.0f}% -> "
                f"сервис {hi['service_level'] * 100:.0f}%, "
                f"запас × {hi['service_factor']:.2f}",
            ]

    return "\n".join(lines)
