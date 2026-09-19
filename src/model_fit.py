"""
Проверка пригодности модели к загруженным данным и автоматическая адаптация.

ЗАЧЕМ
-----
Глобальная модель обучается один раз на каком-то наборе товаров.
Потом пользователь загружает свой файл — и там может быть совсем другой
ассортимент: модель учили на дисках и играх, а пришли молоко и хлеб.

Внешне всё выглядит рабочим: прогноз считается, таблица заполняется,
цифры красивые. Но прогноз систематически смещён, и обнаруживается это
только в экономическом расчёте, где эффект внезапно оказывается нулевым.

Модуль решает это так: перед тем как доверять модели, система проверяет
её на данных пользователя — берёт кусок недавней истории, делает прогноз
и сравнивает с тем, что произошло на самом деле. Если прогноз смещён,
модель дообучается на этих данных, и проверка повторяется.

КАК ИЗМЕРЯЕТСЯ ПРИГОДНОСТЬ
--------------------------
Две метрики, и обе важны по-разному:

    WAPE     — насколько велика ошибка вообще
    СМЕЩЕНИЕ — в какую сторону она систематическая

Для управления запасами смещение опаснее. Модель, которая ошибается
случайно в обе стороны, приводит к нормальному страховому запасу.
Модель, которая стабильно занижает спрос на 20%, приводит к дефициту —
и никакой страховой запас это не лечит, его просто приходится раздувать,
съедая всю экономию.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .evaluate import bias, wape
from .features import (
    BASE_COL, FEATURE_COLUMNS, TARGET_ABS, TARGET_COL, build_training_set,
    inference_rows,
)
from .model import QuantileModel
from .schema import DATE, QTY, SKU

# Пороги пригодности. Подобраны из практики: систематическое смещение
# больше 12% уже заметно бьёт по расчёту запаса, а WAPE выше 35%
# означает, что прогноз мало чем лучше среднего.
MAX_ACCEPTABLE_BIAS = 12.0
MAX_ACCEPTABLE_WAPE = 35.0
# Доверительный интервал обучен на квантиле 0.9, значит факт должен
# укладываться под верхнюю границу примерно в 90 % случаев. Допускаем
# просадку до 80 %: ниже этого страхового запаса уже не хватает.
MIN_Q90_COVERAGE = 0.80

# Сколько дней отрезать для проверки и сколько минимум нужно истории
VALIDATION_DAYS = 45
MIN_HISTORY_FOR_CHECK = 120
MIN_ROWS_FOR_TUNING = 200


@dataclass
class FitReport:
    """Насколько модель подходит к данным пользователя."""

    checked: bool                 # проверку вообще удалось провести
    suitable: bool                # модель пригодна как есть
    wape: float | None
    bias: float | None
    coverage: float | None       # доля фактов под верхней границей прогноза
    samples: int
    action: str                   # что сделали: none / fine_tuned / retrained
    message: str
    wape_after: float | None = None
    bias_after: float | None = None
    coverage_after: float | None = None
    improvement_pct: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _score(model: QuantileModel, dataset: pd.DataFrame
           ) -> tuple[float, float, float, int]:
    """
    Оценивает модель на готовой выборке признаков.

    Возвращает (WAPE, смещение, покрытие q90, число точек).

    ПОКРЫТИЕ q90 — отдельная и очень важная метрика. Модель выдаёт два
    числа: медианный прогноз и верхнюю границу. Разница между ними и есть
    страховой запас. Если фактический спрос укладывается под верхнюю
    границу в 90 % случаев — интервал откалиброван правильно. Если только
    в 70 %, то интервал слишком узкий: запаса систематически не хватает,
    и компенсировать это приходится раздутым коэффициентом, который
    съедает всю экономию. Точность медианы при этом может быть прекрасной,
    поэтому одного WAPE недостаточно.
    """
    feats = [c for c in (model.features or FEATURE_COLUMNS) if c in dataset.columns]
    X = dataset[feats].fillna(0.0)
    q50, q90 = model.predict(X)

    scale = dataset[BASE_COL].to_numpy() * dataset["horizon"].to_numpy()
    y_true = dataset[TARGET_ABS].to_numpy()
    y_pred = q50 * scale
    y_high = q90 * scale

    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 20:
        return float("nan"), float("nan"), float("nan"), int(ok.sum())

    coverage = float((y_true[ok] <= y_high[ok]).mean())
    return (wape(y_true[ok], y_pred[ok]), bias(y_true[ok], y_pred[ok]),
            coverage, int(ok.sum()))


def check_fit(df: pd.DataFrame, model: QuantileModel | None,
              horizon: int = 30) -> FitReport:
    """
    Оценивает, подходит ли модель к этим данным.

    Проверка честная: последние VALIDATION_DAYS дней модель не видит
    при обучении и не должна была видеть — она просто прогнозирует,
    а мы сравниваем с фактом.
    """
    if model is None:
        return FitReport(False, False, None, None, None, 0, "none",
                         "Обученной модели нет — прогноз считается по статистике.")

    if df is None or df.empty:
        return FitReport(False, False, None, None, None, 0, "none", "Нет данных.")

    span = (df[DATE].max() - df[DATE].min()).days
    if span < MIN_HISTORY_FOR_CHECK:
        return FitReport(
            False, True, None, None, None, 0, "none",
            f"Истории мало ({span} дн.) — проверить модель на ней нельзя, "
            "работаем как есть.")

    cutoff = df[DATE].max() - pd.Timedelta(days=VALIDATION_DAYS)
    recent = df[df[DATE] > cutoff]
    if recent.empty:
        return FitReport(False, True, None, None, None, 0, "none",
                         "Не удалось выделить проверочный период.")

    ds = build_training_set(df, [horizon])
    if ds.empty:
        return FitReport(False, True, None, None, None, 0, "none",
                         "Не удалось собрать проверочную выборку.")

    holdout = ds[ds[DATE] > cutoff]
    if len(holdout) < 20:
        holdout = ds.tail(max(20, len(ds) // 5))

    w, b, cov, n = _score(model, holdout)
    if not np.isfinite(w):
        return FitReport(False, True, None, None, None, n, "none",
                         "Проверочных точек слишком мало для оценки.")

    suitable = (abs(b) <= MAX_ACCEPTABLE_BIAS
                and w <= MAX_ACCEPTABLE_WAPE
                and (not np.isfinite(cov) or cov >= MIN_Q90_COVERAGE))
    if suitable:
        msg = (f"Модель проверена на ваших данных и подходит: "
               f"ошибка {w:.1f} %, смещение {b:+.1f} %, "
               f"интервал покрывает {cov * 100:.0f} % фактов.")
    else:
        why = []
        if abs(b) > MAX_ACCEPTABLE_BIAS:
            why.append("прогноз систематически "
                       + ("занижен" if b < 0 else "завышен")
                       + f" на {abs(b):.0f} %")
        if w > MAX_ACCEPTABLE_WAPE:
            why.append(f"ошибка прогноза велика ({w:.0f} %)")
        if np.isfinite(cov) and cov < MIN_Q90_COVERAGE:
            why.append(f"доверительный интервал слишком узкий "
                       f"(покрывает {cov * 100:.0f} % фактов вместо 90 %) — "
                       "страхового запаса будет не хватать")
        msg = ("Модель плохо подходит к этим данным: " + "; ".join(why) + ".")

    return FitReport(True, suitable, round(w, 2), round(b, 2),
                     round(cov, 3) if np.isfinite(cov) else None, n, "none", msg)


def adapt_to_data(
    df: pd.DataFrame,
    model: QuantileModel | None,
    horizon: int = 30,
    force: bool = False,
) -> tuple[QuantileModel | None, FitReport]:
    """
    Проверяет модель на данных пользователя и, если нужно, дообучает.

    Возвращает модель, которой можно доверять на этих данных, и отчёт
    о том, что было сделано. Если дообучение не помогло — возвращается
    исходная модель: лучше предсказуемое поведение, чем случайное.
    """
    report = check_fit(df, model, horizon)
    if model is None:
        return None, report
    if report.suitable and not force:
        return model, report
    if not report.checked and not force:
        return model, report

    # --- дообучаем на данных пользователя ----------------------------------
    cutoff = df[DATE].max() - pd.Timedelta(days=VALIDATION_DAYS)
    train_part = df[df[DATE] <= cutoff]

    ds_train = build_training_set(train_part, [horizon])
    if len(ds_train) < MIN_ROWS_FOR_TUNING:
        report.message += (" Дообучить не получилось: для этого нужно больше "
                           "истории.")
        return model, report

    feats = [c for c in FEATURE_COLUMNS if c in ds_train.columns]
    try:
        tuned = model.fine_tune(ds_train[feats].fillna(0.0), ds_train[TARGET_COL])
    except Exception as e:  # noqa: BLE001
        report.message += f" Дообучение не удалось ({e})."
        return model, report

    # --- проверяем, что стало лучше ----------------------------------------
    ds_full = build_training_set(df, [horizon])
    holdout = ds_full[ds_full[DATE] > cutoff]
    if len(holdout) < 20:
        holdout = ds_full.tail(max(20, len(ds_full) // 5))

    w_after, b_after, cov_after, _ = _score(tuned, holdout)
    if not np.isfinite(w_after):
        return model, report

    # «Плохость» модели складывается из трёх частей: величины ошибки,
    # систематического смещения и недобора покрытия интервала. Последнее
    # весит больше остального — именно оно определяет, хватит ли запаса.
    def badness(w, b, cov):
        gap = max(0.0, MIN_Q90_COVERAGE - (cov if np.isfinite(cov) else 1.0))
        return abs(b) + w + gap * 200

    before_bad = badness(report.wape or 0, report.bias or 0,
                         report.coverage if report.coverage is not None else 1.0)
    after_bad = badness(w_after, b_after, cov_after)

    if after_bad < before_bad * 0.98:
        improvement = (1 - after_bad / max(before_bad, 1e-9)) * 100
        report.action = "fine_tuned"
        report.wape_after = round(w_after, 2)
        report.bias_after = round(b_after, 2)
        report.coverage_after = round(cov_after, 3) if np.isfinite(cov_after) else None
        report.improvement_pct = round(improvement, 1)
        report.suitable = (abs(b_after) <= MAX_ACCEPTABLE_BIAS
                           and w_after <= MAX_ACCEPTABLE_WAPE
                           and (not np.isfinite(cov_after)
                                or cov_after >= MIN_Q90_COVERAGE))
        cov_txt = ""
        if report.coverage is not None and np.isfinite(cov_after):
            cov_txt = (f", покрытие интервала {report.coverage * 100:.0f} → "
                       f"{cov_after * 100:.0f} %")
        report.message = (
            f"Модель дообучена на ваших данных: ошибка "
            f"{report.wape:.1f} → {w_after:.1f} %, смещение "
            f"{report.bias:+.1f} → {b_after:+.1f} %" + cov_txt + ".")
        return tuned, report

    # дообучение не помогло — честно об этом говорим
    report.message += (" Дообучение на этих данных улучшения не дало — "
                       "возможно, истории слишком мало или спрос "
                       "слишком нерегулярный.")
    return model, report
