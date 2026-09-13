"""
Обёртка над квантильной регрессией.

Основной бэкенд — CatBoost. Если он не установлен, используется
HistGradientBoostingRegressor из sklearn: интерфейс и смысл те же,
качество чуть ниже. Это позволяет запустить весь пайплайн где угодно
и не блокировать разработку установкой зависимостей.

ПОЧЕМУ КВАНТИЛИ, А НЕ СРЕДНЕЕ.
Обычная регрессия даёт точечный прогноз, а для расчёта страхового запаса
нужен разброс. Классическая формула safety_stock = 1.65 * sigma * sqrt(L)
предполагает нормальное распределение спроса, которого у штучных товаров
нет и близко (много нулей, редкие всплески, асимметрия).

Мы вместо этого обучаем две модели — на квантиль 0.5 (медиана) и 0.9
(верхняя граница) — и берём страховой запас как их разность.
Распределение берётся из самих данных, а не из предположения.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:  # pragma: no cover - зависит от окружения
    from catboost import CatBoostRegressor, Pool
    HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    HAS_CATBOOST = False

from sklearn.ensemble import HistGradientBoostingRegressor

DEFAULT_PARAMS = dict(
    iterations=1200,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)

FINE_TUNE_PARAMS = dict(
    iterations=300,       # дообучение короткое: рискуем забыть глобальные паттерны
    learning_rate=0.03,
    verbose=False,
    allow_writing_files=False,
)


@dataclass
class QuantileModel:
    """Пара моделей: медиана и верхний квантиль."""

    alpha_low: float = 0.5
    alpha_high: float = 0.9
    features: list[str] | None = None
    backend: str = "catboost" if HAS_CATBOOST else "sklearn"
    model_low: object | None = None
    model_high: object | None = None

    # --- построение -------------------------------------------------------
    def _make(self, alpha: float, params: dict | None = None):
        p = {**DEFAULT_PARAMS, **(params or {})}
        if self.backend == "catboost":
            return CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}", **p)
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=alpha,
            max_iter=p["iterations"] // 4,
            learning_rate=p["learning_rate"],
            max_depth=p["depth"],
            l2_regularization=p["l2_leaf_reg"],
            random_state=p["random_seed"],
            early_stopping=False,
        )

    # --- обучение ---------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
            params: dict | None = None) -> "QuantileModel":
        self.features = list(X.columns)
        Xv = X.to_numpy(dtype=float)
        yv = np.asarray(y, dtype=float)

        for attr, alpha in (("model_low", self.alpha_low), ("model_high", self.alpha_high)):
            m = self._make(alpha, params)
            if self.backend == "catboost":
                es = None
                if eval_set is not None:
                    es = Pool(eval_set[0].to_numpy(dtype=float),
                              np.asarray(eval_set[1], dtype=float))
                m.fit(Xv, yv, eval_set=es,
                      early_stopping_rounds=100 if es is not None else None)
            else:
                m.fit(Xv, yv)
            setattr(self, attr, m)
        return self

    def fine_tune(self, X: pd.DataFrame, y: pd.Series) -> "QuantileModel":
        """
        Дообучение на данных пользователя (режим FINE_TUNE).
        CatBoost умеет продолжать обучение через init_model; sklearn — нет,
        поэтому там просто переобучаемся на объединённой выборке.
        """
        Xv = X[self.features].to_numpy(dtype=float)
        yv = np.asarray(y, dtype=float)

        if self.backend == "catboost" and self.model_low is not None:
            tuned = QuantileModel(self.alpha_low, self.alpha_high,
                                  list(self.features), self.backend)
            for attr, alpha in (("model_low", self.alpha_low),
                                ("model_high", self.alpha_high)):
                base = getattr(self, attr)
                m = CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}",
                                      **FINE_TUNE_PARAMS)
                m.fit(Xv, yv, init_model=base)
                setattr(tuned, attr, m)
            return tuned

        tuned = QuantileModel(self.alpha_low, self.alpha_high,
                              list(self.features), self.backend)
        tuned.fit(X[self.features], y)
        return tuned

    # --- предсказание -----------------------------------------------------
    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xv = X[self.features].to_numpy(dtype=float)
        low = np.asarray(self.model_low.predict(Xv), dtype=float)
        high = np.asarray(self.model_high.predict(Xv), dtype=float)
        low = np.clip(low, 0, None)
        # верхняя граница не может быть ниже медианы
        high = np.maximum(high, low)
        return low, high

    def feature_importance(self) -> pd.DataFrame | None:
        if self.backend != "catboost" or self.model_low is None:
            return None
        imp = self.model_low.get_feature_importance()
        return (pd.DataFrame({"feature": self.features, "importance": imp})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))

    # --- сохранение -------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = dict(alpha_low=self.alpha_low, alpha_high=self.alpha_high,
                    features=self.features, backend=self.backend)
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        if self.backend == "catboost":
            self.model_low.save_model(str(d / "model_q50.cbm"))
            self.model_high.save_model(str(d / "model_q90.cbm"))
        else:
            import pickle
            with open(d / "model_sklearn.pkl", "wb") as f:
                pickle.dump((self.model_low, self.model_high), f)

    @classmethod
    def load(cls, directory: str | Path) -> "QuantileModel":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        obj = cls(meta["alpha_low"], meta["alpha_high"],
                  meta["features"], meta["backend"])
        if meta["backend"] == "catboost":
            if not HAS_CATBOOST:
                raise RuntimeError(
                    "Модель сохранена в формате CatBoost, но библиотека не "
                    "установлена. Выполните: pip install catboost")
            obj.model_low = CatBoostRegressor().load_model(str(d / "model_q50.cbm"))
            obj.model_high = CatBoostRegressor().load_model(str(d / "model_q90.cbm"))
        else:
            import pickle
            with open(d / "model_sklearn.pkl", "rb") as f:
                obj.model_low, obj.model_high = pickle.load(f)
        return obj
