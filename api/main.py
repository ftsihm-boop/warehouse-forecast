"""
HTTP API для веб-интерфейса.

Запуск:
    uvicorn api.main:app --reload --port 8000

Эндпоинты:
    POST /upload          файл -> session_id + отчёт об очистке
    GET  /clean-report    полный отчёт + лог текстом
    GET  /forecast        прогноз по всем SKU на горизонт
    GET  /forecast/curve  дневная раскладка прогноза для графика
    GET  /order-plan      план закупки (главный экран менеджера)
    GET  /history         очищенная история для графика факта
    GET  /health          статус сервиса и загруженной модели

Сессии живут в памяти процесса. Для учебного прототипа этого достаточно;
для продакшена сюда встаёт Redis или таблица в PostgreSQL.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import CleaningReport, clean  # noqa: E402
from src.backtest import Economics  # noqa: E402
from src.demand_classes import CLASS_LABELS_RU, profile_all  # noqa: E402
from src.economics import (  # noqa: E402
    autotune_service_factor, derive_sku_economics, portfolio_economics,
)
from src.forecast import forecast, forecast_curve  # noqa: E402
from src.ingest import IngestError, read_table  # noqa: E402
from src.inventory import build_order_plan  # noqa: E402
from src.model import QuantileModel  # noqa: E402
from src.schema import DATE, QTY, SKU, STOCK  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "global"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SESSION_TTL = timedelta(hours=4)
MAX_SESSIONS = 100

app = FastAPI(title="Управление складскими запасами", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@dataclass
class Session:
    df: pd.DataFrame
    report: CleaningReport
    created: datetime = field(default_factory=datetime.utcnow)


SESSIONS: dict[str, Session] = {}
MODEL: QuantileModel | None = None


@app.on_event("startup")
def _load_model() -> None:
    global MODEL
    if (MODEL_DIR / "meta.json").exists():
        try:
            MODEL = QuantileModel.load(MODEL_DIR)
            print(f"Модель загружена: {MODEL.backend}, {len(MODEL.features or [])} признаков")
        except Exception as e:  # noqa: BLE001
            print(f"Модель не загрузилась: {e}. Работаем в статистическом режиме.")
    else:
        print("Обученная модель не найдена — режим STATS. "
              "Обучите: python -m src.train --data <файл> --out models/global")


def _gc() -> None:
    now = datetime.utcnow()
    for sid in [s for s, v in SESSIONS.items() if now - v.created > SESSION_TTL]:
        SESSIONS.pop(sid, None)
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.pop(min(SESSIONS, key=lambda s: SESSIONS[s].created), None)


def _session(session_id: str) -> Session:
    s = SESSIONS.get(session_id)
    if s is None:
        raise HTTPException(404, "Сессия не найдена или истекла. Загрузите файл заново.")
    return s


def _jsonable(df: pd.DataFrame) -> list[dict]:
    return df.replace({np.nan: None}).to_dict("records")


# --- эндпоинты ----------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "backend": MODEL.backend if MODEL else None,
        "sessions": len(SESSIONS),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...),
                 fill_missing_dates: bool = Query(True),
                 winsorize: bool = Query(True)) -> dict:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Файл больше {MAX_UPLOAD_BYTES // 1024 // 1024} МБ.")
    if not raw:
        raise HTTPException(400, "Файл пустой.")

    try:
        ingested = read_table(raw, file.filename)
    except IngestError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Не удалось разобрать файл: {e}") from e

    df, report = clean(ingested, fill_missing_dates=fill_missing_dates,
                       winsorize=winsorize)
    if df.empty:
        raise HTTPException(422, "После очистки не осталось пригодных данных. "
                                 "Проверьте формат дат и количеств.")

    _gc()
    sid = uuid.uuid4().hex
    SESSIONS[sid] = Session(df=df, report=report)

    return {
        "session_id": sid,
        "filename": file.filename,
        "columns_detected": ingested.mapping,
        "report": report.to_dict(),
    }


@app.get("/clean-report")
def clean_report(session_id: str) -> dict:
    s = _session(session_id)
    return s.report.to_dict()


@app.get("/clean-report/text", response_class=PlainTextResponse)
def clean_report_text(session_id: str) -> str:
    return _session(session_id).report.to_text()


@app.get("/forecast")
def get_forecast(session_id: str, horizon: int = Query(30, ge=1, le=180)) -> dict:
    s = _session(session_id)
    fc = forecast(s.df, horizon=horizon, model=MODEL, model_dir=None)
    return {"horizon": horizon, "items": _jsonable(fc)}


@app.get("/forecast/curve")
def get_curve(session_id: str, sku: str, horizon: int = Query(30, ge=1, le=180)) -> dict:
    s = _session(session_id)
    curve = forecast_curve(s.df, sku, horizon, model=MODEL, model_dir=None)
    if curve.empty:
        raise HTTPException(404, f"Прогноз для «{sku}» построить не удалось.")
    curve = curve.assign(date=curve["date"].dt.strftime("%Y-%m-%d"))
    return {"sku": sku, "horizon": horizon, "points": _jsonable(curve)}


@app.get("/order-plan")
def order_plan(
    session_id: str,
    horizon: int = Query(30, ge=1, le=180),
    lead_time_days: int = Query(7, ge=1, le=90),
    review_period_days: int = Query(7, ge=1, le=90),
    order_cost: float | None = Query(None),
    holding_cost_per_unit_year: float | None = Query(None),
    min_order_qty: float = Query(0.0, ge=0),
    order_multiple: float = Query(1.0, gt=0),
    auto_economics: bool = Query(True),
) -> dict:
    s = _session(session_id)
    fc = forecast(s.df, horizon=horizon, model=MODEL, model_dir=None)

    # Страховой запас по каждому товару считается из его собственной
    # экономики: цены берутся из загруженного файла, если они там есть.
    factors = None
    eco_info: dict = {}
    if auto_economics:
        eco = derive_sku_economics(s.df, cover_days=lead_time_days + review_period_days)
        if not eco.empty:
            factors = dict(zip(eco["sku"], eco["service_factor"]))
            from_data = int((eco["source"] == "данные").sum())
            eco_info = {
                "prices_from_file": from_data,
                "prices_estimated": int(len(eco) - from_data),
                "avg_service_level": round(float(eco["service_level"].mean()), 3),
            }

    plan = build_order_plan(
        fc, horizon_days=horizon, lead_time_days=lead_time_days,
        review_period_days=review_period_days, order_cost=order_cost,
        holding_cost_per_unit_year=holding_cost_per_unit_year,
        min_order_qty=min_order_qty, order_multiple=order_multiple,
        service_factors=factors,
    )
    counts = plan["status"].value_counts().to_dict() if not plan.empty else {}
    return {"horizon": horizon, "lead_time_days": lead_time_days,
            "status_counts": counts, "economics": eco_info,
            "items": _jsonable(plan)}


@app.get("/history")
def history(session_id: str, sku: str | None = None,
            days: int | None = Query(None, ge=1)) -> dict:
    s = _session(session_id)
    df = s.df if sku is None else s.df[s.df[SKU] == sku]
    if df.empty:
        raise HTTPException(404, f"Товар «{sku}» не найден.")
    if days:
        cutoff = df[DATE].max() - pd.Timedelta(days=days)
        df = df[df[DATE] > cutoff]
    out = df[[DATE, SKU, QTY, STOCK]].copy()
    out[DATE] = out[DATE].dt.strftime("%Y-%m-%d")
    return {"rows": _jsonable(out)}


@app.get("/skus")
def skus(session_id: str) -> dict:
    return {"items": _session(session_id).report.per_sku}


@app.get("/economics")
def economics(session_id: str,
              lead_time_days: int = Query(7, ge=1, le=90),
              review_period_days: int = Query(7, ge=1, le=90),
              holding_rate_year: float = Query(0.35, gt=0, le=3.0),
              autotune: bool = Query(False)) -> dict:
    """
    Экономика по данным пользователя и автонастройка страхового запаса.

    Цены берутся из загруженного файла, если они там есть. Уровень
    сервиса по каждому товару считается из его собственной экономики
    (модель газетчика) — у товара с высокой маржой запас больше.

    autotune=true дополнительно прогоняет короткий бэктест, чтобы
    уточнить множитель на реальной истории. Это занимает несколько
    секунд, поэтому по умолчанию выключено.
    """
    s = _session(session_id)
    cover = lead_time_days + review_period_days
    eco = derive_sku_economics(s.df, holding_rate_year=holding_rate_year,
                               cover_days=cover)
    out = {
        "items": _jsonable(eco),
        "portfolio": portfolio_economics(s.df, holding_rate_year=holding_rate_year,
                                         cover_days=cover),
    }
    if autotune:
        econ = Economics.from_data(s.df, holding_rate_year=holding_rate_year)
        out["autotune"] = autotune_service_factor(
            s.df, MODEL, econ=econ, lead_time=lead_time_days,
            review_period=review_period_days)
    return out


@app.get("/demand-classes")
def demand_classes(session_id: str) -> dict:
    """
    Классификация товаров по характеру спроса (методика SBC).

    Интерфейсу это нужно, чтобы показать у каждого товара честную
    пометку: по нему строится ML-прогноз или он ведётся по правилу
    min/max, потому что спрос нерегулярный.
    """
    s = _session(session_id)
    profiles = profile_all(s.df)
    if profiles.empty:
        return {"items": [], "summary": {}}

    counts = profiles["demand_class"].value_counts().to_dict()
    n = len(profiles)
    n_ok = int(profiles["forecastable"].sum())
    return {
        "items": _jsonable(profiles),
        "summary": {
            "total": n,
            "forecastable": n_ok,
            "not_forecastable": n - n_ok,
            "forecastable_revenue_share": round(
                float(profiles.loc[profiles["forecastable"], "revenue_share"].sum()), 3),
            "by_class": counts,
            "class_labels": CLASS_LABELS_RU,
        },
    }
