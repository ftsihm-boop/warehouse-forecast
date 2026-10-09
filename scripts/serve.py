"""
Сервер для сайта — без единой сторонней библиотеки.

ЗАЧЕМ ОТДЕЛЬНЫЙ СЕРВЕР, ЕСЛИ ЕСТЬ api/main.py
---------------------------------------------
api/main.py написан на FastAPI — это правильный выбор для продакшена,
но он требует установки трёх пакетов, а на учебной машине установка
может не пройти (нет интернета, корпоративный прокси, старый pip).

Этот скрипт делает ровно то же самое на стандартной библиотеке Python:
ничего ставить не нужно, он запустится везде, где есть Python. Логика
расчётов одна и та же — оба сервера дёргают одни и те же функции из src/.

ЗАПУСК
------
    python scripts/serve.py

Дальше открыть в браузере http://localhost:8000 — сайт отдаётся тем же
сервером, поэтому никаких настроек адреса не требуется.

    python scripts/serve.py --port 8080      другой порт
    python scripts/serve.py --host 0.0.0.0   пустить в локальную сеть
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
import uuid
from datetime import datetime, timedelta
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.accuracy import assess_accuracy  # noqa: E402
from src.anomalies import build_anomaly_report  # noqa: E402
from src.backtest import economic_impact  # noqa: E402
from src.cleaning import clean  # noqa: E402
from src.dataset_info import describe_dataset  # noqa: E402
from src.demand_classes import CLASS_LABELS_RU, profile_all  # noqa: E402
from src.economics import derive_sku_economics, portfolio_economics  # noqa: E402
from src.enrich import feature_overview  # noqa: E402
from src.forecast import curve_from_totals, forecast  # noqa: E402
from src.ingest import IngestError, read_table  # noqa: E402
from src.model import QuantileModel  # noqa: E402
from src.model_fit import adapt_to_data  # noqa: E402
from src.order_report import build_purchase_order, order_filename  # noqa: E402
from src.planning import (  # noqa: E402
    SOURCE_LABELS_RU, clamp_days, effective_horizon, file_lead_times, make_plan,
    resolve_lead_times,
)
from src.recommendations import build_recommendations  # noqa: E402
from src.schema import (  # noqa: E402
    DATE, DEFAULT_LEAD_DAYS, DEFAULT_REVIEW_DAYS, FLAG_CENSORED, QTY, SKU, STOCK,
)

WEB_DIR = ROOT / "web"
MODEL_DIR = ROOT / "models" / "global"
MAX_UPLOAD = 25 * 1024 * 1024

# Сколько живёт сессия и сколько их держать одновременно.
#
# В сессии лежат очищенные данные пользователя и — если модель
# дообучалась — её персональная копия. Это десятки мегабайт на каждую
# загрузку. Контейнеру на площадке выделено меньше трёхсот мегабайт
# памяти, из которых полторы сотни занимают сами библиотеки, поэтому
# копить сессии часами нельзя: несколько загрузок подряд — и процесс
# убьют по нехватке памяти прямо во время показа.
#
# Час жизни покрывает любой разумный сеанс работы, а предел в три
# сессии — одновременную демонстрацию с нескольких вкладок.
#
# Срок скользящий: каждое обращение продлевает сессию. Менеджер, который
# открыл сайт с утра и вернулся к нему после обеда, не должен терять
# работу. А если сервер всё же перезапустился, страница сама загрузит
# сохранённый файл заново (см. восстановление в web/index.html).
SESSION_TTL = timedelta(hours=2)
MAX_SESSIONS = 3

# --- отслеживание этапов обработки файла ----------------------------------
# Загрузка занимает секунды, а дообучение модели — ещё несколько, и всё
# это время пользователь видел пустой экран и не понимал, завис сайт или
# работает. Крутилка «просто подождите» тут не годится: непонятно, ждать
# секунду или минуту.
#
# Поэтому обработчик отмечает реальные этапы, а страница их опрашивает.
# Прогресс настоящий: он отражает то, что сервер делает прямо сейчас,
# а не таймер, нарисованный в браузере.
PROGRESS: dict[str, dict] = {}
PROGRESS_LOCK = threading.Lock()
PROGRESS_TTL = timedelta(minutes=10)


def _progress(uid: str | None, stage: str, pct: int) -> None:
    if not uid:
        return
    with PROGRESS_LOCK:
        PROGRESS[uid] = {"stage": stage, "pct": pct,
                         "ts": datetime.utcnow().isoformat()}
        if len(PROGRESS) > 200:
            cutoff = datetime.utcnow() - PROGRESS_TTL
            for k in [k for k, v in PROGRESS.items()
                      if datetime.fromisoformat(v["ts"]) < cutoff]:
                PROGRESS.pop(k, None)


def h_upload_progress(q) -> dict:
    uid = (q.get("uid") or [""])[0]
    with PROGRESS_LOCK:
        return dict(PROGRESS.get(uid) or {"stage": "Готовлюсь…", "pct": 0})

SESSIONS: dict[str, dict] = {}
SESSION_LOCK = threading.Lock()
MODEL: QuantileModel | None = None


def _load_model() -> None:
    global MODEL
    if (MODEL_DIR / "meta.json").exists():
        try:
            MODEL = QuantileModel.load(MODEL_DIR)
            print(f"  модель загружена ({MODEL.backend})")
        except Exception as e:  # noqa: BLE001
            print(f"  модель не загрузилась: {e}")
            print("  прогноз будет считаться по статистике")
    else:
        print("  обученной модели нет — прогноз по статистике")
        print("  обучить: python -m src.train --data <файл> --out models/global")


def _jsonable(obj):
    """Приводит numpy/pandas-типы к тому, что понимает json."""
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(r) for r in obj.to_dict("records")]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d")
    if obj is pd.NaT or obj is None:
        return None
    return obj


def _gc_sessions() -> None:
    """
    Убирает просроченные сессии и ограничивает их число.

    Ограничение по количеству важнее ограничения по времени: за час
    можно загрузить файл десять раз, и все десять наборов данных будут
    висеть в памяти одновременно. Лишние вытесняются начиная с самых
    старых — пользователь работает с последней загрузкой.
    """
    now = datetime.utcnow()
    with SESSION_LOCK:
        for sid in [s for s, v in SESSIONS.items()
                    if now - v.get("touched", v["created"]) > SESSION_TTL]:
            SESSIONS.pop(sid, None)

        if len(SESSIONS) > MAX_SESSIONS:
            by_age = sorted(SESSIONS.items(),
                            key=lambda kv: kv[1].get("touched", kv[1]["created"]))
            for sid, _v in by_age[:len(SESSIONS) - MAX_SESSIONS]:
                SESSIONS.pop(sid, None)


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _model_for(s: dict) -> QuantileModel | None:
    """
    Модель для этой сессии.

    Если при загрузке выяснилось, что глобальная модель плохо подходит
    к данным пользователя, в сессии лежит дообученная версия — и все
    расчёты должны идти через неё, иначе адаптация была бы бесполезной.
    """
    return s.get("model") or MODEL


def _session(sid: str | None) -> dict:
    if not sid:
        raise ApiError(400, "Не передан session_id.")
    with SESSION_LOCK:
        s = SESSIONS.get(sid)
        if s is not None:
            s["touched"] = datetime.utcnow()
    if s is None:
        raise ApiError(404, "Сессия не найдена или истекла. Загрузите файл заново.")
    return s


# --- обработчики эндпоинтов ---------------------------------------------------

DEMO_FILE = WEB_DIR / "demo.xlsx"   # лежит в web/: каталог data/ в образ не попадает


def h_health(_q) -> dict:
    return {"status": "ok", "model_loaded": MODEL is not None,
            "demo": DEMO_FILE.exists(),
            "backend": MODEL.backend if MODEL else None,
            "sessions": len(SESSIONS)}


def h_history(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    df = s["df"]
    sku = q.get("sku", [None])[0]
    if sku:
        df = df[df[SKU] == sku]
        if df.empty:
            raise ApiError(404, f"Товар «{sku}» не найден.")
    out = df[[DATE, SKU, QTY, STOCK]].copy()
    out[DATE] = out[DATE].dt.strftime("%Y-%m-%d")
    return {"rows": _jsonable(out)}


# --- кэш прогноза ------------------------------------------------------------
# Прогноз зависит только от горизонта. Срок поставки, период заказов, ручные
# правки остатков и «в пути» его не меняют — они работают с готовыми цифрами.
# Поэтому прогноз считается один раз на горизонт и лежит в сессии, а смена
# срока поставки пересчитывает лишь арифметику вокруг него (сотые доли секунды).
def _forecast_cached(s: dict, horizon: int, uid: str | None = None) -> pd.DataFrame:
    with s["fc_lock"]:
        if horizon not in s["fc"]:
            _progress(uid, f"Считаю прогноз на {horizon} дн.…", 60)
            s["fc"][horizon] = forecast(s["df"], horizon=horizon,
                                        model=_model_for(s), model_dir=None)
        return s["fc"][horizon]


def _accuracy_cached(s: dict, horizon: int) -> dict:
    with s["fc_lock"]:
        if horizon not in s["acc"]:
            s["acc"][horizon] = _jsonable(
                assess_accuracy(s["df"], horizon, model=_model_for(s), model_dir=None))
        return s["acc"][horizon]


def h_forecast(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    horizon = int(q.get("horizon", [30])[0])
    return {"horizon": horizon, "items": _jsonable(_forecast_cached(s, horizon))}


def h_forecast_curve(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    sku = q.get("sku", [None])[0]
    horizon = int(q.get("horizon", [30])[0])
    fc = _forecast_cached(s, horizon)
    row = fc[fc["sku"] == sku]
    g = s["df"][s["df"][SKU] == sku]
    if row.empty or g.empty:
        raise ApiError(404, f"Прогноз для «{sku}» построить не удалось.")
    curve = curve_from_totals(g, float(row.iloc[0]["q50"]),
                              float(row.iloc[0]["q90"]), horizon)
    if curve.empty:
        raise ApiError(404, f"Прогноз для «{sku}» построить не удалось.")
    curve = curve.assign(date=curve["date"].dt.strftime("%Y-%m-%d"))
    return {"sku": sku, "horizon": horizon, "points": _jsonable(curve)}


def h_sku(q) -> dict:
    """Всё для карточки товара одним запросом: история, прогноз, показатели."""
    s = _session(q.get("session_id", [None])[0])
    sku = q.get("sku", [None])[0]
    horizon = int(q.get("horizon", [30])[0])
    g = s["df"][s["df"][SKU] == sku].sort_values(DATE)
    if g.empty:
        raise ApiError(404, f"Товар «{sku}» не найден.")
    hist = g[[DATE, QTY, STOCK]].copy()
    hist["c"] = g[FLAG_CENSORED].astype(int) if FLAG_CENSORED in g.columns else 0
    hist[DATE] = hist[DATE].dt.strftime("%Y-%m-%d")

    fc = _forecast_cached(s, horizon)
    row = fc[fc["sku"] == sku]
    curve = []
    if not row.empty and np.isfinite(row.iloc[0]["q50"]):
        c = curve_from_totals(g, float(row.iloc[0]["q50"]),
                              float(row.iloc[0]["q90"]), horizon)
        if not c.empty:
            curve = _jsonable(c.assign(date=c["date"].dt.strftime("%Y-%m-%d")))
    feat = next((f for f in s["features"]["table"] if f["sku"] == sku), None)
    acc = next((a for a in (s["acc"].get(14) or {}).get("items", [])
                if a["sku"] == sku), None)
    return {"sku": sku, "horizon": horizon,
            "history": [{"d": r[DATE], "q": r[QTY],
                         "s": None if pd.isna(r[STOCK]) else r[STOCK], "c": r["c"]}
                        for _, r in hist.iterrows()],
            "forecast": curve, "features": feat, "accuracy": acc}


# --- план закупки ------------------------------------------------------------
def _parse_params(body: dict) -> dict:
    """Приводит настройки от страницы к безопасному виду."""
    def num_map(raw, lo, hi, integer):
        out = {}
        for k, v in (raw or {}).items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(f):
                continue
            f = min(max(f, lo), hi)
            out[str(k)] = int(round(f)) if integer else f
        return out

    try:
        horizon = int(body.get("horizon", 30))
    except (TypeError, ValueError):
        horizon = 30
    return {
        "horizon": min(max(horizon, 7), 180),
        "default_lead": clamp_days(body.get("default_lead", DEFAULT_LEAD_DAYS)),
        "review_period": clamp_days(body.get("review_period", DEFAULT_REVIEW_DAYS), 1, 120),
        "lead_overrides": num_map(body.get("lead_overrides"), 1, 120, True),
        "in_transit": num_map(body.get("in_transit"), 0, 1e9, False),
        "stock_overrides": num_map(body.get("stock_overrides"), 0, 1e9, False),
    }


def _compute_plan(s: dict, params: dict, uid: str | None = None) -> dict:
    """
    Один расчёт плана — для экрана и для заявки в Excel. Если бы каждый
    считал сам, цифры на экране и в скачанном файле со временем разошлись
    бы, и доверять было бы нечему.
    """
    df = s["df"]
    leads, _src = resolve_lead_times(df, params["default_lead"], params["lead_overrides"])
    horizon, horizon_note = effective_horizon(params["horizon"], leads, params["review_period"])
    fc = _forecast_cached(s, horizon, uid)
    res = make_plan(df, fc, horizon, default_lead=params["default_lead"],
                    review_period=params["review_period"],
                    lead_overrides=params["lead_overrides"],
                    in_transit=params["in_transit"],
                    stock_overrides=params["stock_overrides"])
    res["horizon"] = horizon
    res["horizon_note"] = horizon_note
    s["last_leads"] = res["lead_times"]          # для расчёта эффекта
    s["last_params"] = params
    return res


def h_plan(body: dict) -> dict:
    s = _session(body.get("session_id"))
    params = _parse_params(body)
    res = _compute_plan(s, params, body.get("uid"))
    plan = res["plan"]

    acc = s["acc"].get(14)
    recs = build_recommendations(
        plan, s["anomalies"], acc, s["features"]["table"],
        res["prices"], res["costs"], lead_from_file=bool(file_lead_times(s["df"])))

    items = _jsonable(plan)
    for it in items:
        it["lead_source_label"] = SOURCE_LABELS_RU.get(it.get("lead_source"), "")
    return {
        "params": {**params, "horizon_used": res["horizon"]},
        "horizon_note": res["horizon_note"],
        "as_of": res["as_of"].strftime("%Y-%m-%d"),
        "summary": _jsonable(res["summary"]),
        "economics": _jsonable(res["eco_info"]),
        "items": items,
        "recommendations": _jsonable(recs),
    }


def h_plan_xlsx_data(body: dict) -> tuple[bytes, str]:
    s = _session(body.get("session_id"))
    params = _parse_params(body)
    res = _compute_plan(s, params)
    rep = s.get("report")
    df = s["df"]
    items = _jsonable(res["plan"])
    custom = sum(1 for v in res["lead_sources"].values() if v != "default")
    meta = {
        "source_file": s.get("filename", "—"),
        "period": f"{df[DATE].min():%d.%m.%Y} — {df[DATE].max():%d.%m.%Y}",
        "horizon": res["horizon"], "lead_time": params["default_lead"],
        "lead_custom": custom or None,
        "review_period": params["review_period"],
        "clean_report": rep.to_dict() if rep is not None else {},
        "model_fit": s.get("fit") or {},
        "model_backend": (m.backend if (m := _model_for(s)) else None),
    }
    return build_purchase_order(items, res["costs"], meta), order_filename()


def h_order_plan(q) -> dict:
    """Совместимость со старым адресом: те же расчёты, настройки в адресной строке."""
    body = {"session_id": q.get("session_id", [None])[0],
            "horizon": q.get("horizon", [30])[0],
            "default_lead": q.get("lead_time_days", [DEFAULT_LEAD_DAYS])[0],
            "review_period": q.get("review_period_days", [DEFAULT_REVIEW_DAYS])[0]}
    out = h_plan(body)
    counts: dict[str, int] = {}
    for it in out["items"]:
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    out.update({"horizon": out["params"]["horizon_used"],
                "lead_time_days": out["params"]["default_lead"],
                "status_counts": counts})
    return out


# --- описание данных ------------------------------------------------------------
def h_dataset(q) -> dict:
    return _jsonable(_session(q.get("session_id", [None])[0])["dataset"])


def h_features(q) -> dict:
    return _jsonable(_session(q.get("session_id", [None])[0])["features"])


def h_anomalies(q) -> dict:
    return _jsonable(_session(q.get("session_id", [None])[0])["anomalies"])


def h_accuracy(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    horizon = min(max(int(q.get("horizon", [14])[0]), 7), 60)
    return _accuracy_cached(s, horizon)


def _session_payload(sid: str, s: dict) -> dict:
    """Всё, что странице нужно, чтобы нарисоваться после загрузки или F5."""
    df = s["df"]
    ds = s["dataset"]
    return {
        "session_id": sid, "filename": s.get("filename"),
        "columns_detected": s.get("mapping"),
        "report": _jsonable(s["report"].to_dict()),
        "model_fit": _jsonable(s.get("fit")) if s.get("fit") else None,
        "overview": _jsonable(ds["overview"]),
        "skus": sorted(df[SKU].unique().tolist()),
        "has_lead_column": bool(file_lead_times(df)),
        "file_leads": file_lead_times(df),
        "has_stock": bool(df[STOCK].notna().any()) if STOCK in df.columns else False,
        "anomaly_headline": s["anomalies"]["headline"],
        "anomaly_health": s["anomalies"]["health"],
        "attention_count": s["anomalies"]["totals"]["attention_kinds"],
        "accuracy_label": (s["acc"].get(14) or {}).get("overall_label"),
        "accuracy_level": (s["acc"].get(14) or {}).get("overall"),
    }


def h_session(q) -> dict:
    sid = q.get("session_id", [None])[0]
    return _session_payload(sid, _session(sid))


def h_demand_classes(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    profiles = profile_all(s["df"])
    if profiles.empty:
        return {"items": [], "summary": {}}
    n = len(profiles)
    n_ok = int(profiles["forecastable"].sum())
    return {
        "items": _jsonable(profiles),
        "summary": {
            "total": n, "forecastable": n_ok, "not_forecastable": n - n_ok,
            "by_class": _jsonable(profiles["demand_class"].value_counts().to_dict()),
            "class_labels": CLASS_LABELS_RU,
        },
    }


def h_economics(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    cover = int(q.get("lead_time_days", [7])[0]) + int(q.get("review_period_days", [7])[0])
    eco = derive_sku_economics(s["df"], cover_days=cover)
    return {"items": _jsonable(eco),
            "portfolio": _jsonable(portfolio_economics(s["df"], cover_days=cover))}


def h_clean_report(q) -> dict:
    return _jsonable(_session(q.get("session_id", [None])[0])["report"].to_dict())


def h_economic_impact(q) -> dict:
    """
    Экономический эффект: снижение затрат на хранение и предотвращённые
    потери от дефицита. Считается бэктестом, поэтому занимает секунды —
    интерфейс вызывает это по кнопке, а не при загрузке файла.
    Результат кэшируется в сессии: повторный запрос мгновенный.
    """
    s = _session(q.get("session_id", [None])[0])
    lead = int(q.get("lead_time_days", [7])[0])
    review = int(q.get("review_period_days", [7])[0])

    # scope: сколько самых оборотистых товаров брать. "all" — все,
    # у кого хватает истории.
    raw_scope = (q.get("scope", ["10"])[0] or "10").lower()
    scope = None if raw_scope in ("all", "все", "0") else max(1, int(raw_scope))

    # если менеджер уже задавал сроки по товарам — эффект считается с ними
    leads = s.get("last_leads") or None
    sig = hash(tuple(sorted((leads or {}).items())))
    key = f"impact_{lead}_{review}_{raw_scope}_{sig}"
    if key in s:
        return s[key]

    out = _jsonable(economic_impact(s["df"], _model_for(s), max_skus=scope,
                                    lead_time=lead, review_period=review,
                                    fit=s.get("fit"), lead_times=leads))
    s[key] = out
    # сохраняем последний расчёт, чтобы он попал в текстовый отчёт
    s["last_impact"] = out
    return out


ROUTES = {
    "/health": h_health,
    "/history": h_history,
    "/forecast": h_forecast,
    "/forecast/curve": h_forecast_curve,
    "/sku": h_sku,
    "/session": h_session,
    "/dataset": h_dataset,
    "/features": h_features,
    "/anomalies": h_anomalies,
    "/accuracy": h_accuracy,
    "/order-plan": h_order_plan,
    "/demand-classes": h_demand_classes,
    "/economics": h_economics,
    "/clean-report": h_clean_report,
    "/economic-impact": h_economic_impact,
    "/upload-progress": h_upload_progress,
}

XLSX_TYPE = ("application/vnd.openxmlformats-officedocument."
             "spreadsheetml.sheet")

STATIC_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


def _impact_to_text(d: dict) -> str:
    """Экономический эффект в том же текстовом виде, что и отчёт об очистке."""
    def pct(v):
        return "н/д" if v is None else f"{v:.1f} %"

    def money(v):
        return f"{v:,.0f} \u20BD".replace(",", " ")

    L = [
        "ЭКОНОМИЧЕСКИЙ ЭФФЕКТ",
        "=" * 52,
        f"Охват расчёта:                  {d.get('skus_analyzed')} товар(ов) "
        f"из {d.get('skus_total_eligible')} пригодных",
        f"Период бэктеста:                {d.get('period_days')} дней",
        "-" * 52,
        "ЗАТРАТЫ НА ХРАНЕНИЕ",
        f"  было (ручное планирование):   {money(d.get('holding_before', 0))}",
        f"  стало (система):              {money(d.get('holding_after', 0))}",
        f"  снижение:                     {pct(d.get('holding_saved_pct'))}",
        "",
        "ПОТЕРИ ОТ ДЕФИЦИТА (упущенная выручка)",
        f"  было:                         {money(d.get('lost_before', 0))}",
        f"  стало:                        {money(d.get('lost_after', 0))}",
        f"  предотвращено:                {money(d.get('lost_prevented', 0))} "
        f"({pct(d.get('lost_prevented_pct'))})",
        "",
        "ТОВАРНЫЙ ОСТАТОК (замороженные средства)",
        f"  было:                         {money(d.get('stock_before', 0))}",
        f"  стало:                        {money(d.get('stock_after', 0))}",
        f"  снижение:                     {pct(d.get('stock_reduced_pct'))}",
        "-" * 52,
        f"СОВОКУПНЫЙ ЭФФЕКТ ЗА ПЕРИОД:    {money(d.get('total_effect_period', 0))}",
        f"ЭФФЕКТ В МЕСЯЦ:                 {money(d.get('effect_per_month', 0))}",
        "-" * 52,
        "ДОПУЩЕНИЯ РАСЧЁТА",
        f"  закупочная цена:              {money(d.get('unit_cost', 0))}",
        f"  наценка:                      {d.get('margin', 0) * 100:.1f} %",
        f"  страховой запас:              \u00d7 {d.get('service_factor')}"
        + f" (медиана; по товарам от \u00d7 {d.get('factor_min')} "
          f"до \u00d7 {d.get('factor_max')})"
        + (f", ориентир \u00d7 {d.get('theoretical_factor')}"
           if d.get("theoretical_factor") else ""),
        "",
        "УПРАВЛЕНИЕ ЗАКУПКОЙ",
        f"  по прогнозу:                  {d.get('skus_managed', 0)} товар(ов)",
        f"  оставлено на ручном:          {d.get('skus_manual', 0)} товар(ов)"
        + ("\n    " + ", ".join(str(s) for s in (d.get("manual_skus") or []))
           if d.get("manual_skus") else ""),
        "",
        "  Метод: история прогнана дважды — как если бы закупками управляли",
        "  вручную (средний спрос за 28 дней с запасом прочности 1.25) и как",
        "  это делает система (прогноз модели + страховой запас по",
        "  доверительному интервалу). Разница между политиками и есть эффект.",
    ]

    diag = d.get("diagnosis") or {}
    if diag.get("notes"):
        L += ["-" * 52, "ЗАМЕЧАНИЯ:"]
        for n in diag["notes"]:
            L.append("  ! " + n)
    return "\n".join(L)


class Handler(BaseHTTPRequestHandler):
    server_version = "WarehouseForecast/1.0"

    def log_message(self, fmt, *args):  # компактный лог
        if "GET /health" not in (fmt % args):
            sys.stderr.write("  %s\n" % (fmt % args))

    # --- вспомогательное ---------------------------------------------------
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _download(self, data: bytes, filename: str,
                  ctype: str, status: int = 200) -> None:
        """
        Отдаёт файл на скачивание.

        Имя файла русское, поэтому заголовок пишется в форме filename*
        по RFC 5987: без неё браузер сохранит документ под искажённым
        именем. Простой filename= оставлен рядом для старых клиентов.
        """
        from urllib.parse import quote

        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header(
            "Content-Disposition",
            "attachment; filename=\"order.xlsx\"; "
            f"filename*=UTF-8''{quote(filename)}")
        self.end_headers()
        self.wfile.write(data)

    def _text(self, text: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    # --- GET ----------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path, q = parsed.path, parse_qs(parsed.query)

        if path == "/order-plan/xlsx":
            try:
                data, name = h_plan_xlsx_data({
                    "session_id": q.get("session_id", [None])[0],
                    "horizon": q.get("horizon", [30])[0],
                    "default_lead": q.get("lead_time_days", [DEFAULT_LEAD_DAYS])[0],
                    "review_period": q.get("review_period_days", [DEFAULT_REVIEW_DAYS])[0]})
                self._download(data, name, XLSX_TYPE)
            except ApiError as e:
                self._json({"detail": e.message}, e.status)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json({"detail": f"Не удалось собрать заявку: {e}"}, 500)
            return

        if path == "/demo.xlsx":
            if not DEMO_FILE.exists():
                self._text("Not found", 404)
                return
            body = DEMO_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", XLSX_TYPE)
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/clean-report/text":
            try:
                s = _session(q.get("session_id", [None])[0])
                text = s["report"].to_text()
                # если эффект уже считали — он идёт в тот же отчёт,
                # чтобы выгрузка была цельной и её можно было приложить
                # к пояснительной записке как есть
                impact = s.get("last_impact")
                if impact and not impact.get("error"):
                    text += "\n\n" + _impact_to_text(impact)
                self._text(text)
            except ApiError as e:
                self._text(e.message, e.status)
            return

        if path in ROUTES:
            try:
                self._json(ROUTES[path](q))
            except ApiError as e:
                self._json({"detail": e.message}, e.status)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json({"detail": f"Ошибка сервера: {e}"}, 500)
            return

        self._serve_static(path)

    def _serve_static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (WEB_DIR / rel).resolve()
        try:
            target.relative_to(WEB_DIR.resolve())
        except ValueError:
            self._text("Forbidden", 403)
            return
        if not target.is_file():
            self._text("Not found", 404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",
                         STATIC_TYPES.get(target.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    # --- POST ---------------------------------------------------------------
    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 2 * 1024 * 1024:
            raise ApiError(413, "Слишком большой запрос.")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError) as e:
            raise ApiError(400, "Запрос не разобран.") from e

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/upload":
                self._handle_upload()
            elif path == "/plan":
                self._json(h_plan(self._read_json()))
            elif path == "/plan/xlsx":
                data, name = h_plan_xlsx_data(self._read_json())
                self._download(data, name, XLSX_TYPE)
            else:
                self._json({"detail": "Неизвестный адрес."}, 404)
        except ApiError as e:
            self._json({"detail": e.message}, e.status)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"detail": f"Ошибка сервера: {e}"}, 500)

    def _handle_upload(self) -> None:
        uid = (parse_qs(urlparse(self.path).query).get("uid") or [""])[0]
        _progress(uid, "Принимаю файл…", 5)

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ApiError(400, "Пустой запрос.")
        if length > MAX_UPLOAD:
            raise ApiError(413, f"Файл больше {MAX_UPLOAD // 1024 // 1024} МБ.")

        ctype = self.headers.get("Content-Type", "")
        raw = self.rfile.read(length)

        # разбор multipart/form-data средствами стандартной библиотеки
        header = f"Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        msg = BytesParser(policy=email_default).parsebytes(header + raw)
        filename, payload = None, None
        if msg.is_multipart():
            for part in msg.iter_parts():
                if part.get_param("name", header="content-disposition") == "file":
                    filename = part.get_filename() or "upload.csv"
                    payload = part.get_payload(decode=True)
                    break
        if payload is None:
            raise ApiError(400, "В запросе нет файла.")

        _progress(uid, "Разбираю таблицу и распознаю колонки…", 15)
        try:
            ingested = read_table(payload, filename)
        except IngestError as e:
            raise ApiError(422, str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise ApiError(422, f"Не удалось разобрать файл: {e}") from e

        _progress(uid, "Чищу данные: дубли, выбросы, пропуски…", 30)
        df, report = clean(ingested)
        if df.empty:
            raise ApiError(422, "После очистки не осталось пригодных данных.")

        # Проверяем, подходит ли модель к этим данным, и при
        # необходимости дообучаем её прямо сейчас. Так пользователь
        # получает прогноз, настроенный под его ассортимент, а не
        # усреднённый по чужому.
        session_model, fit = None, None
        if MODEL is not None:
            _progress(uid, "Дообучаю модель на ваших данных…", 45)
            try:
                session_model, fit_report = adapt_to_data(df, MODEL)
                fit = fit_report.to_dict()
                if fit_report.action != "fine_tuned":
                    session_model = None      # осталась глобальная
                print(f"  проверка модели: {fit_report.message}")
            except Exception as e:  # noqa: BLE001
                print(f"  проверка модели не удалась: {e}")

        _progress(uid, "Описываю данные и ищу аномалии…", 70)
        sid = uuid.uuid4().hex
        sess = {"df": df, "report": report, "model": session_model, "fit": fit,
                "filename": filename, "mapping": ingested.mapping,
                "created": datetime.utcnow(), "touched": datetime.utcnow(),
                "fc": {}, "acc": {}, "fc_lock": threading.RLock()}
        sess["dataset"] = describe_dataset(df, report)
        sess["anomalies"] = build_anomaly_report(df, report)
        sess["features"] = feature_overview(df)

        # Первый прогноз и оценка точности считаются сразу, пока человек
        # смотрит на индикатор: так экран «Что заказать» откроется мгновенно.
        _progress(uid, "Строю прогноз спроса…", 80)
        _forecast_cached(sess, 30, uid)
        _progress(uid, "Проверяю точность на прошлом…", 92)
        _accuracy_cached(sess, 14)

        with SESSION_LOCK:
            SESSIONS[sid] = sess
        # Чистим ПОСЛЕ добавления: иначе новая сессия ложится сверх
        # лимита и их фактически остаётся на одну больше.
        _gc_sessions()

        _progress(uid, "Готово", 100)
        self._json(_session_payload(sid, sess))


def main() -> None:
    # На локальной машине сервер слушает только себя — так безопаснее.
    # В контейнере всё наоборот: адрес и порт задаёт платформа через
    # переменные окружения, и слушать нужно все интерфейсы, иначе
    # снаружи до приложения не достучаться.
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "8000")))
    a = ap.parse_args()

    print("=" * 58)
    print("  Сервер системы управления складскими запасами")
    print("=" * 58)
    _load_model()

    if not (WEB_DIR / "index.html").exists():
        print(f"  ВНИМАНИЕ: не найден {WEB_DIR / 'index.html'}")
        print("  сайт отдаваться не будет, API работать будет")

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    shown = "localhost" if a.host in ("127.0.0.1", "0.0.0.0") else a.host
    print("-" * 58)
    print(f"  Откройте в браузере:  http://{shown}:{a.port}")
    print("  Остановить: Ctrl+C")
    print("-" * 58)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
        srv.shutdown()


if __name__ == "__main__":
    main()
