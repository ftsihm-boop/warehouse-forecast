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

from src.cleaning import clean  # noqa: E402
from src.demand_classes import CLASS_LABELS_RU, profile_all  # noqa: E402
from src.economics import derive_sku_economics, portfolio_economics  # noqa: E402
from src.forecast import forecast, forecast_curve  # noqa: E402
from src.ingest import IngestError, read_table  # noqa: E402
from src.inventory import build_order_plan  # noqa: E402
from src.model import QuantileModel  # noqa: E402
from src.schema import DATE, QTY, SKU, STOCK  # noqa: E402

WEB_DIR = ROOT / "web"
MODEL_DIR = ROOT / "models" / "global"
MAX_UPLOAD = 25 * 1024 * 1024
SESSION_TTL = timedelta(hours=4)

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
    now = datetime.utcnow()
    with SESSION_LOCK:
        for sid in [s for s, v in SESSIONS.items()
                    if now - v["created"] > SESSION_TTL]:
            SESSIONS.pop(sid, None)


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _session(sid: str | None) -> dict:
    if not sid:
        raise ApiError(400, "Не передан session_id.")
    with SESSION_LOCK:
        s = SESSIONS.get(sid)
    if s is None:
        raise ApiError(404, "Сессия не найдена или истекла. Загрузите файл заново.")
    return s


# --- обработчики эндпоинтов ---------------------------------------------------

def h_health(_q) -> dict:
    return {"status": "ok", "model_loaded": MODEL is not None,
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


def h_forecast(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    horizon = int(q.get("horizon", [30])[0])
    fc = forecast(s["df"], horizon=horizon, model=MODEL, model_dir=None)
    return {"horizon": horizon, "items": _jsonable(fc)}


def h_forecast_curve(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    sku = q.get("sku", [None])[0]
    horizon = int(q.get("horizon", [30])[0])
    curve = forecast_curve(s["df"], sku, horizon, model=MODEL, model_dir=None)
    if curve.empty:
        raise ApiError(404, f"Прогноз для «{sku}» построить не удалось.")
    curve = curve.assign(date=curve["date"].dt.strftime("%Y-%m-%d"))
    return {"sku": sku, "horizon": horizon, "points": _jsonable(curve)}


def h_order_plan(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    horizon = int(q.get("horizon", [30])[0])
    lead = int(q.get("lead_time_days", [7])[0])
    review = int(q.get("review_period_days", [7])[0])

    fc = forecast(s["df"], horizon=horizon, model=MODEL, model_dir=None)

    # страховой запас по каждому товару — из его собственной экономики
    factors, eco_info = None, {}
    eco = derive_sku_economics(s["df"], cover_days=lead + review)
    if not eco.empty:
        factors = dict(zip(eco["sku"], eco["service_factor"]))
        from_data = int((eco["source"] == "данные").sum())
        eco_info = {"prices_from_file": from_data,
                    "prices_estimated": int(len(eco) - from_data),
                    "avg_service_level": round(float(eco["service_level"].mean()), 3)}

    plan = build_order_plan(fc, horizon_days=horizon, lead_time_days=lead,
                            review_period_days=review, service_factors=factors)
    counts = plan["status"].value_counts().to_dict() if not plan.empty else {}
    return {"horizon": horizon, "lead_time_days": lead,
            "status_counts": _jsonable(counts), "economics": _jsonable(eco_info),
            "items": _jsonable(plan)}


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


ROUTES = {
    "/health": h_health,
    "/history": h_history,
    "/forecast": h_forecast,
    "/forecast/curve": h_forecast_curve,
    "/order-plan": h_order_plan,
    "/demand-classes": h_demand_classes,
    "/economics": h_economics,
    "/clean-report": h_clean_report,
}

STATIC_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


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

        if path == "/clean-report/text":
            try:
                s = _session(q.get("session_id", [None])[0])
                self._text(s["report"].to_text())
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
    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/upload":
            self._json({"detail": "Неизвестный адрес."}, 404)
            return
        try:
            self._handle_upload()
        except ApiError as e:
            self._json({"detail": e.message}, e.status)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json({"detail": f"Ошибка сервера: {e}"}, 500)

    def _handle_upload(self) -> None:
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

        try:
            ingested = read_table(payload, filename)
        except IngestError as e:
            raise ApiError(422, str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise ApiError(422, f"Не удалось разобрать файл: {e}") from e

        df, report = clean(ingested)
        if df.empty:
            raise ApiError(422, "После очистки не осталось пригодных данных.")

        _gc_sessions()
        sid = uuid.uuid4().hex
        with SESSION_LOCK:
            SESSIONS[sid] = {"df": df, "report": report,
                             "created": datetime.utcnow()}

        self._json({"session_id": sid, "filename": filename,
                    "columns_detected": ingested.mapping,
                    "report": _jsonable(report.to_dict())})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
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
