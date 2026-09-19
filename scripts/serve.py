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

from src.backtest import economic_impact  # noqa: E402
from src.cleaning import clean  # noqa: E402
from src.demand_classes import CLASS_LABELS_RU, profile_all  # noqa: E402
from src.economics import derive_sku_economics, portfolio_economics  # noqa: E402
from src.forecast import forecast, forecast_curve  # noqa: E402
from src.ingest import IngestError, read_table  # noqa: E402
from src.inventory import build_order_plan  # noqa: E402
from src.model import QuantileModel  # noqa: E402
from src.model_fit import adapt_to_data  # noqa: E402
from src.schema import DATE, QTY, SKU, STOCK  # noqa: E402

WEB_DIR = ROOT / "web"
MODEL_DIR = ROOT / "models" / "global"
MAX_UPLOAD = 25 * 1024 * 1024
SESSION_TTL = timedelta(hours=4)

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
    fc = forecast(s["df"], horizon=horizon, model=_model_for(s), model_dir=None)
    return {"horizon": horizon, "items": _jsonable(fc)}


def h_forecast_curve(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    sku = q.get("sku", [None])[0]
    horizon = int(q.get("horizon", [30])[0])
    curve = forecast_curve(s["df"], sku, horizon, model=_model_for(s), model_dir=None)
    if curve.empty:
        raise ApiError(404, f"Прогноз для «{sku}» построить не удалось.")
    curve = curve.assign(date=curve["date"].dt.strftime("%Y-%m-%d"))
    return {"sku": sku, "horizon": horizon, "points": _jsonable(curve)}


def h_order_plan(q) -> dict:
    s = _session(q.get("session_id", [None])[0])
    horizon = int(q.get("horizon", [30])[0])
    lead = int(q.get("lead_time_days", [7])[0])
    review = int(q.get("review_period_days", [7])[0])

    fc = forecast(s["df"], horizon=horizon, model=_model_for(s), model_dir=None)

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

    key = f"impact_{lead}_{review}_{raw_scope}"
    if key in s:
        return s[key]

    out = _jsonable(economic_impact(s["df"], _model_for(s), max_skus=scope,
                                    lead_time=lead, review_period=review,
                                    fit=s.get("fit")))
    s[key] = out
    # сохраняем последний расчёт, чтобы он попал в текстовый отчёт
    s["last_impact"] = out
    return out


ROUTES = {
    "/health": h_health,
    "/history": h_history,
    "/forecast": h_forecast,
    "/forecast/curve": h_forecast_curve,
    "/order-plan": h_order_plan,
    "/demand-classes": h_demand_classes,
    "/economics": h_economics,
    "/clean-report": h_clean_report,
    "/economic-impact": h_economic_impact,
    "/upload-progress": h_upload_progress,
}

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

        _progress(uid, "Готовлю прогноз…", 90)
        _gc_sessions()
        sid = uuid.uuid4().hex
        with SESSION_LOCK:
            SESSIONS[sid] = {"df": df, "report": report,
                             "model": session_model, "fit": fit,
                             "created": datetime.utcnow()}

        _progress(uid, "Готово", 100)
        self._json({"session_id": sid, "filename": filename,
                    "columns_detected": ingested.mapping,
                    "report": _jsonable(report.to_dict()),
                    "model_fit": _jsonable(fit) if fit else None})


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
