"""
Встраивает связь с API в существующий сайт.

Сайт остаётся рабочим БЕЗ сервера: если API недоступен, он считает
прогноз сам, как и раньше. Если сервер есть — берёт настоящую модель.
Это не запасной вариант «на всякий случай», а осмысленный режим:
файл можно открыть двойным кликом и показать без всякой установки.

Скрипт идемпотентный — повторный запуск ничего не сломает.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "web" / "index.html"

MARKER = "/* === API-INTEGRATION === */"

# --- 1. Состояние API ---------------------------------------------------------
STATE_ANCHOR = '  var viewMode = "single"; // "single" | "all"'
STATE_CODE = '''  var viewMode = "single"; // "single" | "all"

  /* === API-INTEGRATION === */
  // Адрес сервера с моделью. Порядок поиска:
  //   1) ?api=http://... в адресной строке (удобно для демонстрации)
  //   2) тот же хост, откуда открыта страница (когда всё на одном сервере)
  //   3) localhost:8000 — обычный локальный запуск
  var API_BASE = (function () {
    try {
      var p = new URLSearchParams(location.search).get("api");
      if (p) return p.replace(/\\/$/, "");
      if (location.protocol === "http:" || location.protocol === "https:") {
        return location.origin;
      }
    } catch (e) { /* открыт как file:// — идём дальше */ }
    return "http://localhost:8000";
  })();

  var sessionId = null;        // выдаётся сервером после загрузки файла
  var serverStats = {};        // sku -> строка плана закупки с сервера
  var serverMeta = {};         // sku -> режим прогноза и класс спроса
  var cleaningReport = null;   // отчёт об очистке данных
  var usingServer = false;     // работаем от модели или считаем локально
'''

# --- 2. Клиент API ------------------------------------------------------------
CLIENT_ANCHOR = "  function computeStats(sku) {"
CLIENT_CODE = '''  /* === API-INTEGRATION === */

  function apiUrl(path, params) {
    var u = API_BASE + path;
    var qs = Object.keys(params || {})
      .filter(function (k) { return params[k] !== null && params[k] !== undefined; })
      .map(function (k) { return encodeURIComponent(k) + "=" + encodeURIComponent(params[k]); })
      .join("&");
    return qs ? u + "?" + qs : u;
  }

  // Загружает файл на сервер. Возвращает true, если модель приняла файл.
  async function uploadToServer(file) {
    var fd = new FormData();
    fd.append("file", file);
    var res = await fetch(API_BASE + "/upload", { method: "POST", body: fd });
    if (!res.ok) {
      var detail = "";
      try { detail = (await res.json()).detail || ""; } catch (e) { /* пусто */ }
      throw new Error(detail || ("Сервер вернул " + res.status));
    }
    var out = await res.json();
    sessionId = out.session_id;
    cleaningReport = out.report;
    return true;
  }

  // Забирает с сервера всё, что нужно интерфейсу: очищенную историю,
  // план закупки и классы спроса.
  async function loadFromServer() {
    var histRes = await fetch(apiUrl("/history", { session_id: sessionId }));
    if (!histRes.ok) throw new Error("Не удалось получить историю продаж");
    var hist = (await histRes.json()).rows || [];

    data = {};
    skuOrder = [];
    hist.forEach(function (row) {
      var sku = row.sku;
      if (!data[sku]) {
        data[sku] = { history: [], stockEntries: [], currentStock: null };
        skuOrder.push(sku);
      }
      var d = parseISODateInput(row.date);
      if (!d) return;
      data[sku].history.push({ date: d, qty: Number(row.qty) || 0 });
      if (row.stock !== null && row.stock !== undefined) {
        data[sku].stockEntries.push({ date: d, stock: Number(row.stock) });
        data[sku].currentStock = Number(row.stock);
      }
    });

    var planRes = await fetch(apiUrl("/order-plan", {
      session_id: sessionId,
      horizon: horizonDays,
      lead_time_days: leadTimeDays
    }));
    if (!planRes.ok) throw new Error("Не удалось получить план закупки");
    var plan = (await planRes.json()).items || [];

    serverStats = {};
    plan.forEach(function (p) {
      var hasStock = p.stock !== null && p.stock !== undefined && !isNaN(p.stock);
      var daily = p.forecast_horizon_q50 / Math.max(horizonDays, 1);
      serverStats[p.sku] = {
        avgDaily: daily,
        spanDays: null,
        forecastHorizon: p.forecast_horizon_q50,
        forecastHigh: p.forecast_horizon_q90,
        safetyStock: p.safety_stock,
        reorderPoint: p.reorder_point,
        stock: hasStock ? p.stock : null,
        hasStock: hasStock,
        daysUntilStockout: p.days_of_supply >= 0 ? p.days_of_supply : null,
        recommendedOrder: p.recommended_order,
        // статусы сервера богаче, чем у локального расчёта: сводим их
        // к трём, которые умеет рисовать таблица
        status: (p.status === "critical" || p.status === "alert") ? "alert"
              : (p.status === "excess" || p.status === "ok") ? "ok" : "unknown",
        serverStatus: p.status,
        statusLabel: p.status_label,
        mode: p.mode,
        note: p.note
      };
    });

    // классы спроса — не критично, если не ответит
    serverMeta = {};
    try {
      var clsRes = await fetch(apiUrl("/demand-classes", { session_id: sessionId }));
      if (clsRes.ok) {
        ((await clsRes.json()).items || []).forEach(function (c) {
          serverMeta[c.sku] = c;
        });
      }
    } catch (e) { /* работаем без классов */ }

    skuOrder.forEach(function (sku) {
      data[sku].history.sort(function (a, b) { return a.date - b.date; });
    });
    usingServer = true;
  }

  // Пересчёт при смене горизонта или срока поставки: на сервере это
  // другая модель, а не пересчёт той же формулы, поэтому перезапрашиваем.
  async function refreshFromServer() {
    if (!usingServer || !sessionId) return false;
    try {
      await loadFromServer();
      renderAll();
      renderServerPanel();
      return true;
    } catch (e) {
      return false;
    }
  }

'''

# --- 3. computeStats берёт серверный результат --------------------------------
COMPUTE_OLD = """  function computeStats(sku) {
    var rec = data[sku];
    var hist = rec.history;
    if (!hist || hist.length === 0) return null;"""
COMPUTE_NEW = """  function computeStats(sku) {
    /* === API-INTEGRATION === */
    // Если сервер посчитал — берём его результат: там настоящая модель,
    // обученная на исторических данных, а не среднее по окну.
    if (usingServer && serverStats[sku]) return serverStats[sku];

    var rec = data[sku];
    var hist = rec.history;
    if (!hist || hist.length === 0) return null;"""

# --- 4. routeFile пробует сервер ----------------------------------------------
ROUTE_OLD = """  function routeFile(file) {
    if (!file) return;
    if (/\\.xlsx$/i.test(file.name)) {
      handleXLSXFile(file);
    } else {
      var reader = new FileReader();
      reader.onload = function (e) { handleFileText(e.target.result); };
      reader.readAsText(file, "UTF-8");
    }
  }"""
ROUTE_NEW = """  function routeFile(file) {
    if (!file) return;
    /* === API-INTEGRATION === */
    // Сначала пробуем сервер с обученной моделью. Не ответил — считаем
    // сами, как раньше: страница остаётся работоспособной без установки.
    routeFileViaServer(file);
  }

  async function routeFileViaServer(file) {
    setServerStatus("connecting");
    try {
      await uploadToServer(file);
      await loadFromServer();
      if (!skuOrder.length) throw new Error("Сервер не вернул ни одного товара");

      // дальше — та же инициализация, что и при локальной загрузке
      var allDates = [];
      skuOrder.forEach(function (sku) {
        data[sku].history.forEach(function (h) { allDates.push(h.date); });
      });
      globalMinDate = new Date(Math.min.apply(null, allDates));
      globalMaxDate = new Date(Math.max.apply(null, allDates));
      var windowLen = 30;
      var candidateStart = addDays(globalMaxDate, -(windowLen - 1));
      rangeStart = candidateStart < globalMinDate ? new Date(globalMinDate) : candidateStart;
      rangeEnd = new Date(globalMaxDate);
      viewMode = "single";
      selectedSku = skuOrder[0];
      if (prevYearCheckbox) prevYearCheckbox.checked = false;
      errorBox.className = "error-box";

      applyViewMode();
      updateRangeInputs();
      showApp();
      renderAll();
      renderServerPanel();
      return;
    } catch (err) {
      usingServer = false;
      sessionId = null;
      serverStats = {};
      setServerStatus("offline", err && err.message);
    }
    routeFileLocally(file);
  }

  function routeFileLocally(file) {
    if (/\\.xlsx$/i.test(file.name)) {
      handleXLSXFile(file);
    } else {
      var reader = new FileReader();
      reader.onload = function (e) { handleFileText(e.target.result); };
      reader.readAsText(file, "UTF-8");
    }
  }"""

# --- 5. Панель статуса и отчёта -----------------------------------------------
PANEL_ANCHOR = "  function renderAll() {"
PANEL_CODE = '''  /* === API-INTEGRATION === */

  function setServerStatus(state, detail) {
    var el = document.getElementById("server-status");
    if (!el) return;
    el.className = "server-status " + state;
    if (state === "connecting") {
      el.textContent = "Подключаюсь к модели...";
    } else if (state === "online") {
      el.textContent = "Прогноз строит обученная модель";
    } else {
      el.textContent = "Модель недоступна — расчёт по простой формуле"
        + (detail ? " (" + detail + ")" : "");
    }
  }

  // Сноска под таблицей должна говорить правду о том, чем считали.
  function updateFootnote() {
    var el = document.querySelector(".footnote");
    if (!el) return;
    if (usingServer) {
      el.removeAttribute("data-i18n");   // иначе перевод перезапишет текст
      el.textContent = "Прогноз строит модель градиентного бустинга, "
        + "обученная на исторических данных. Страховой запас рассчитан по "
        + "доверительному интервалу прогноза и экономике товара, а не по "
        + "фиксированному нормативу. Бейдж у товара показывает, каким "
        + "методом он посчитан.";
    }
  }

  // Поле «Период расчёта спроса» относится к локальной формуле:
  // когда считает модель, оно ни на что не влияет — прячем, чтобы
  // не создавать впечатление работающей настройки.
  function toggleLocalOnlyControls() {
    var input = document.getElementById("demand-window-input");
    var label = document.getElementById("demand-window-all-label");
    var field = input ? input.closest(".field") : null;
    [field, label].forEach(function (el) {
      if (el) el.style.display = usingServer ? "none" : "";
    });
  }

  // Показывает, что именно сделала очистка с загруженным файлом.
  function renderServerPanel() {
    var panel = document.getElementById("clean-report");
    if (!panel) return;

    toggleLocalOnlyControls();
    updateFootnote();

    if (!usingServer || !cleaningReport) {
      panel.style.display = "none";
      setServerStatus("offline");
      return;
    }
    setServerStatus("online");

    var r = cleaningReport;
    var items = [
      ["Прочитано строк", r.rows_read],
      ["Дублей объединено", r.duplicates_merged],
      ["Пропущенных дат дозаполнено", r.dates_filled],
      ["Возвратов обнулено", r.returns_zeroed],
      ["Выбросов обрезано", r.outliers_winsorized],
      ["Дней дефицита найдено", r.censored_days]
    ].filter(function (p) { return p[1]; });

    var html = '<div class="clean-head">Данные обработаны</div><div class="clean-grid">';
    items.forEach(function (p) {
      html += '<div class="clean-item"><span class="n">' + p[1] +
              '</span><span class="l">' + escapeHtml(p[0]) + '</span></div>';
    });
    html += '</div>';

    if (r.warnings && r.warnings.length) {
      html += '<ul class="clean-warn">';
      r.warnings.forEach(function (w) {
        html += '<li>' + escapeHtml(w) + '</li>';
      });
      html += '</ul>';
    }
    if (sessionId) {
      html += '<a class="clean-log" target="_blank" href="' +
              apiUrl("/clean-report/text", { session_id: sessionId }) +
              '">Скачать полный отчёт</a>';
    }
    panel.innerHTML = html;
    panel.style.display = "block";
  }

  function modeBadge(sku) {
    var st = serverStats[sku];
    if (!usingServer || !st) return "";
    var cls = "badge-ml", text = "модель";
    if (st.mode === "minmax") { cls = "badge-minmax"; text = "min/max"; }
    else if (st.mode === "stats") { cls = "badge-stats"; text = "статистика"; }
    else if (st.mode === "fine_tune") { cls = "badge-ml"; text = "дообучена"; }
    var meta = serverMeta[sku];
    var title = meta ? (meta.class_label + " · " + (meta.recommendation || "")) : "";
    return '<span class="mode-badge ' + cls + '" title="' + escapeHtml(title) +
           '">' + text + '</span>';
  }

'''

# --- 6. Бейдж в таблицу -------------------------------------------------------
TABLE_OLD = '''        "<td>" + escapeHtml(r.sku) + "</td>" +'''
TABLE_NEW = '''        "<td>" + escapeHtml(r.sku) + modeBadge(r.sku) + "</td>" +'''

# --- 6б. Смена параметров перезапрашивает модель ------------------------------
HORIZON_OLD = '''  horizonSelect.addEventListener("change", function () {
    horizonDays = parseInt(horizonSelect.value, 10);
    renderAll();
  });'''
HORIZON_NEW = '''  horizonSelect.addEventListener("change", function () {
    horizonDays = parseInt(horizonSelect.value, 10);
    /* === API-INTEGRATION === */
    // На сервере другой горизонт — это другой прогноз модели,
    // а не пересчёт той же формулы, поэтому перезапрашиваем.
    if (usingServer) { refreshFromServer(); return; }
    renderAll();
  });'''

LEADTIME_OLD = '''  leadtimeInput.addEventListener("change", function () {
    var v = parseInt(leadtimeInput.value, 10);
    leadTimeDays = (isNaN(v) || v < 1) ? 14 : v;
    leadtimeInput.value = leadTimeDays;
    renderAll();
  });'''
LEADTIME_NEW = '''  leadtimeInput.addEventListener("change", function () {
    var v = parseInt(leadtimeInput.value, 10);
    leadTimeDays = (isNaN(v) || v < 1) ? 14 : v;
    leadtimeInput.value = leadTimeDays;
    /* === API-INTEGRATION === */
    if (usingServer) { refreshFromServer(); return; }
    renderAll();
  });'''

# --- 7. Разметка панели -------------------------------------------------------
HTML_ANCHOR = '''    <div class="stats-row">'''
HTML_CODE = '''    <!-- === API-INTEGRATION === -->
    <div id="server-status" class="server-status offline"></div>
    <div id="clean-report" class="panel clean-report" style="display:none"></div>

    <div class="stats-row">'''

# --- 8. Стили -----------------------------------------------------------------
CSS_ANCHOR = "</style>"
CSS_CODE = '''
  /* === API-INTEGRATION === */
  .server-status {
    font-size: 13px; padding: 7px 12px; border-radius: 7px;
    margin-bottom: 12px; display: inline-block;
  }
  .server-status.online { background: #E8F3EC; color: #1F5F3F; }
  .server-status.offline { background: #FBF0E4; color: #8A5A22; }
  .server-status.connecting { background: #EDF1F5; color: #44586B; }

  .clean-report { padding: 16px 18px; margin-bottom: 16px; }
  .clean-head { font-weight: 600; margin-bottom: 12px; color: #2A3B47; }
  .clean-grid { display: flex; flex-wrap: wrap; gap: 22px; }
  .clean-item { display: flex; flex-direction: column; }
  .clean-item .n { font-size: 19px; font-weight: 600; color: #35566B; }
  .clean-item .l { font-size: 12px; color: #6B7C88; margin-top: 2px; }
  .clean-warn {
    margin: 12px 0 0; padding-left: 18px; font-size: 13px;
    color: #8A5A22; line-height: 1.5;
  }
  .clean-log {
    display: inline-block; margin-top: 12px; font-size: 13px;
    color: #35566B;
  }

  .mode-badge {
    display: inline-block; margin-left: 8px; padding: 2px 7px;
    border-radius: 4px; font-size: 11px; font-weight: 600;
    vertical-align: middle; cursor: help;
  }
  .badge-ml { background: #E3EDF3; color: #2F5468; }
  .badge-stats { background: #EDF1F5; color: #5A6B78; }
  .badge-minmax { background: #F6EBDC; color: #8A5A22; }
</style>'''


def apply(text: str) -> tuple[str, list[str]]:
    applied: list[str] = []

    def once(src: str, old: str, new: str, name: str) -> str:
        if old not in src:
            print(f"  ПРОПУЩЕНО: {name} — якорь не найден", file=sys.stderr)
            return src
        applied.append(name)
        return src.replace(old, new, 1)

    text = once(text, STATE_ANCHOR, STATE_CODE, "состояние API")
    text = once(text, CLIENT_ANCHOR, CLIENT_CODE + CLIENT_ANCHOR, "клиент API")
    text = once(text, COMPUTE_OLD, COMPUTE_NEW, "computeStats")
    text = once(text, ROUTE_OLD, ROUTE_NEW, "routeFile")
    text = once(text, PANEL_ANCHOR, PANEL_CODE + PANEL_ANCHOR, "панель отчёта")
    text = once(text, TABLE_OLD, TABLE_NEW, "бейдж режима")
    text = once(text, HORIZON_OLD, HORIZON_NEW, "перезапрос при смене горизонта")
    text = once(text, LEADTIME_OLD, LEADTIME_NEW, "перезапрос при смене срока поставки")
    text = once(text, HTML_ANCHOR, HTML_CODE, "разметка панели")
    text = once(text, CSS_ANCHOR, CSS_CODE, "стили")
    return text, applied


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Не найден {SRC}")
    text = SRC.read_text(encoding="utf-8")

    if MARKER in text:
        print("Сайт уже подключён к модели — ничего не меняю.")
        return

    text, applied = apply(text)
    SRC.write_text(text, encoding="utf-8")
    print(f"Применено правок: {len(applied)}")
    for a in applied:
        print(f"  + {a}")


if __name__ == "__main__":
    main()
