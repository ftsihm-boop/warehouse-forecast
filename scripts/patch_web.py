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
  var modelFit = null;         // проверка модели на данных пользователя
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
    modelFit = out.model_fit || null;
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
      // Если модель дообучилась под эти данные — об этом стоит сказать:
      // пользователь видит, что система подстроилась, а не работает
      // усреднённым прогнозом по чужому ассортименту.
      if (modelFit && modelFit.action === "fine_tuned") {
        el.textContent = "Модель дообучена под ваши данные";
      } else if (modelFit && modelFit.checked && !modelFit.suitable) {
        el.className = "server-status warn";
        el.textContent = "Модель слабо подходит к этим данным — "
          + "обучите её на этом файле";
      } else {
        el.textContent = "Прогноз строит обученная модель";
      }
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
    renderImpactPanel();

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

    if (modelFit && modelFit.message) {
      var cls = modelFit.action === "fine_tuned" ? "fit-tuned"
              : (modelFit.suitable ? "fit-ok" : "fit-warn");
      html += '<div class="model-fit ' + cls + '">'
        + escapeHtml(modelFit.message) + '</div>';
    }

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

  // Экономический эффект: снижение затрат на хранение и предотвращённые
  // потери от дефицита. Считается бэктестом на истории, это занимает
  // около полуминуты, поэтому запускается по кнопке, а не автоматически.
  async function loadEconomicImpact() {
    var panel = document.getElementById("impact-body");
    if (!panel || !sessionId) return;

    var sel = document.getElementById("impact-scope");
    var wide = sel && (sel.value === "all" || parseInt(sel.value, 10) > 10);
    panel.innerHTML = '<div class="impact-loading">Считаю эффект на вашей '
      + 'истории продаж' + (wide ? ' по расширенному охвату — это может занять '
      + 'минуту и дольше' : ' — это занимает около 30 секунд') + '...</div>';

    try {
      var scopeSel = document.getElementById("impact-scope");
      var scope = scopeSel ? scopeSel.value : "10";
      var res = await fetch(apiUrl("/economic-impact", {
        session_id: sessionId, lead_time_days: leadTimeDays, scope: scope
      }));
      if (!res.ok) throw new Error("Сервер вернул " + res.status);
      var d = await res.json();
      if (d.error) { panel.innerHTML = '<div class="impact-loading">' + escapeHtml(d.error) + '</div>'; return; }

      var pctText = function (v) {
        return (v === null || v === undefined) ? "—" : fmt(v, 1) + " %";
      };
      var money = function (v) { return fmt(v, 0) + " \\u20BD"; };

      var html = '<div class="impact-cards">';
      html += '<div class="impact-card good">'
        + '<div class="ic-num">' + pctText(d.holding_saved_pct) + '</div>'
        + '<div class="ic-lab">снижение затрат на хранение</div>'
        + '<div class="ic-sub">' + money(d.holding_before) + ' → '
        + money(d.holding_after) + '</div></div>';

      html += '<div class="impact-card good">'
        + '<div class="ic-num">' + pctText(d.lost_prevented_pct) + '</div>'
        + '<div class="ic-lab">предотвращённые потери от дефицита</div>'
        + '<div class="ic-sub">' + money(d.lost_before) + ' → '
        + money(d.lost_after) + '</div></div>';

      html += '<div class="impact-card">'
        + '<div class="ic-num">' + pctText(d.stock_reduced_pct) + '</div>'
        + '<div class="ic-lab">снижение товарного остатка</div>'
        + '<div class="ic-sub">' + money(d.stock_before) + ' → '
        + money(d.stock_after) + '</div></div>';

      html += '<div class="impact-card accent">'
        + '<div class="ic-num">' + money(d.effect_per_month) + '</div>'
        + '<div class="ic-lab">совокупный эффект в месяц</div>'
        + '<div class="ic-sub">за период ' + money(d.total_effect_period) + '</div></div>';
      html += '</div>';

      var diag = d.diagnosis || {};
      if (diag.notes && diag.notes.length) {
        html += '<div class="impact-diag ' + (diag.level || 'ok') + '">';
        diag.notes.forEach(function (n) {
          html += '<div class="diag-line">' + escapeHtml(n) + '</div>';
        });
        html += '</div>';
      }

      html += '<p class="impact-note">Расчёт сделан бэктестом: по '
        + d.skus_analyzed + ' товарам из ' + d.skus_total_eligible
        + ' пригодных, за последние '
        + d.period_days + ' дней история прогнана дважды — как если бы '
        + 'закупками управляли вручную (средний спрос с запасом прочности) '
        + 'и как это делает система. Закупочная цена ' + money(d.unit_cost)
        + ', наценка ' + fmt(d.margin * 100, 1) + ' % — '
        + (d.prices_from_file === 0 ? 'оценка' : 'из вашего файла') + '. '
        + 'Страховой запас × ' + d.service_factor
        + (d.theoretical_factor ? ' (теоретический оптимум × ' + d.theoretical_factor + ')' : '')
        + '.</p>';

      panel.innerHTML = html;
    } catch (e) {
      panel.innerHTML = '<div class="impact-loading">Не удалось рассчитать: '
        + escapeHtml(e.message) + '</div>';
    }
  }

  function renderImpactPanel() {
    var panel = document.getElementById("impact-panel");
    if (!panel) return;
    panel.style.display = usingServer ? "block" : "none";
    var btn = document.getElementById("impact-btn");
    if (btn && !btn.dataset.bound) {
      btn.dataset.bound = "1";
      btn.addEventListener("click", function () {
        btn.disabled = true;
        loadEconomicImpact().finally(function () { btn.disabled = false; });
      });
    }
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

# --- 6в. Подсказка про средний спрос ------------------------------------------
# У серверного расчёта нет понятия «последние N дней»: модель смотрит
# на всю историю и взвешивает её сама. Старый текст подставлял туда null.
TOOLTIP_OLD = '''      var avgTooltip = tf("avgTooltip", { days: r.stats.spanDays, avg: fmt(r.stats.avgDaily, 1) });'''
TOOLTIP_NEW = '''      /* === API-INTEGRATION === */
      var avgTooltip;
      if (usingServer && r.stats.spanDays === null) {
        avgTooltip = "Средний дневной спрос по прогнозу модели: "
          + fmt(r.stats.avgDaily, 1) + " шт/день."
          + (r.stats.forecastHigh
              ? " Прогноз на горизонт: " + fmt(r.stats.forecastHorizon, 0)
                + " шт, верхняя граница " + fmt(r.stats.forecastHigh, 0) + " шт."
              : "")
          + (r.stats.note ? " " + r.stats.note : "");
      } else {
        avgTooltip = tf("avgTooltip", { days: r.stats.spanDays, avg: fmt(r.stats.avgDaily, 1) });
      }'''

# --- 6г. Симметричные поля у мини-графиков ------------------------------------
# В компактном режиме подписи оси Y не рисуются (см. `if (!compact)` ниже
# по коду), поэтому отступ в 30px слева оставался пустым — график
# выглядел сдвинутым вправо внутри карточки.
CHARTPAD_OLD = '''      ? { top: 8, right: 6, bottom: 18, left: 30 }'''
CHARTPAD_NEW = '''      ? { top: 8, right: 8, bottom: 18, left: 8 }'''

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
    <div id="impact-panel" class="panel impact-panel" style="display:none">
      <div class="impact-head">
        <div>
          <div class="impact-title">Экономический эффект</div>
          <div class="impact-hint">Сравнение с ручным планированием закупок на вашей истории продаж</div>
        </div>
        <div class="impact-controls">
          <label for="impact-scope">По товарам</label>
          <select id="impact-scope">
            <option value="10" selected>10 самых оборотистых</option>
            <option value="25">25 самых оборотистых</option>
            <option value="50">50 самых оборотистых</option>
            <option value="all">по всем товарам</option>
          </select>
          <button id="impact-btn" class="impact-btn">Рассчитать</button>
        </div>
      </div>
      <div id="impact-body"></div>
    </div>

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

  .server-status.warn { background: #FBF0E4; color: #8A5A22; }
  .model-fit {
    margin-top: 12px; padding: 9px 12px; border-radius: 7px;
    font-size: 12.5px; line-height: 1.5;
  }
  .model-fit.fit-ok { background: #EDF5EF; color: #2C5A3C; }
  .model-fit.fit-tuned { background: #E9F0F5; color: #2B4C63; }
  .model-fit.fit-warn { background: #FBF3E5; color: #7E5A1E; }
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

  /* Бейдж компактный: он добавляется в первую колонку, а таблица
     и без него занимала почти всю ширину — лишние пиксели тут
     выталкивают колонку «Статус» за край. */
  .mode-badge {
    display: inline-block; margin-left: 6px; padding: 1px 5px;
    border-radius: 4px; font-size: 10px; font-weight: 600;
    vertical-align: middle; cursor: help; letter-spacing: .1px;
  }
  /* и слегка поджимаем ячейки, чтобы таблица снова помещалась целиком */
  tbody td, thead th { padding-left: 11px; padding-right: 11px; }
  .badge-ml { background: #E3EDF3; color: #2F5468; }
  .badge-stats { background: #EDF1F5; color: #5A6B78; }
  .badge-minmax { background: #F6EBDC; color: #8A5A22; }

  .impact-panel { padding: 16px 18px; margin-bottom: 16px; }
  .impact-head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; flex-wrap: wrap;
  }
  .impact-title { font-weight: 600; color: #2A3B47; }
  .impact-hint { font-size: 12px; color: #6B7C88; margin-top: 2px; }
  .impact-btn {
    padding: 8px 16px; border-radius: 7px; border: 1px solid #35566B;
    background: #35566B; color: #fff; font-size: 13px; cursor: pointer;
  }
  .impact-btn:hover { background: #2A4759; }
  .impact-btn:disabled { opacity: .55; cursor: default; }
  .impact-loading { font-size: 13px; color: #6B7C88; padding: 14px 0 4px; }
  .impact-cards {
    display: flex; flex-wrap: wrap; gap: 14px; margin-top: 16px;
  }
  .impact-card {
    flex: 1 1 170px; padding: 13px 15px; border-radius: 8px;
    background: #F4F6F8; border: 1px solid #E3E8EC;
  }
  .impact-card.good { background: #EDF5EF; border-color: #D2E5D8; }
  .impact-card.accent { background: #F7EFE4; border-color: #EBD9C2; }
  .impact-card .ic-num { font-size: 21px; font-weight: 600; color: #2A3B47; }
  .impact-card .ic-lab { font-size: 12px; color: #55666F; margin-top: 3px; line-height: 1.35; }
  .impact-card .ic-sub { font-size: 11px; color: #8394A0; margin-top: 6px; }
  .impact-controls { display: flex; align-items: center; gap: 10px; }
  .impact-controls label { font-size: 12px; color: #6B7C88; }
  .impact-controls select {
    padding: 7px 9px; border-radius: 7px; border: 1px solid #D3DBE0;
    font-size: 13px; background: #fff; color: #2A3B47;
  }
  .impact-diag {
    margin-top: 14px; padding: 11px 13px; border-radius: 8px; font-size: 12.5px;
    line-height: 1.5;
  }
  .impact-diag.ok { background: #EDF5EF; color: #2C5A3C; }
  .impact-diag.warn { background: #FBF3E5; color: #7E5A1E; }
  .impact-diag.bad { background: #F8ECEA; color: #8A3A2A; }
  .impact-diag .diag-line + .diag-line { margin-top: 7px; }
  .impact-note {
    font-size: 12px; color: #6B7C88; line-height: 1.55; margin: 14px 0 0;
  }
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
    text = once(text, TOOLTIP_OLD, TOOLTIP_NEW, "подсказка среднего спроса")
    text = once(text, CHARTPAD_OLD, CHARTPAD_NEW, "поля мини-графиков")
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
