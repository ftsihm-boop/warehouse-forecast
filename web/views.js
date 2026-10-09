/* Разделы «Прогноз», «Эффект» и «Данные». */
"use strict";

const WEEKDAY_FULL = { пн: "понедельникам", вт: "вторникам", ср: "средам", чт: "четвергам", пт: "пятницам", сб: "субботам", вс: "воскресеньям" };
const LEVEL_PILL = { excellent: "green", good: "brand", low: "red", unknown: "gray", none: "gray" };

/* ============================================================
   Прогноз по товару
   ============================================================ */
VIEWS.forecast = function () {
  const skus = S.session.skus;
  if (!S.fcSku || !skus.includes(S.fcSku)) S.fcSku = (S.plan && S.plan.items[0] && S.plan.items[0].sku) || skus[0];
  const ranges = [[90, "3 месяца"], [180, "Полгода"], [365, "Год"], [0, "Всё время"]];
  const smooth = S.smoothUser ?? (S.range === 0 || S.range >= 365);
  put($app(), h`
    <div class="busy-wrap" data-busy="fc">${veil()}
      <div class="page-head"><div>
        <h1 class="h1">Продажи и прогноз</h1>
        <p class="lead">Тёмная линия — сколько продавали. Зелёная — сколько ждём. Бледная полоса показывает, до скольких продажи могут вырасти при высоком спросе.</p>
      </div></div>
      <section class="panel">
        <div class="fc-head">
          <div class="field"><label for="fcSku">Товар</label>
            <select class="input" id="fcSku" data-in="fc_sku">${skus.map((s) => h`<option value="${s}" ${s === S.fcSku ? raw("selected") : ""}>${s}</option>`)}</select></div>
          <div class="fc-ctrl">
            <div class="seg" role="group" aria-label="Период на графике">${ranges.map(([d, l]) => h`<button type="button" data-act="range" data-d="${d}" aria-pressed="${String(S.range === d)}">${l}</button>`)}</div>
            <div class="seg" role="group" aria-label="Сглаживание">
              <button type="button" data-act="smooth" data-v="0" aria-pressed="${String(!smooth)}">По дням</button>
              <button type="button" data-act="smooth" data-v="1" aria-pressed="${String(smooth)}">Среднее за неделю</button></div>
            <label class="check"><input type="checkbox" data-in="fc_prev" ${S.prevYear ? raw("checked") : ""}> Сравнить с прошлым годом</label>
          </div>
        </div>
        <div class="chart" id="fcChart" style="min-height:260px"></div>
        <div class="legend">
          <span><i style="background:var(--ink)"></i>продажи</span>
          <span><i style="background:var(--brand)"></i>прогноз</span>
          <span><i class="band" style="background:var(--brand)"></i>если спрос высокий</span>
          <span id="lgPrev" ${S.prevYear ? "" : raw("hidden")}><i style="background:var(--ink-3)"></i>тот же период год назад</span>
          <span><i style="background:var(--red);height:9px;width:3px"></i>дни, когда товара не было</span>
        </div>
      </section>
      <section class="panel divide" id="fcStats" style="margin-top:16px"></section>
    </div>`);
  loadFc();
};

async function loadFc() {
  const horizon = (S.plan && S.plan.params.horizon_used) || S.set.horizon;
  const d = S.fcData;
  if (d && d.sku === S.fcSku && d.horizon === horizon) return paintFc();
  setBusy("fc", "Загружаю продажи и прогноз…");
  try {
    const r = await api("/sku?" + qs({ session_id: S.sid, sku: S.fcSku, horizon }));
    if (S.tab !== "forecast" || r.sku !== S.fcSku) return;
    S.fcData = r;
    paintFc();
  } catch (e) { fail(e); }
  finally { setBusy("fc", null); }
}

function paintFc() {
  const d = S.fcData, host = $("#fcChart");
  if (!d || !host) return;
  const smooth = S.smoothUser ?? (S.range === 0 || S.range >= 365);
  renderForecastChart(host, { hist: d.history, fc: d.forecast, rangeDays: S.range, prevYear: S.prevYear, smooth });
  const f = d.features || {}, it = S.plan && S.plan.items.find((i) => i.sku === d.sku);
  const sum50 = d.forecast.reduce((a, r) => a + r.q50, 0), sum90 = d.forecast.reduce((a, r) => a + r.q90, 0);
  const stats = [];
  stats.push(h`<div class="stat"><b class="num">${fNum(f.per_day_recent, f.per_day_recent < 10 ? 1 : 0)} шт</b><span>продаётся в день за последние 4 недели</span></div>`);
  if (d.forecast.length) stats.push(h`<div class="stat"><b class="num">${fInt(sum50)} шт</b><span>ждём за ${fDays(d.horizon)}${sum90 > sum50 ? `, при высоком спросе — до ${fInt(sum90)}` : ""}</span></div>`);
  else stats.push(h`<div class="stat"><b>—</b><span>по этому товару прогноз не строится: продажи слишком нерегулярны</span></div>`);
  if (f.trend_28 != null) stats.push(h`<div class="stat"><b class="num">${fSign(f.trend_28 * 100)}${NB}%</b><span>к предыдущим 4 неделям</span></div>`);
  if (f.peak_weekday && f.weekday_strength >= 0.15) stats.push(h`<div class="stat"><b>${f.peak_weekday}</b><span>лучший день недели; слабее всего — ${f.low_weekday}</span></div>`);
  if (it && it.stock_known) stats.push(h`<div class="stat"><b class="num">${fDays(Math.round(Math.max(it.days_of_supply, 0)))}</b><span>хватит остатка · <button class="link" type="button" data-act="gosku" data-sku="${d.sku}" data-to="plan">к заказу</button></span></div>`);
  const notes = (f.notes || []);
  const acc = d.accuracy;
  put($("#fcStats"), h`
    <div class="stats">${stats}</div>
    ${notes.length || acc ? h`<div style="padding:16px 22px;display:grid;gap:8px;font-size:14.5px;color:var(--ink-2)">
      ${acc ? h`<p>Точность прогноза по этому товару: <span class="pill ${LEVEL_PILL[acc.level]}">${acc.level_label}</span> ${acc.short}</p>` : ""}
      ${notes.map((n) => h`<p>${n}</p>`)}</div>` : ""}`);
}

Object.assign(VIEW_ACTIONS, {
  range: (t) => { S.range = +t.dataset.d; S.smoothUser = null; save(); VIEWS.forecast(); },
  smooth: (t) => { S.smoothUser = t.dataset.v === "1"; VIEWS.forecast(); },
});
Object.assign(VIEW_INPUTS, {
  fc_sku: (t) => { S.fcSku = t.value; save(); loadFc(); },
  fc_prev: (t) => { S.prevYear = t.checked; save(); const l = $("#lgPrev"); if (l) l.hidden = !S.prevYear; paintFc(); },
});

/* ============================================================
   Эффект
   ============================================================ */
VIEWS.effect = function () {
  const I = S.impact;
  put($app(), h`
    <div class="busy-wrap" data-busy="impact">${veil()}
      <div class="page-head"><div>
        <h1 class="h1">Сколько это даёт</h1>
        <p class="lead">Мы «переиграли» последние три месяца дважды: как будто вы закупали по старой привычке и как будто — по подсказкам системы. Разница и есть выгода.</p>
      </div>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <select class="input" id="impScope" data-in="imp_scope" style="width:auto" aria-label="Сколько товаров проверять">
          <option value="10" ${S.impScope !== "all" ? raw("selected") : ""}>10 самых ходовых товаров</option>
          <option value="all" ${S.impScope === "all" ? raw("selected") : ""}>Все подходящие товары</option>
        </select>
        <button class="btn" type="button" data-act="impact">${I ? "Пересчитать" : "Посчитать выгоду"}</button>
      </div></div>
      <div id="impBody">${I ? "" : h`<div class="panel"><div class="empty"><b>Здесь появится расчёт</b>Нажмите «Посчитать выгоду». Это займёт от нескольких секунд до минуты: система проигрывает историю заново.</div></div>`}</div>
    </div>`);
  if (I) paintImpact();
};

function paintImpact() {
  const I = S.impact, host = $("#impBody");
  if (!I || !host) return;
  if (I.error) { put(host, h`<div class="note bad"><span class="ic">!</span><span>${I.error}</span></div>`); return; }
  const good = I.beneficial !== false && I.effect_per_month > 0;
  const cmp = (title, before, after, fmt, hint, floor = 1) => {
    const mx = Math.max(before, after, floor, 1);
    return h`<div class="cmp-row"><div class="lab"><b>${title}</b><span class="muted small">${hint}</span></div>
      <div class="cmp-val"><span>Было (по привычке): ${fmt(before)}</span></div><div class="cmp-bar before"><i style="width:${before / mx * 100}%"></i></div>
      <div class="cmp-val"><span>Стало (по системе): ${fmt(after)}</span></div><div class="cmp-bar after"><i style="width:${after / mx * 100}%"></i></div></div>`;
  };
  put(host, h`
    <section class="panel effect-hero">
      ${good ? h`<div class="big num">${fRub(I.effect_per_month)} <span style="font-size:.4em;font-weight:500;color:var(--ink-2)">в месяц</span></div>
        <p class="lead">Столько система экономит на ${I.skus_managed} ${plural(I.skus_managed, "товаре", "товарах", "товарах")}: меньше денег лежит на складе и меньше продаж теряется из-за пустых полок. За три месяца проверки — ${fRub(I.total_effect_period)}.</p>`
        : h`<div class="big">Выгоды пока нет</div><p class="lead">На этих данных система не обошла ручное планирование. Ниже — почему.</p>`}
    </section>
    <section class="panel" style="margin-top:16px"><div class="cmp">
      ${cmp("Деньги, замороженные в товаре", I.stock_before, I.stock_after, fRub, I.stock_reduced_pct != null ? `на ${fPct(I.stock_reduced_pct, 1)} меньше` : "")}
      ${cmp("Расходы на хранение за период", I.holding_before, I.holding_after, fRub, I.holding_saved_pct != null ? `на ${fPct(I.holding_saved_pct, 1)} меньше` : "")}
      ${cmp("Упущенная выручка из-за дефицита", I.lost_before, I.lost_after, fRub, I.lost_prevented > 0 ? `сохранено ${fRub(I.lost_prevented)}` : "", I.holding_before)}
    </div></section>
    ${I.manual_skus && I.manual_skus.length ? h`<div class="note warn" style="margin-top:16px"><span class="ic">!</span><span>Для ${I.skus_manual} ${plural(I.skus_manual, "товара", "товаров", "товаров")} система не советует полагаться на прогноз — лучше заказывать по-старому: ${I.manual_skus.map((s) => "«" + s + "»").join(", ")}.</span></div>` : ""}
    ${(I.diagnosis && I.diagnosis.notes || []).map((n) => h`<div class="note" style="margin-top:12px"><span class="ic">i</span><span>${n}</span></div>`)}
    <details class="panel panel-pad" style="margin-top:16px"><summary style="cursor:pointer;font-weight:500">Как это посчитано</summary>
      <div class="small" style="color:var(--ink-2);margin-top:12px;display:grid;gap:8px;max-width:70ch">
        <p>Проверено ${I.skus_analyzed} из ${I.skus_total_eligible} подходящих товаров за последние ${I.period_days} дней.</p>
        <p>«По привычке» — заказ по средним продажам за 4 недели с запасом в четверть. «По системе» — прогноз спроса плюс страховой запас, который подбирается отдельно под каждый товар.</p>
        <p>Закупочная цена и наценка взяты из вашего файла: наценка ≈ ${fPct((I.margin || 0) * 100, 0)}.</p>
      </div></details>`);
}

Object.assign(VIEW_ACTIONS, {
  impact: async () => {
    setBusy("impact", "Проигрываю последние три месяца — это может занять до минуты…");
    try {
      S.impact = await api("/economic-impact?" + qs({ session_id: S.sid, scope: S.impScope === "all" ? "all" : "10", lead_time_days: S.set.default_lead, review_period_days: S.set.review_period }));
      if (S.tab === "effect") VIEWS.effect();
    } catch (e) { fail(e); } finally { setBusy("impact", null); }
  },
});
Object.assign(VIEW_INPUTS, { imp_scope: (t) => { S.impScope = t.value; S.impact = null; VIEWS.effect(); } });

/* ============================================================
   Данные: о файле · признаки · аномалии · точность
   ============================================================ */
const DTABS = [["about", "О файле"], ["feat", "Признаки"], ["anom", "Аномалии"], ["acc", "Точность"]];
const DEND = { about: "dataset", feat: "features", anom: "anomalies", acc: "accuracy" };
const DKEY = { about: "dataset", feat: "features", anom: "anomalies", acc: "acc" };

VIEWS.data = function () {
  put($app(), h`
    <div class="busy-wrap" data-busy="data">${veil()}
      <div class="page-head"><div><h1 class="h1">Данные</h1>
        <p class="lead">Что система увидела в вашем файле, какие признаки по нему посчитала, что нашла подозрительного и насколько можно верить прогнозу.</p></div></div>
      <div class="seg subtabs" role="group" aria-label="Раздел данных">
        ${DTABS.map(([k, l]) => h`<button type="button" data-act="dtab" data-k="${k}" aria-pressed="${String(S.dtab === k)}">${l}${k === "anom" && S.session.attention_count ? h` <span class="dotcount amber" style="margin-left:4px">${S.session.attention_count}</span>` : ""}</button>`)}
      </div>
      <div id="dataBody"></div>
    </div>`);
  loadData();
};

async function loadData() {
  const k = S.dtab;
  if (!S[DKEY[k]]) {
    setBusy("data", "Загружаю…");
    try {
      const r = await api(`/${DEND[k]}?` + qs({ session_id: S.sid, horizon: k === "acc" ? 14 : null }));
      S[DKEY[k]] = r;
    } catch (e) { setBusy("data", null); return fail(e); }
    setBusy("data", null);
  }
  if (S.tab === "data" && S.dtab === k) paintData();
}

function paintData() {
  const host = $("#dataBody");
  if (!host) return;
  ({ about: paintAbout, feat: paintFeat, anom: paintAnom, acc: paintAcc }[S.dtab])(host);
}

/* ---- о файле ---- */
function paintAbout(host) {
  const D = S.dataset, o = D.overview;
  put(host, h`
    <section class="panel"><div class="kv divide" style="grid-auto-flow:row">
      <div><b class="num">${fInt(o.sku_count)}</b><span>${plural(o.sku_count, "товар", "товара", "товаров")} в файле</span></div>
      <div><b class="num">${o.date_min} – ${o.date_max}</b><span>период, ${fInt(o.days)} дн.</span></div>
      <div><b class="num">${fInt(o.rows_final)}</b><span>строк после очистки (в файле было ${fInt(o.rows_read)})</span></div>
      <div><b class="num">${fInt(o.total_qty)}</b><span>штук продано всего</span></div>
      ${o.revenue ? h`<div><b class="num">${fRub(o.revenue)}</b><span>выручка за период</span></div>` : ""}
    </div></section>
    <div class="note" style="margin-top:16px"><span class="ic">i</span><span>${D.history_note}</span></div>
    ${(D.notes || []).map((n) => h`<div class="note" style="margin-top:10px"><span class="ic">i</span><span>${n}</span></div>`)}
    <h2 class="h2" style="margin:28px 0 12px">Что мы нашли в колонках</h2>
    <section class="panel scroll-x"><table class="t"><thead><tr><th>Колонка</th><th>Зачем нужна</th><th>Заполнено</th><th>Значения</th></tr></thead><tbody>
      ${D.columns.map((c) => h`<tr><td><b>${c.title}</b> ${c.required ? h`<span class="pill gray">обязательна</span>` : ""}</td>
        <td style="max-width:420px;color:var(--ink-2)">${c.description}${!c.present && c.hint ? h`<br><span style="color:var(--amber)">${c.hint}</span>` : ""}</td>
        <td>${c.present ? h`<span class="fill"><i style="width:${c.fill_pct}%"></i></span>${c.fill_pct}${NB}%` : h`<span class="pill amber">не найдена</span>`}</td>
        <td class="num" style="color:var(--ink-2)">${c.range}</td></tr>`)}
    </tbody></table></section>
    <h2 class="h2" style="margin:28px 0 12px">Товары</h2>
    <section class="panel scroll-x"><table class="t"><thead><tr><th>Товар</th><th class="r">Продано всего</th><th class="r">В день</th><th class="r">Остаток сейчас</th><th>Вклад в продажи</th><th>Характер спроса</th></tr></thead><tbody>
      ${D.skus.map((s) => h`<tr><td>${s.sku}</td><td class="r num">${fInt(s.total)}</td><td class="r num">${fNum(s.per_day, s.per_day < 10 ? 1 : 0)}</td>
        <td class="r num">${s.stock == null ? "—" : fInt(s.stock)}</td>
        <td><span class="pill ${s.abc === "A" ? "brand" : "gray"}">${s.abc === "A" ? "высокий (A)" : s.abc === "B" ? "средний (B)" : "низкий (C)"}</span></td>
        <td style="color:var(--ink-2)">${s.demand_label}</td></tr>`)}
    </tbody></table></section>`);
}

/* ---- признаки ---- */
function paintFeat(host) {
  const F = S.features;
  const pc = (v) => v == null ? "—" : h`<span style="color:${v > 0.1 ? "var(--green)" : v < -0.1 ? "var(--red)" : "inherit"}">${fSign(v * 100)}${NB}%</span>`;
  put(host, h`
    <p class="lead" style="margin:-6px 0 18px;max-width:70ch">Признаки — это числа, которые система считает по продажам каждого товара. Часть из них она использует, чтобы строить прогноз; остальные нужны вам, чтобы понимать ассортимент.</p>
    <section class="panel"><div style="padding:16px 22px 4px"><h2 class="h2">По чему строится прогноз</h2>
      <p class="small muted" style="margin-top:4px">${F.model_feature_total} ${plural(F.model_feature_total, "признак", "признака", "признаков")} в ${F.model_features.length} группах — модель видит только форму продаж, а не название товара.</p></div>
      <div class="feat-groups">${F.model_features.map((g) => h`<div><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><b>${g.title}</b><span class="pill brand">${g.count}</span></div><p>${g.description}</p></div>`)}</div></section>
    <section class="panel" style="margin-top:16px"><div style="padding:16px 22px 4px"><h2 class="h2">Показатели для вас</h2>
      <p class="small muted" style="margin-top:4px">Считаются поверх ваших данных, чтобы объяснить ассортимент. На прогноз напрямую не влияют.</p></div>
      <div class="feat-groups">${F.derived.map((g) => h`<div><div style="display:flex;gap:8px;align-items:center"><b>${g.title}</b><span class="pill gray">для анализа</span></div><p>${g.description}</p></div>`)}</div></section>
    <h2 class="h2" style="margin:28px 0 12px">По каждому товару</h2>
    <section class="panel scroll-x"><table class="t"><thead><tr>
      <th>Товар</th><th title="Вклад в деньги: A — главные товары">ABC</th><th title="Стабильность: X — ровно, Z — трудно предсказать">XYZ</th>
      <th class="r">Тренд за месяц</th><th class="r">К прошлому году</th><th>Лучший день</th><th class="r">Без товара</th><th class="r">Хватит на</th><th class="r">Маржа</th><th>Что заметили</th></tr></thead><tbody>
      ${F.table.map((r) => h`<tr><td style="min-width:200px">${r.sku}</td>
        <td><span class="pill ${(r.abc_money || r.abc_units) === "A" ? "brand" : "gray"}">${r.abc_money || r.abc_units}</span></td>
        <td title="${F.xyz_labels[r.xyz] || ""}"><span class="pill gray">${r.xyz}</span></td>
        <td class="r num">${pc(r.trend_28)}</td><td class="r num">${pc(r.yoy_28)}</td>
        <td>${r.peak_weekday || "—"}</td>
        <td class="r num">${r.stockout_share != null ? fPct(r.stockout_share * 100, 0) : "—"}</td>
        <td class="r num">${r.stock_days != null ? fDays(Math.round(r.stock_days)) : "—"}</td>
        <td class="r num">${r.margin_pct != null ? fPct(r.margin_pct * 100, 0) : "—"}</td>
        <td style="color:var(--ink-2);min-width:240px;font-size:13.5px">${(r.notes || []).join(" ") || "—"}</td></tr>`)}
    </tbody></table></section>
    <p class="tiny muted" style="margin-top:10px">ABC: A — товары, дающие 80 % продаж, B — следующие 15 %, C — остальное. XYZ: X — спрос ровный, Y — заметные колебания, Z — трудно предсказать.</p>`);
}

/* ---- аномалии ---- */
const UNITS = { "товаров": ["товар", "товара", "товаров"], "дней": ["день", "дня", "дней"], "строк": ["строка", "строки", "строк"],
  "названий": ["название", "названия", "названий"], "случаев": ["случай", "случая", "случаев"] };
const unitWord = (u, n) => UNITS[u] ? plural(n, ...UNITS[u]) : u;

function paintAnom(host) {
  const A = S.anomalies;
  const pill = { fixed: "green", attention: "amber", info: "gray" };
  put(host, h`
    <section class="panel health ${A.health}"><i class="dot"></i><div><b style="font-size:17px">${A.health_label}</b>
      <div class="small" style="color:var(--ink-2)">${A.headline}${A.totals.rows_touched ? ` Затронуто ${fInt(A.totals.rows_touched)} строк (${fNum(A.totals.rows_touched_pct, 1)} %).` : ""}</div></div></section>
    ${A.groups.length ? h`<div style="display:grid;gap:12px;margin-top:16px">${A.groups.map((g) => h`
      <details class="panel an" ${g.severity === "attention" ? raw("open") : ""}>
        <summary><span class="an-title">${g.title}<span class="pill ${pill[g.severity]}">${g.severity_label}</span></span>
          <span class="muted small num" style="white-space:nowrap">${fInt(g.count)} ${unitWord(g.unit, g.count)}</span></summary>
        <div class="what">
          <p><b>Что это.</b> ${g.what}</p>
          <p><b>Чем мешает.</b> ${g.why}</p>
          <p><b>${g.severity === "fixed" ? "Что сделали." : "Что делать."}</b> ${g.action}</p>
        </div>
        ${g.examples.length ? h`<div class="ex scroll-x"><table class="t"><thead><tr><th>Товар</th><th>Когда</th><th>Подробности</th></tr></thead><tbody>
          ${g.examples.map((e) => h`<tr><td>${e.sku}</td><td class="num">${e.date || "—"}</td><td style="color:var(--ink-2)">${e.detail}</td></tr>`)}
        </tbody></table></div>${g.count > g.examples.length ? h`<p class="tiny muted" style="margin-top:8px">Показаны первые ${g.examples.length}${g.skus_total > 1 ? `; всего затронуто товаров: ${g.skus_total}` : ""}.</p>` : ""}` : ""}
      </details>`)}</div>` : h`<div class="panel" style="margin-top:16px"><div class="empty"><b>Аномалий не найдено</b>Данные чистые.</div></div>`}`);
}

/* ---- точность ---- */
function paintAcc(host) {
  const A = S.acc;
  const meter = A.overall === "excellent" || A.overall === "good" || A.overall === "low" ? A.overall : "";
  const m = (i) => i.metrics;
  put(host, h`
    <section class="panel acc-big">
      <span class="muted small">Точность прогноза</span>
      <div class="word ${meter}">${A.overall_label}</div>
      <p style="color:var(--ink-2);max-width:60ch">${A.short}</p>
      <div class="acc-meter ${meter}" aria-hidden="true"><i></i><i></i><i></i></div>
      <div class="chips" style="margin-top:14px">
        ${[["excellent", "отличная", "green"], ["good", "хорошая", "brand"], ["low", "низкая", "red"], ["none", "не прогнозируется", "gray"], ["unknown", "мало данных", "gray"]]
          .filter(([k]) => A.counts[k]).map(([k, l, c]) => h`<span class="pill ${c}">${l}: ${A.counts[k]}</span>`)}
      </div>
      <details style="margin-top:12px"><summary style="cursor:pointer;color:var(--brand-ink);font-weight:500">Подробнее: как мы это проверяли</summary>
        <ul class="small" style="color:var(--ink-2);margin:10px 0 0;padding-left:18px;display:grid;gap:6px;max-width:70ch">${A.notes.map((n) => h`<li>${n}</li>`)}</ul></details>
    </section>
    <h2 class="h2" style="margin:28px 0 12px">По каждому товару</h2>
    <section class="panel divide">${A.items.map((i) => h`
      <details class="acc-row"><summary><span style="min-width:0"><b style="font-weight:500">${i.sku}</b></span>
        <span style="display:flex;align-items:center;gap:12px"><span class="tiny muted">подробнее</span><span class="pill ${LEVEL_PILL[i.level]}">${i.level_label}</span></span></summary>
        <div class="body"><p>${i.short}</p>
          ${m(i) ? h`<ul>${i.notes.map((n) => h`<li>${n}</li>`)}</ul>
            <p class="tiny muted" style="margin-top:8px">Проверок: ${m(i).checks} · ошибка ${fNum(m(i).wape, 1)} % · перекос ${fSign(m(i).bias, 1)} % · страховой запас хватало в ${fNum(m(i).q90_coverage * 100, 0)} % случаев</p>` : ""}
        </div></details>`)}</section>`);
}

Object.assign(VIEW_ACTIONS, {
  dtab: (t) => { S.dtab = t.dataset.k; save(); VIEWS.data(); },
});
