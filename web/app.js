/* Склад — заказы по прогнозу. Ядро: состояние, загрузка файла, восстановление, раздел «Что заказать». */
"use strict";

const S = {
  sid: null, session: null, health: null,
  tab: "plan", dtab: "about",
  set: { horizon: 30, default_lead: 7, review_period: 7, lead_overrides: {}, in_transit: {}, stock_overrides: {} },
  fileKey: null,
  plan: null, filter: "all", q: "", open: new Set(), limit: 100,
  acc: null, dataset: null, features: null, anomalies: null, impact: null,
  fcSku: null, fcData: null, range: 365, prevYear: false, smoothUser: null,
  busy: {}, seenHorizons: new Set(),
};
const VIEWS = {};
const VIEW_ACTIONS = {};   // клики, которые добавляют остальные разделы
const VIEW_INPUTS = {};    // изменения полей в остальных разделах
const $app = () => $("#app");

/* ---------- сохранение между перезагрузками ---------- */
function save() {
  lsSet({ sid: S.sid, tab: S.tab, dtab: S.dtab, set: S.set, fileKey: S.fileKey, fcSku: S.fcSku, range: S.range, prevYear: S.prevYear });
}
function load() {
  const o = lsGet();
  if (o.set) S.set = { ...S.set, ...o.set, lead_overrides: o.set.lead_overrides || {}, in_transit: o.set.in_transit || {}, stock_overrides: o.set.stock_overrides || {} };
  S.sid = o.sid || null;
  S.tab = VIEWS_ORDER.some((t) => t.id === o.tab) ? o.tab : "plan";
  S.dtab = o.dtab || "about";
  S.fileKey = o.fileKey || null;
  S.fcSku = o.fcSku || null;
  if (o.range != null) S.range = o.range;
  S.prevYear = !!o.prevYear;
}

/* ---------- индикаторы загрузки ---------- */
function setBusy(key, text) {
  if (text) S.busy[key] = text; else delete S.busy[key];
  paintBusy();
}
function paintBusy() {
  $("#progress").classList.toggle("on", Object.keys(S.busy).length > 0);
  $$("[data-busy]").forEach((el) => {
    const t = S.busy[el.dataset.busy];
    el.classList.toggle("is-busy", !!t);
    const s = $(".busy-card span", el);
    if (s && t) s.textContent = t;
  });
}
const veil = () => h`<div class="busy-veil" aria-hidden="true"><div class="busy-card"><i class="spin"></i><span></span></div></div>`;

function toast(msg, bad) {
  const t = document.createElement("div");
  t.className = "toast" + (bad ? " bad" : "");
  t.setAttribute("role", bad ? "alert" : "status");
  t.textContent = msg;
  $("#toasts").append(t);
  setTimeout(() => t.remove(), bad ? 7000 : 3500);
}
const fail = (e) => { if (e && e.name === "AbortError") return; console.error(e); toast(e.message || "Что-то пошло не так.", true); };

/* ---------- каркас: вкладки ---------- */
const VIEWS_ORDER = [
  { id: "plan", label: "Что заказать" },
  { id: "recs", label: "Советы" },
  { id: "forecast", label: "Прогноз" },
  { id: "effect", label: "Эффект" },
  { id: "data", label: "Данные" },
];

function paintTabs() {
  const urgent = S.plan ? S.plan.recommendations.items.filter((r) => r.group === "urgent").length : 0;
  const attn = S.session ? S.session.attention_count : 0;
  put($("#tabs"), h`${VIEWS_ORDER.map((t) => h`
    <button class="tab" role="tab" data-act="tab" data-tab="${t.id}" aria-selected="${String(S.tab === t.id)}">${t.label}
      ${t.id === "recs" && urgent ? h`<span class="dotcount" title="Срочных советов">${urgent}</span>` : ""}
      ${t.id === "data" && attn ? h`<span class="dotcount amber" title="Что проверить в данных">${attn}</span>` : ""}
    </button>`)}`);
}

function go(tab, extra) {
  S.tab = tab;
  if (extra) Object.assign(S, extra);
  save();
  paintTabs();
  renderTab();
  window.scrollTo({ top: 0 });
}
function renderTab() {
  const v = VIEWS[S.tab];
  if (v) v();
  paintBusy();
}

/* ---------- экран загрузки ---------- */
function showUpload(msg) {
  $("#tabs").classList.add("hidden");
  $("#topFile").classList.add("hidden");
  const demo = S.health && S.health.demo;
  put($app(), h`
    <section class="upload">
      <h1 class="h1">Загрузите продажи — получите список заказа</h1>
      <p class="lead">Система изучит историю продаж и подскажет, какие товары и когда заказать, чтобы не остаться без них и не завалить склад лишним.</p>
      ${msg ? h`<div class="note bad" style="margin-top:20px" role="alert"><span class="ic">!</span><span>${msg}</span></div>` : ""}
      <div class="drop" id="drop" tabindex="0" role="button" data-act="pick" aria-label="Выбрать файл с продажами">
        <b>Перетащите файл сюда</b>
        <span>или нажмите, чтобы выбрать. Подойдёт Excel или CSV из кассы / 1С.</span>
        <div class="drop-actions">
          <button class="btn" type="button" data-act="pick">Выбрать файл</button>
          ${demo ? h`<button class="btn ghost" type="button" data-act="demo">Попробовать на примере</button>` : ""}
        </div>
      </div>
      <dl class="need">
        <div><dt>Обязательно</dt><dd>Дата, название товара, сколько продано за день.</dd></div>
        <div><dt>Желательно</dt><dd>Остаток на складе, цена продажи, закупочная цена.</dd></div>
        <div><dt>Для точных дат заказа</dt><dd>Срок поставки в днях — у каждого товара свой.</dd></div>
      </dl>
    </section>`);
  const drop = $("#drop");
  if (drop) {
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadFile(f, f.name); });
    drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#fileInput").click(); } });
  }
}

function showStage(title, sub) {
  $("#tabs").classList.add("hidden");
  put($app(), h`
    <section class="upload">
      <h1 class="h1">${title}</h1>
      <div class="panel stage">
        <div style="display:flex;gap:12px;align-items:center"><i class="spin"></i><b id="stageText">Готовлюсь…</b></div>
        <div class="bar"><i id="stageBar" style="width:4%"></i></div>
        <p class="small muted">${sub}</p>
      </div>
    </section>`);
}

let uploading = false;
async function uploadFile(file, name, opts = {}) {
  if (uploading) return;
  uploading = true;
  const uid = Math.random().toString(36).slice(2);
  showStage(opts.restore ? "Восстанавливаю ваши данные" : "Разбираю файл", opts.restore
    ? "Сервер был перезапущен — загружаю ваш файл заново. Это займёт несколько секунд."
    : "Обычно это занимает 10–30 секунд: система очищает данные, ищет ошибки и строит первый прогноз.");
  setBusy("upload", "Загружаю файл…");
  const poll = setInterval(async () => {
    try {
      const p = await api("/upload-progress?uid=" + uid);
      const t = $("#stageText"), b = $("#stageBar");
      if (t) t.textContent = p.stage;
      if (b) b.style.width = Math.max(4, p.pct) + "%";
    } catch { /* пропускаем */ }
  }, 700);
  try {
    const fd = new FormData();
    fd.append("file", file, name);
    const resp = await fetch("/upload?uid=" + uid, { method: "POST", body: fd });
    if (!resp.ok) {
      let msg = "Не удалось загрузить файл.";
      try { msg = (await resp.json()).detail || msg; } catch { /* не json */ }
      throw new ApiError(resp.status, msg);
    }
    const sess = await resp.json();
    const key = name + ":" + file.size;
    if (S.fileKey !== key) { S.set.lead_overrides = {}; S.set.in_transit = {}; S.set.stock_overrides = {}; }
    S.fileKey = key;
    S.sid = sess.session_id; S.session = sess;
    S.plan = S.acc = S.dataset = S.features = S.anomalies = S.impact = S.fcData = null;
    S.seenHorizons = new Set([30]);
    S.open = new Set();
    await idb.put({ name, blob: file, key });
    save();
    startApp();
  } catch (e) {
    S.sid = null; save();
    showUpload(e.message || "Не удалось загрузить файл.");
  } finally {
    clearInterval(poll);
    uploading = false;
    setBusy("upload", null);
  }
}

/* ---------- запуск и восстановление после F5 ---------- */
async function boot() {
  load();
  try { S.health = await api("/health"); } catch { S.health = null; }
  if (S.sid) {
    showStage("Открываю ваши данные", "Секунду…");
    try {
      S.session = await api("/session?session_id=" + S.sid);
      startApp();
      return;
    } catch (e) {
      if (e.status === 404) {
        const f = await idb.get();
        if (f) { await uploadFile(f.blob, f.name, { restore: true }); return; }
      } else if (e.status === 0) {
        showUpload(e.message);
        return;
      }
    }
  }
  showUpload();
}

function startApp() {
  $("#tabs").classList.remove("hidden");
  $("#topFile").classList.remove("hidden");
  $("#topFileName").textContent = S.session.filename || "файл";
  $("#topFileName").title = S.session.filename || "";
  paintTabs();
  renderTab();
  if (!S.plan) runPlan("Считаю план заказа…");
}

async function newFile() {
  if (!confirm("Загрузить другой файл? Текущий план и ваши правки (сроки поставки, остатки) будут сброшены.")) return;
  await idb.clear();
  S.sid = null; S.session = null; S.plan = null; S.fileKey = null;
  S.set.lead_overrides = {}; S.set.in_transit = {}; S.set.stock_overrides = {};
  save();
  showUpload();
}

/* ---------- расчёт плана ---------- */
let planCtl = null, planSeq = 0;
function planBody() {
  return { session_id: S.sid, horizon: S.set.horizon, default_lead: S.set.default_lead, review_period: S.set.review_period,
    lead_overrides: S.set.lead_overrides, in_transit: S.set.in_transit, stock_overrides: S.set.stock_overrides };
}
async function runPlan(text) {
  if (!S.sid) return;
  if (planCtl) planCtl.abort();
  planCtl = new AbortController();
  const my = ++planSeq;
  const newHorizon = !S.seenHorizons.has(S.set.horizon);
  setBusy("plan", text || (newHorizon ? "Считаю прогноз на новый срок — это займёт несколько секунд…" : "Пересчитываю план…"));
  try {
    const p = await api("/plan", { method: "POST", body: planBody(), signal: planCtl.signal });
    if (my !== planSeq) return;
    S.plan = p;
    S.seenHorizons.add(S.set.horizon); S.seenHorizons.add(p.params.horizon_used);
    paintTabs();
    if (S.tab === "plan") paintPlan();
    if (S.tab === "recs") VIEWS.recs();
  } catch (e) {
    if (e.name === "AbortError") return;
    if (e.status === 404) {
      const f = await idb.get();
      if (f) { await uploadFile(f.blob, f.name, { restore: true }); return; }
    }
    fail(e);
  } finally {
    if (my === planSeq) setBusy("plan", null);
  }
}
const schedulePlan = debounce(runPlan, 350);
function changed(text) {
  save();
  setBusy("plan", text || "Пересчитываю план…");
  schedulePlan(text);
}

/* ---------- раздел «Что заказать» ---------- */
const HORIZONS = [[14, "2 нед."], [30, "Месяц"], [60, "2 мес."], [90, "3 мес."]];
const stepper = (key, val, label, unit = "дн.", extra = "") => h`
  <div class="stepper" ${raw(extra)}>
    <button type="button" data-act="step" data-k="${key}" data-d="-1" aria-label="${label}: меньше">−</button>
    <input class="num" inputmode="numeric" data-in="${key}" value="${val}" aria-label="${label}">
    <span class="unit">${unit}</span>
    <button type="button" data-act="step" data-k="${key}" data-d="1" aria-label="${label}: больше">+</button>
  </div>`;

VIEWS.plan = function () {
  const hasLeadCol = S.session && S.session.has_lead_column;
  put($app(), h`
    <div class="busy-wrap" data-busy="plan">
      ${veil()}
      <div class="page-head" id="planHero"></div>
      <section class="panel" aria-label="Настройки расчёта">
        <div class="settings">
          <div class="field">
            <span class="lab">На сколько вперёд планируем</span>
            <div class="seg" role="group" aria-label="Период планирования">
              ${HORIZONS.map(([d, l]) => h`<button type="button" data-act="horizon" data-d="${d}" aria-pressed="${String(S.set.horizon === d)}">${l}</button>`)}
            </div>
            <p class="hint">Чем дольше период, тем больше получится заказ.</p>
          </div>
          <div class="field">
            <span class="lab">Срок поставки по умолчанию</span>
            ${stepper("default_lead", S.set.default_lead, "Срок поставки")}
            <p class="hint">${hasLeadCol ? "В файле есть сроки по товарам — они важнее этого числа. Его получат товары, для которых срок не указан."
              : "За сколько дней привозят товар. Для отдельного товара срок можно поменять — раскройте его в списке."}</p>
          </div>
          <div class="field">
            <span class="lab">Как часто вы заказываете</span>
            ${stepper("review_period", S.set.review_period, "Период между заказами")}
            <p class="hint">Раз во сколько дней вы отправляете заказ поставщику. Чем реже заказ — тем больше нужен запас.</p>
          </div>
        </div>
        <div class="asof" id="planAsOf"></div>
      </section>
      <div id="planNote" style="margin-top:14px"></div>
      <div class="toolbar">
        <div class="chips" id="planChips" role="group" aria-label="Фильтр товаров"></div>
        <label class="search"><span class="sr">Найти товар</span>
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="9" cy="9" r="6"/><path d="m14 14 4 4"/></svg>
          <input class="input" id="planSearch" type="search" placeholder="Найти товар" value="${S.q}" autocomplete="off"></label>
      </div>
      <section class="panel divide" id="planList" aria-label="Список товаров"></section>
      <p class="tiny muted" style="margin-top:12px">Нажмите на товар, чтобы увидеть подробности и поправить срок поставки, остаток или то, что уже едет.</p>
    </div>`);
  if (S.plan) paintPlan(); else paintPlanSkeleton();
};

function paintPlanSkeleton() {
  put($("#planHero"), h`<div><div class="skel" style="height:40px;width:min(420px,80vw)"></div><div class="skel" style="height:18px;width:min(300px,60vw);margin-top:12px"></div></div>`);
  put($("#planList"), h`${[1, 2, 3, 4].map(() => h`<div style="padding:22px 20px"><div class="skel" style="height:18px;width:60%"></div></div>`)}`);
}

function matches(it) {
  const k = it.action_kind;
  const f = S.filter;
  const ok = f === "all" || (f === "now" && k === "now") || (f === "soon" && (k === "soon" || k === "later"))
    || (f === "none" && k === "none") || (f === "excess" && k === "excess") || (f === "unknown" && k === "unknown");
  return ok && (!S.q || it.sku.toLowerCase().includes(S.q.toLowerCase()));
}

function paintPlan() {
  const P = S.plan;
  if (!P || !$("#planList")) return;
  const sm = P.summary, items = P.items;

  // заголовок
  let title, sub = "";
  if (!sm.total) { title = "Нет товаров для расчёта"; }
  else if (sm.now > 0) {
    title = `Сегодня нужно заказать ${sm.now} ${plural(sm.now, "товар", "товара", "товаров")}`;
    sub = `Сумма заказа — ${fRub(sm.order_value_now)}.` + (sm.soon ? ` Ещё ${sm.soon} — в ближайшие 3 дня.` : "");
    if (sm.shortage_skus) sub += ` У ${sm.shortage_skus} ${plural(sm.shortage_skus, "товара", "товаров", "товаров")} остаток кончится раньше, чем придёт поставка — чем раньше заказ, тем меньше потерь.`;
  } else if (sm.soon > 0) {
    title = `Срочного нет, но в ближайшие дни закажите ${sm.soon} ${plural(sm.soon, "товар", "товара", "товаров")}`;
  } else if (sm.later > 0) {
    const nx = items.filter((i) => i.action_kind === "later").sort((a, b) => a.order_in_days - b.order_in_days)[0];
    title = "Сегодня заказывать ничего не нужно";
    sub = nx ? `Ближайший заказ — «${nx.sku}» до ${fDay(nx.order_by_date)}.` : "";
  } else { title = "Заказ сейчас не требуется"; sub = "Остатков хватает на весь выбранный период."; }
  if (sm.unknown) sub += ` По ${sm.unknown} ${plural(sm.unknown, "товару", "товарам", "товарам")} в файле нет остатка — укажите его в списке.`;

  const acc = S.session && S.session.accuracy_label;
  put($("#planHero"), h`
    <div><h1 class="h1">${title}</h1>${sub ? h`<p class="lead">${sub}</p>` : ""}</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
      <button class="btn ghost" type="button" data-act="goacc" title="Насколько можно доверять прогнозу">Точность прогноза: <b id="accWord">${acc || "…"}</b></button>
      <button class="btn" type="button" data-act="xlsx" id="xlsxBtn"${sm.order_lines_total ? "" : " disabled"}>Скачать заявку (Excel)</button>
    </div>`);

  put($("#planAsOf"), h`Расчёт на ${fDay(P.as_of)} — это день после последних данных в файле.
    ${Object.keys(S.set.lead_overrides).length + Object.keys(S.set.in_transit).length + Object.keys(S.set.stock_overrides).length
      ? h` Вы внесли правки вручную. <button class="link" type="button" data-act="resetall">Сбросить все правки</button>` : ""}`);
  put($("#planNote"), P.horizon_note ? h`<div class="note"><span class="ic">i</span><span>${P.horizon_note}</span></div>` : "");

  // фильтры
  const cnt = { all: items.length, now: sm.now, soon: sm.soon + sm.later, none: sm.none, excess: sm.excess, unknown: sm.unknown };
  const chips = [["all", "Все"], ["now", "Заказать сегодня"], ["soon", "Скоро"], ["none", "Заказ не нужен"], ["excess", "Много на складе"], ["unknown", "Нет остатка"]];
  put($("#planChips"), h`${chips.filter(([k]) => k === "all" || cnt[k] > 0).map(([k, l]) => h`
    <button class="chip" type="button" data-act="filter" data-f="${k}" aria-pressed="${String(S.filter === k)}">${l} <span class="n">${cnt[k]}</span></button>`)}`);

  paintRows();
}

function paintRows() {
  const P = S.plan;
  if (!P) return;
  const list = P.items.filter(matches);
  const shown = list.slice(0, S.limit);
  put($("#planList"), list.length ? h`
    ${shown.map(rowHTML)}
    ${list.length > shown.length ? h`<div style="padding:14px 20px"><button class="btn ghost small" type="button" data-act="more">Показать все (${list.length})</button></div>` : ""}`
    : h`<div class="empty"><b>Ничего не найдено</b>Измените фильтр или поисковый запрос.</div>`);
}

function runway(it) {
  if (!it.stock_known) return h`<div class="run-cap">Остаток неизвестен — укажите его, и мы покажем, на сколько хватит товара.</div>`;
  const lead = it.lead_time_days, dos = Math.max(it.days_of_supply, 0);
  const cap = 60;
  const max = Math.max(lead * 1.6, Math.min(dos, cap) * 1.04, 14);
  const have = Math.min(dos, max) / max * 100;
  const tick = lead / max * 100;
  const gap = dos < lead ? h`<div class="run-gap" style="left:${dos / max * 100}%;width:${(lead - dos) / max * 100}%"></div>` : "";
  const right = tick > 80;
  const cap2 = it.stockout_gap_days > 0
    ? h`<div class="run-cap bad">Товар кончится за ${fDays(Math.round(dos))}, а поставка идёт ${fDays(lead)}: без товара около ${fDays(it.stockout_gap_days)}</div>`
    : h`<div class="run-cap">Хватит на ${dos > cap ? cap + "+" : fDays(Math.round(dos))}${it.in_transit ? ` (+ в пути ${fInt(it.in_transit)} шт)` : ""} · поставка ${fDays(lead)}</div>`;
  return h`<div class="run" data-s="${it.status}" aria-hidden="true">
      <div class="run-track"><div class="run-have" style="width:${have}%"></div>${gap}</div>
      <div class="run-tick" style="left:${tick}%"></div>
      <div class="run-lab" style="${right ? `right:${100 - tick}%;margin-right:6px` : `left:${tick}%;margin-left:6px`}">поставка</div>
    </div>${cap2}`;
}

const PILL = { now: "red", soon: "amber", later: "gray", none: "green", excess: "blue", unknown: "gray" };
function rowHTML(it) {
  const open = S.open.has(it.sku);
  const k = it.action_kind;
  const label = k === "none" ? "Заказ не нужен" : k === "excess" ? "Много на складе" : it.action;
  const showAmt = it.recommended_order > 0 && (k === "now" || k === "soon" || k === "later");
  return h`<div class="row ${open ? "open" : ""}" data-k="${k}" data-sku="${it.sku}">
    <button class="row-main edge" type="button" data-act="toggle" aria-expanded="${String(open)}">
      <div>
        <div class="row-name">${it.sku}</div>
        <div class="row-sub">${it.stock_known ? `${fInt(it.stock)} шт на складе` : "остаток неизвестен"} · продаётся ≈ ${fNum(it.daily_demand, it.daily_demand < 10 ? 1 : 0)} шт в день</div>
      </div>
      <div class="run-col">${runway(it)}</div>
      <div class="row-act">
        <span class="pill ${PILL[k]}">${label}</span>
        ${showAmt ? h`<span class="amount num">${fInt(it.recommended_order)} шт · ${fRub(it.order_value)}</span>` : ""}
      </div>
      <svg class="caret" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m5 8 5 5 5-5"/></svg>
    </button>
    ${open ? detailHTML(it) : ""}
  </div>`;
}

function detailHTML(it) {
  const manualLead = it.lead_source === "manual";
  const hasTransit = it.in_transit > 0;
  const stockEdited = Object.prototype.hasOwnProperty.call(S.set.stock_overrides, it.sku);
  const dos = Math.round(Math.max(it.days_of_supply, 0));
  const facts = [];
  if (it.stock_known) {
    facts.push(h`<p>Остатка хватит примерно на <b>${fDays(dos)}</b>${it.stockout_date ? h` — до ${fDay(it.stockout_date)}` : ""}.</p>`);
    if (it.stockout_gap_days > 0) facts.push(h`<p class="bad">Поставка идёт ${fDays(it.lead_time_days)}, поэтому товара не будет около <b>${fDays(it.stockout_gap_days)}</b>. За это время недопродадим ≈ ${fInt(it.expected_lost_units)} шт.</p>`);
    else facts.push(h`<p>Поставка идёт ${fDays(it.lead_time_days)} (${it.lead_source_label}) — остатка на это время хватит.</p>`);
  }
  if (it.order_by_date) {
    facts.push(h`<p>Заказать нужно ${it.order_in_days <= 0 ? h`<b>сегодня</b>` : h`не позже <b>${fDay(it.order_by_date)}</b> (через ${fDays(it.order_in_days)})`}.${it.arrival_date ? h` Если заказать сегодня, товар придёт ${fDay(it.arrival_date)}.` : ""}</p>`);
  }
  if (it.recommended_order > 0 && it.days_after_order >= 0) {
    facts.push(h`<p>Если заказать <b>${fInt(it.recommended_order)} шт</b>, товара хватит ещё примерно на <b>${fDays(Math.round(it.days_after_order))}</b>${it.covered_until ? h` — до ${fDay(it.covered_until)}` : ""}.
      ${it.days_after_order_high >= 0 ? h` Если продажи окажутся выше ожидаемого — на ${fDays(Math.round(it.days_after_order_high))}.` : ""}</p>`);
  } else if (it.stock_known && it.recommended_order <= 0) {
    facts.push(h`<p>Заказ сейчас не нужен: остатка хватает с запасом на выбранный период.</p>`);
  }
  if (it.note) facts.push(h`<p class="muted small">${it.note}</p>`);
  const pit = { ...it, params_horizon: S.plan.params.horizon_used };

  return h`<div class="detail">
    <div class="edit">
      <div class="field">
        <span class="lab">Срок поставки<span class="src">${it.lead_source_label}</span></span>
        ${stepper("sku_lead", it.lead_time_days, "Срок поставки товара", "дн.", `data-sku="${it.sku}"`)}
        ${manualLead ? h`<p class="hint"><button class="link" type="button" data-act="resetlead" data-sku="${it.sku}">Вернуть прежний срок</button></p>` : ""}
      </div>
      <div class="field">
        <label for="tr-${it.sku}">Уже заказано и едет</label>
        <div class="stepper" style="width:150px"><input class="num" id="tr-${it.sku}" style="width:100%" inputmode="decimal" data-in="sku_transit" data-sku="${it.sku}" value="${hasTransit ? it.in_transit : ""}" placeholder="0"><span class="unit">шт</span></div>
        <p class="hint">Если вы уже сделали заказ, который ещё не пришёл, — впишите сюда.</p>
      </div>
      <div class="field">
        <label for="st-${it.sku}">Остаток на складе</label>
        <div class="stepper" style="width:150px"><input class="num" id="st-${it.sku}" style="width:100%" inputmode="decimal" data-in="sku_stock" data-sku="${it.sku}" value="${it.stock_known ? it.stock : ""}" placeholder="неизвестно"><span class="unit">шт</span></div>
        <p class="hint">${stockEdited ? h`Вы изменили остаток. <button class="link" type="button" data-act="resetstock" data-sku="${it.sku}">Взять из файла</button>` : "Из файла. Если на складе другое число — поправьте."}</p>
      </div>
      <div class="detail-links"><button class="link" type="button" data-act="gosku" data-sku="${it.sku}" data-to="forecast">Продажи и прогноз →</button></div>
    </div>
    <div class="facts">
      ${facts}
      ${it.stock_known ? h`<div class="proj"><div class="small muted" style="margin-bottom:4px">Как будет убывать остаток, если заказать сегодня</div>${raw(projectionSVG(pit))}
        <div class="legend" style="padding:6px 0 0"><span><i style="background:var(--brand)"></i>ожидаемые продажи</span><span><i style="background:var(--amber)"></i>если спрос высокий</span></div></div>` : ""}
    </div>
  </div>`;
}

/* ---------- Советы ---------- */
const REC_DEST = (r) => r.key.startsWith("anom_") ? ["data", "anom"] : (r.key === "lowacc" || r.key === "noforecast") ? ["data", "acc"] : null;
VIEWS.recs = function () {
  const R = S.plan && S.plan.recommendations;
  if (!R) {
    put($app(), h`<div class="busy-wrap" data-busy="plan">${veil()}<div class="page-head"><h1 class="h1">Советы</h1></div><div class="panel"><div class="empty"><i class="spin"></i><p style="margin-top:10px">Готовлю советы…</p></div></div></div>`);
    return;
  }
  const groups = R.groups.map((g) => ({ ...g, items: R.items.filter((r) => r.group === g.code) })).filter((g) => g.items.length);
  put($app(), h`
    <div class="busy-wrap" data-busy="plan">${veil()}
      <div class="page-head"><div><h1 class="h1">${R.headline}</h1>
        <p class="lead">Что стоит сделать, начиная с самого важного. Рядом — деньги, если их можно посчитать.</p></div></div>
      ${groups.length ? groups.map((g) => h`
        <section class="rec-group" aria-label="${g.title}">
          <h2 class="h2">${g.title}</h2>
          <div style="display:grid;gap:12px">${g.items.map((r) => {
            const dest = REC_DEST(r);
            return h`<article class="panel rec ${r.group}">
              <div><div class="t">${r.title}</div><p class="p">${r.text}</p></div>
              ${r.money ? h`<div class="money"><b class="num">${fRub(r.money)}</b><span>${r.money_label}</span></div>` : ""}
              ${r.skus.length ? h`<div class="skus">${r.skus.slice(0, 8).map((s) => h`<button class="chip" type="button" data-act="gosku" data-sku="${s}" data-to="plan">${s}</button>`)}
                ${r.skus.length > 8 ? h`<span class="tiny muted" style="align-self:center">и ещё ${r.skus.length - 8}</span>` : ""}</div>` : ""}
              ${dest ? h`<div class="more"><button class="link small" type="button" data-act="gotab" data-tab="data" data-dtab="${dest[1] === "acc" ? "acc" : "anom"}">Открыть подробности →</button></div>` : ""}
            </article>`;
          })}</div>
        </section>`) : h`<div class="panel"><div class="empty"><b>Советов нет</b>Всё в порядке.</div></div>`}
    </div>`);
};

/* ---------- обработчики ---------- */
function setSetting(k, v) {
  const lim = { default_lead: [1, 120], review_period: [1, 120] }[k];
  v = clamp(Math.round(Number(v)), lim[0], lim[1]);
  if (!isFinite(v)) return;
  S.set[k] = v;
  const inp = $(`[data-in="${k}"]`); if (inp) inp.value = v;
  changed(k === "default_lead" ? "Пересчитываю даты заказа с новым сроком поставки…" : "Пересчитываю план…");
}
function curItem(sku) { return S.plan && S.plan.items.find((i) => i.sku === sku); }
function setLead(sku, v) {
  v = clamp(Math.round(Number(v)), 1, 120);
  if (!isFinite(v)) return;
  S.set.lead_overrides[sku] = v;
  changed(`Пересчитываю «${sku}»: срок поставки ${fDays(v)}…`);
}

const ACTIONS = {
  pick: () => $("#fileInput").click(),
  demo: async () => {
    try { setBusy("upload", "Загружаю пример…"); const r = await api("/demo.xlsx", { raw: true }); const b = await r.blob(); await uploadFile(b, "пример-магазин.xlsx"); }
    catch (e) { fail(e); } finally { setBusy("upload", null); }
  },
  newfile: newFile,
  tab: (t) => go(t.dataset.tab),
  gotab: (t) => go(t.dataset.tab, { dtab: t.dataset.dtab || S.dtab }),
  goacc: () => go("data", { dtab: "acc" }),
  horizon: (t) => {
    S.set.horizon = +t.dataset.d;
    $$("[data-act=horizon]").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.d === S.set.horizon)));
    changed(S.seenHorizons.has(S.set.horizon) ? "Пересчитываю план…" : "Считаю прогноз на новый срок — это займёт несколько секунд…");
  },
  step: (t) => {
    const k = t.dataset.k, d = +t.dataset.d;
    if (k === "sku_lead") { const sku = t.closest("[data-sku]").dataset.sku; const it = curItem(sku); setLead(sku, (S.set.lead_overrides[sku] ?? it.lead_time_days) + d); const i = $(`[data-in=sku_lead]`, t.parentElement); if (i) i.value = S.set.lead_overrides[sku]; }
    else setSetting(k, S.set[k] + d);
  },
  filter: (t) => { S.filter = t.dataset.f; $$("#planChips .chip").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.f === S.filter))); S.limit = 100; paintRows(); },
  more: () => { S.limit = 1e6; paintRows(); },
  toggle: (t) => {
    const sku = t.closest(".row").dataset.sku;
    if (S.open.has(sku)) S.open.delete(sku); else S.open.add(sku);
    const it = curItem(sku), row = t.closest(".row");
    row.outerHTML = rowHTML(it).s;
  },
  resetlead: (t) => { delete S.set.lead_overrides[t.dataset.sku]; changed("Возвращаю прежний срок поставки…"); },
  resetstock: (t) => { delete S.set.stock_overrides[t.dataset.sku]; changed("Беру остаток из файла…"); },
  resetall: () => { S.set.lead_overrides = {}; S.set.in_transit = {}; S.set.stock_overrides = {}; changed("Сбрасываю ваши правки…"); },
  gosku: (t) => {
    const sku = t.dataset.sku, to = t.dataset.to;
    if (to === "forecast") { S.fcSku = sku; go("forecast"); }
    else { S.filter = "all"; S.q = ""; S.open.add(sku); S.limit = 1e6; go("plan"); setTimeout(() => { const r = $$(".row").find((x) => x.dataset.sku === sku); if (r) r.scrollIntoView({ block: "center", behavior: "smooth" }); }, 50); }
  },
  xlsx: async (t) => {
    t.disabled = true; const old = t.innerHTML; t.innerHTML = '<i class="spin"></i> Готовлю файл…';
    setBusy("xlsx", "Готовлю заявку…");
    try {
      const r = await api("/plan/xlsx", { method: "POST", body: planBody(), raw: true });
      const b = await r.blob(), a = document.createElement("a");
      a.href = URL.createObjectURL(b);
      a.download = `Заявка_на_закупку_${new Date().toISOString().slice(0, 10)}.xlsx`;
      document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      toast("Заявка скачана");
    } catch (e) { fail(e); } finally { t.disabled = false; t.innerHTML = old; setBusy("xlsx", null); }
  },
};

document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-act]");
  if (!t || t.closest("[disabled]")) return;
  const fn = ACTIONS[t.dataset.act] || VIEW_ACTIONS[t.dataset.act];
  if (fn) { if (t.tagName === "A") e.preventDefault(); fn(t, e); }
});
document.addEventListener("change", (e) => {
  const t = e.target;
  if (t.id === "fileInput") { const f = t.files[0]; t.value = ""; if (f) uploadFile(f, f.name); return; }
  const k = t.dataset && t.dataset.in;
  if (!k) return;
  if (VIEW_INPUTS[k]) return VIEW_INPUTS[k](t);
  if (k === "default_lead" || k === "review_period") return setSetting(k, t.value);
  const sku = t.closest("[data-sku]") && t.closest("[data-sku]").dataset.sku;
  if (k === "sku_lead") return setLead(sku, t.value);
  if (k === "sku_transit") {
    const v = parseFloat(String(t.value).replace(",", "."));
    if (isFinite(v) && v > 0) S.set.in_transit[sku] = v; else delete S.set.in_transit[sku];
    return changed(`Учитываю заказ в пути по «${sku}»…`);
  }
  if (k === "sku_stock") {
    const raw_ = String(t.value).trim();
    if (raw_ === "") delete S.set.stock_overrides[sku];
    else { const v = parseFloat(raw_.replace(",", ".")); if (isFinite(v) && v >= 0) S.set.stock_overrides[sku] = v; else return toast("Остаток — это число, например 120", true); }
    return changed(`Пересчитываю «${sku}» с новым остатком…`);
  }
});
document.addEventListener("input", (e) => {
  if (e.target.id === "planSearch") { S.q = e.target.value; S.limit = 100; paintRows(); }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.matches && e.target.matches("[data-in]")) e.target.blur();
});

window.addEventListener("DOMContentLoaded", boot);
