/* Графики на чистом SVG, без библиотек. */
"use strict";

const C = {
  brand: "var(--brand)", brandSoft: "var(--brand-soft)", ink: "var(--ink)", ink2: "var(--ink-2)", ink3: "var(--ink-3)",
  line: "var(--line)", line2: "var(--line-2)", red: "var(--red)", amber: "var(--amber)", blue: "var(--blue)",
};

function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  const n = m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10;
  return n * p;
}
const pathOf = (pts) => {
  let d = "", pen = false;
  pts.forEach(([x, y]) => {
    if (x == null || y == null || !isFinite(y)) { pen = false; return; }
    d += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    pen = true;
  });
  return d;
};
function trail(arr, w) {
  return arr.map((_, i) => {
    let s = 0, n = 0;
    for (let k = Math.max(0, i - w + 1); k <= i; k++) if (arr[k] != null) { s += arr[k]; n++; }
    return n ? s / n : null;
  });
}

/* ---------------------------------------------------------------
   «Хватит ли товара»: как будет убывать остаток после заказа
   --------------------------------------------------------------- */
function projectionSVG(it) {
  const W = 540, H = 196, L = 42, R = 12, T = 16, B = 30;
  const pw = W - L - R, ph = H - T - B;
  const lead = Math.max(1, it.lead_time_days || 1);
  const d50 = Math.max(it.daily_demand || 0, 0);
  const d90 = Math.max((it.forecast_horizon_q90 || 0) / Math.max(it.params_horizon || 30, 1), d50);
  const stock = Math.max(it.stock || 0, 0);
  const add = (it.in_transit || 0) + (it.recommended_order || 0);
  const need = d50 > 0 ? (stock + add) / d50 : 60;
  const days = Math.round(clamp(Math.max(lead * 1.7, Math.min(need * 1.15, 90), 21), 21, 90));

  const at = (d, t) => {
    // остаток к концу дня t; склад не уходит ниже нуля, непроданное — дефицит
    let s = stock - d * t;
    if (t >= lead) s = Math.max(stock - d * lead, 0) + add - d * (t - lead);
    return Math.max(s, 0);
  };
  const A = [], B_ = [];
  let ymax = stock + add;
  for (let t = 0; t <= days; t++) { A.push(at(d50, t)); B_.push(at(d90, t)); }
  ymax = niceMax(Math.max(ymax, ...A) * 1.05);

  const X = (t) => L + (t / days) * pw;
  const Y = (v) => T + ph - (v / ymax) * ph;
  const lineA = pathOf(A.map((v, t) => [X(t), Y(v)]));
  const lineB = pathOf(B_.map((v, t) => [X(t), Y(v)]));
  const area = lineA + `L${X(days)} ${Y(0)}L${X(0)} ${Y(0)}Z`;

  // когда запас кончится до прихода поставки
  const zeroAt = d50 > 0 ? stock / d50 : Infinity;
  const gap = zeroAt < lead
    ? `<rect x="${X(zeroAt)}" y="${T}" width="${X(lead) - X(zeroAt)}" height="${ph}" fill="${C.red}" opacity=".13"/>
       <text x="${(X(zeroAt) + X(lead)) / 2}" y="${T + 13}" text-anchor="middle" font-size="12" fill="${C.red}">нет товара</text>` : "";

  let grid = "";
  for (let i = 0; i <= 3; i++) {
    const v = (ymax / 3) * i;
    grid += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${C.line2}"/>
             <text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end" font-size="11.5" fill="${C.ink3}">${fInt(v)}</text>`;
  }
  const step = days > 45 ? 14 : 7;
  let xt = "";
  for (let t = 0; t <= days; t += step) {
    xt += `<text x="${X(t)}" y="${H - 9}" text-anchor="middle" font-size="11.5" fill="${C.ink3}">${t === 0 ? "сегодня" : "+" + t + " дн."}</text>`;
  }
  const arrLab = add > 0 ? "приедет заказ" : "приедет поставка";
  const lx = X(lead), anchor = lx > W - 90 ? "end" : "start";
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Как будет убывать остаток товара">
    ${grid}${gap}
    <path d="${area}" fill="${C.brandSoft}" opacity=".7"/>
    <path d="${lineB}" fill="none" stroke="${C.amber}" stroke-width="1.6" stroke-dasharray="4 4"/>
    <path d="${lineA}" fill="none" stroke="${C.brand}" stroke-width="2.4" stroke-linejoin="round"/>
    <line x1="${lx}" x2="${lx}" y1="${T}" y2="${T + ph}" stroke="${C.ink}" stroke-width="1.5" stroke-dasharray="3 3"/>
    <text x="${lx + (anchor === "end" ? -6 : 6)}" y="${T + ph - 6}" text-anchor="${anchor}" font-size="12" fill="${C.ink}">${arrLab}</text>
    ${xt}
  </svg>`;
}

/* ---------------------------------------------------------------
   Продажи и прогноз по одному товару
   --------------------------------------------------------------- */
function renderForecastChart(host, o) {
  const { hist, fc, rangeDays, prevYear, smooth } = o;
  if (!hist.length) { host.innerHTML = ""; return; }
  const lastIso = hist[hist.length - 1].d;
  const first = rangeDays ? Math.max(0, hist.length - rangeDays) : 0;
  const view = hist.slice(first);
  const hmap = new Map(hist.map((r) => [r.d, r]));

  const days = view.map((r) => r.d).concat(fc.map((r) => r.date));
  const nH = view.length, n = days.length;
  const qh = view.map((r) => r.q).concat(fc.map(() => null));
  const q50 = new Array(nH).fill(null).concat(fc.map((r) => r.q50));
  const q90 = new Array(nH).fill(null).concat(fc.map((r) => r.q90));
  const prev = prevYear ? days.map((d) => { const r = hmap.get(addDays(d, -364)); return r ? r.q : null; }) : null;

  const w = smooth ? 7 : 1;
  const comb50 = trail(qh.map((v, i) => v ?? q50[i]), w);
  const comb90 = trail(qh.map((v, i) => v ?? q90[i]), w);
  const prevS = prev ? trail(prev, w) : null;
  const sh = comb50.map((v, i) => (i < nH ? v : null));
  const sf = comb50.map((v, i) => (i >= nH - 1 && fc.length ? v : null));
  const sf90 = comb90.map((v, i) => (i >= nH && fc.length ? v : null));

  const W = 1000, L = 52, R = 14, T = 18, PH = 250, GAP = 44, SH = 56, B = 30;
  const H = T + PH + GAP + SH + B, PW = W - L - R;
  const all = [...sh, ...sf, ...sf90, ...(prevS || [])].filter((v) => v != null);
  const ymax = niceMax(Math.max(...all, 1) * 1.06);
  const X = (i) => L + (n > 1 ? (i / (n - 1)) * PW : 0);
  const Y = (v) => T + PH - (v / ymax) * PH;

  let grid = "";
  for (let k = 0; k <= 4; k++) {
    const v = (ymax / 4) * k;
    grid += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${C.line2}"/>
             <text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end" font-size="12" fill="${C.ink3}">${fNum(v, ymax < 10 ? 1 : 0)}</text>`;
  }
  let xt = "";
  const ticks = 6, long = n > 200;
  for (let k = 0; k <= ticks; k++) {
    const i = Math.round((k / ticks) * (n - 1));
    const d = toDate(days[i]);
    const lab = long ? MONTHS_S[d.getUTCMonth()] + " " + String(d.getUTCFullYear()).slice(2)
      : d.getUTCDate() + " " + MONTHS_S[d.getUTCMonth()];
    xt += `<text x="${X(i)}" y="${T + PH + 18}" text-anchor="${k === 0 ? "start" : k === ticks ? "end" : "middle"}" font-size="12" fill="${C.ink3}">${lab}</text>`;
  }

  const band = fc.length ? (() => {
    const up = [], lo = [];
    for (let i = nH; i < n; i++) { up.push([X(i), Y(Math.max(comb90[i], comb50[i]))]); lo.push([X(i), Y(comb50[i])]); }
    const top = up.map(([x, y], j) => (j ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1)).join("");
    const bot = lo.reverse().map(([x, y]) => "L" + x.toFixed(1) + " " + y.toFixed(1)).join("");
    return `<path d="${top}${bot}Z" fill="${C.brand}" opacity=".16"/>`;
  })() : "";

  const xNow = fc.length ? X(nH - 0.5) : null;
  const now = fc.length ? `<line x1="${xNow}" x2="${xNow}" y1="${T}" y2="${T + PH}" stroke="${C.ink3}" stroke-dasharray="3 4"/>
    <text x="${xNow + 6}" y="${T + 12}" font-size="12" fill="${C.ink2}">прогноз</text>` : "";

  // остаток на складе
  const stockVals = view.map((r) => r.s);
  const smax = niceMax(Math.max(...stockVals.filter((v) => v != null), 1) * 1.05);
  const SY0 = T + PH + GAP, SY = (v) => SY0 + SH - (v / smax) * SH;
  const sPts = stockVals.map((v, i) => [X(i), v == null ? null : SY(v)]);
  const stockLine = pathOf(sPts);
  const stockArea = stockLine ? stockLine + `L${X(nH - 1)} ${SY(0)}L${X(0)} ${SY(0)}Z` : "";
  let outs = "";
  view.forEach((r, i) => { if (r.c) outs += `<rect x="${X(i) - 1}" y="${SY0 + SH - 5}" width="2.4" height="9" fill="${C.red}"/>`; });
  const stockBlock = `<text x="${L}" y="${SY0 - 8}" font-size="12" fill="${C.ink2}">Остаток на складе</text>
    <line x1="${L}" x2="${W - R}" y1="${SY(0)}" y2="${SY(0)}" stroke="${C.line}"/>
    <path d="${stockArea}" fill="${C.blue}" opacity=".16"/><path d="${stockLine}" fill="none" stroke="${C.blue}" stroke-width="1.5"/>
    <text x="${L - 8}" y="${SY(smax) + 4}" text-anchor="end" font-size="11.5" fill="${C.ink3}">${fInt(smax)}</text>${outs}`;

  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="График продаж и прогноза">
    ${grid}${xt}${now}
    ${prevS ? `<path d="${pathOf(prevS.map((v, i) => [X(i), v == null ? null : Y(v)]))}" fill="none" stroke="${C.ink3}" stroke-width="1.4" stroke-dasharray="5 4"/>` : ""}
    ${band}
    <path d="${pathOf(sh.map((v, i) => [X(i), v == null ? null : Y(v)]))}" fill="none" stroke="${C.ink}" stroke-width="${smooth ? 2 : 1.3}" stroke-linejoin="round"/>
    <path d="${pathOf(sf.map((v, i) => [X(i), v == null ? null : Y(v)]))}" fill="none" stroke="${C.brand}" stroke-width="2.4" stroke-linejoin="round"/>
    ${stockBlock}
    <line class="cur" x1="0" x2="0" y1="${T}" y2="${SY0 + SH}" stroke="${C.ink3}" opacity="0"/>
    <rect class="hit" x="${L}" y="${T}" width="${PW}" height="${SY0 + SH - T}" fill="transparent"/>
  </svg><div class="tip" hidden></div>`;

  const svg = host.querySelector("svg"), tip = host.querySelector(".tip"), cur = host.querySelector(".cur");
  const hit = host.querySelector(".hit");
  const move = (ev) => {
    const r = svg.getBoundingClientRect();
    const vx = ((ev.clientX - r.left) / r.width) * W;
    const i = clamp(Math.round(((vx - L) / PW) * (n - 1)), 0, n - 1);
    const isF = i >= nH;
    const rows = [`<b>${fDayY(days[i])}</b>`];
    if (isF) {
      const f = fc[i - nH];
      rows.push(`Прогноз: ${fNum(f.q50, 0)} шт (до ${fNum(f.q90, 0)})`);
    } else {
      rows.push(`Продано: ${fNum(view[i].q, 0)} шт`);
      if (view[i].s != null) rows.push(`Остаток: ${fNum(view[i].s, 0)} шт`);
      if (view[i].c) rows.push(`<span style="color:#ffb4a6">товара не было</span>`);
    }
    if (prev && prev[i] != null) rows.push(`Год назад: ${fNum(prev[i], 0)} шт`);
    tip.innerHTML = rows.join("<br>");
    tip.hidden = false;
    const px = (X(i) / W) * r.width;
    tip.style.left = clamp(px, 70, r.width - 70) + "px";
    tip.style.top = "30px";
    cur.setAttribute("x1", X(i)); cur.setAttribute("x2", X(i)); cur.setAttribute("opacity", ".7");
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", () => { tip.hidden = true; cur.setAttribute("opacity", "0"); });
}
