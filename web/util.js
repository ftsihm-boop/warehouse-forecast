/* Общие функции: безопасная вёрстка из шаблонов, форматы, запросы, хранилище. */
"use strict";

/* ---------- безопасные шаблоны ---------- */
class Raw { constructor(s) { this.s = s; } }
const raw = (s) => new Raw(String(s));
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);
function val(v) {
  if (v == null || v === false) return "";
  if (v instanceof Raw) return v.s;
  if (Array.isArray(v)) return v.map(val).join("");
  return esc(v);
}
/** h`<b>${text}</b>` — всё подставляемое экранируется, кроме вложенных h`` */
function h(strings, ...vals) {
  let out = "";
  strings.forEach((s, i) => { out += s; if (i < vals.length) out += val(vals[i]); });
  return new Raw(out);
}
const put = (el, r) => { if (el) el.innerHTML = r instanceof Raw ? r.s : esc(r); };
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

/* ---------- форматы ---------- */
const NB = " ";
const MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"];
const MONTHS_S = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];

function plural(n, one, few, many) {
  n = Math.abs(Math.round(n));
  const a = n % 10, b = n % 100;
  if (a === 1 && b !== 11) return one;
  if (a >= 2 && a <= 4 && !(b >= 12 && b <= 14)) return few;
  return many;
}
const fInt = (n) => (n == null || !isFinite(n)) ? "—" : Math.round(n).toLocaleString("ru-RU").replace(/ | /g, NB);
function fNum(n, d = 1) {
  if (n == null || !isFinite(n)) return "—";
  return n.toLocaleString("ru-RU", { maximumFractionDigits: d, minimumFractionDigits: 0 }).replace(/ | /g, NB);
}
const fRub = (n) => (n == null || !isFinite(n)) ? "—" : fInt(n) + NB + "₽";
const fPct = (n, d = 0) => (n == null || !isFinite(n)) ? "—" : fNum(n, d) + NB + "%";
const fSign = (n, d = 0) => (n == null || !isFinite(n)) ? "—" : (n > 0 ? "+" : n < 0 ? "−" : "") + fNum(Math.abs(n), d);
const toDate = (iso) => new Date(iso + "T00:00:00Z");
function fDay(iso) { // «12 октября»
  if (!iso) return "—";
  const d = toDate(iso);
  return d.getUTCDate() + NB + MONTHS[d.getUTCMonth()];
}
function fDayY(iso) { // «12.10.2026»
  if (!iso) return "—";
  const d = toDate(iso);
  return String(d.getUTCDate()).padStart(2, "0") + "." + String(d.getUTCMonth() + 1).padStart(2, "0") + "." + d.getUTCFullYear();
}
const fDays = (n) => fInt(n) + NB + plural(n, "день", "дня", "дней");
const addDays = (iso, n) => new Date(toDate(iso).getTime() + n * 864e5).toISOString().slice(0, 10);

/* ---------- хранилище ---------- */
const LS_KEY = "wf.state.v2";
function lsGet() { try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}") || {}; } catch { return {}; } }
function lsSet(o) { try { localStorage.setItem(LS_KEY, JSON.stringify(o)); } catch { /* приватный режим */ } }

/* Исходный файл лежит в IndexedDB: если сервер перезапустился и забыл
   сессию, страница молча загружает файл заново — пользователь ничего не теряет. */
const idb = {
  open() {
    return new Promise((res, rej) => {
      if (!window.indexedDB) return rej(new Error("no idb"));
      const r = indexedDB.open("wf-files", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("f");
      r.onsuccess = () => res(r.result);
      r.onerror = () => rej(r.error);
    });
  },
  async put(rec) {
    try {
      const db = await this.open();
      await new Promise((res, rej) => { const t = db.transaction("f", "readwrite"); t.objectStore("f").put(rec, "last"); t.oncomplete = res; t.onerror = () => rej(t.error); });
    } catch { /* не критично */ }
  },
  async get() {
    try {
      const db = await this.open();
      return await new Promise((res, rej) => { const q = db.transaction("f").objectStore("f").get("last"); q.onsuccess = () => res(q.result || null); q.onerror = () => rej(q.error); });
    } catch { return null; }
  },
  async clear() {
    try {
      const db = await this.open();
      await new Promise((res) => { const t = db.transaction("f", "readwrite"); t.objectStore("f").delete("last"); t.oncomplete = res; t.onerror = res; });
    } catch { /* ок */ }
  },
};

/* ---------- запросы ---------- */
class ApiError extends Error { constructor(status, msg) { super(msg); this.status = status; } }
async function api(path, { method = "GET", body, signal, raw: wantRaw } = {}) {
  let resp;
  try {
    resp = await fetch(path, {
      method, signal,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError(0, "Нет связи с сервером. Проверьте интернет и попробуйте ещё раз.");
  }
  if (!resp.ok) {
    let msg = "Не получилось выполнить запрос.";
    try { const j = await resp.json(); if (j.detail) msg = j.detail; } catch { /* не json */ }
    throw new ApiError(resp.status, msg);
  }
  return wantRaw ? resp : resp.json();
}
const qs = (o) => Object.entries(o).filter(([, v]) => v != null).map(([k, v]) => k + "=" + encodeURIComponent(v)).join("&");

const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
