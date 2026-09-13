"""
Слои 1 и 2 очистки: разбор файла и приведение типов.

Слой 1 — распознавание структуры:
  * читаем CSV/XLSX, при нескольких листах выбираем самый «похожий на данные»
  * ищем строку шапки (в выгрузках 1С сверху 3-5 строк с названием отчёта)
  * сопоставляем колонки со словарём синонимов

Слой 2 — типы и форматы:
  * даты: Excel serial, ISO, ДД.ММ.ГГГГ, ДД/ММ/ГГ, «31 декабря 2025»
  * числа: «1 234,56», «1,234.56», неразрывный пробел, «12 шт», «—», «н/д»
  * товар: trim, схлопывание пробелов, ё→е при сравнении
"""

from __future__ import annotations

import datetime as _dt
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import COLUMN_SYNONYMS, DATE, QTY, REQUIRED_COLUMNS, SKU, STOCK

HEADER_SCAN_ROWS = 20  # сколько верхних строк просматриваем в поисках шапки

_MONTHS_RU = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "мая": 5, "май": 5, "июн": 6,
    "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
}

_NA_TOKENS = {"", "-", "--", "—", "–", "н/д", "нд", "na", "n/a", "nan", "null", "none", "?"}


class IngestError(Exception):
    """Файл не удалось разобрать — сообщение предназначено пользователю."""


@dataclass
class IngestResult:
    frame: pd.DataFrame
    sheet_name: str | None
    header_row: int
    mapping: dict[str, str]          # каноническое имя -> имя колонки в файле
    dropped_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --- нормализация заголовков --------------------------------------------------

def normalize_header(value: object) -> str:
    """«Остаток на складе, шт.» -> «остатокнаскладешт»"""
    s = str(value if value is not None else "")
    s = s.replace("ё", "е").replace("Ё", "Е").lower()
    s = re.sub(r"[\s\u00a0_\-.,;:()\[\]{}/\\'\"«»]+", "", s)
    return s.strip()


def match_column(header: object) -> str | None:
    """Возвращает каноническое имя колонки или None."""
    norm = normalize_header(header)
    if not norm:
        return None
    for canon, variants in COLUMN_SYNONYMS.items():
        for v in variants:
            nv = normalize_header(v)
            if norm == nv:
                return canon
    # мягкое совпадение: заголовок начинается с варианта либо содержит его
    for canon, variants in COLUMN_SYNONYMS.items():
        for v in variants:
            nv = normalize_header(v)
            if len(nv) >= 4 and (norm.startswith(nv) or nv in norm):
                return canon
    return None


def score_header_row(cells: list[object]) -> tuple[int, dict[str, int]]:
    """Сколько канонических колонок распозналось в данной строке."""
    mapping: dict[str, int] = {}
    for i, c in enumerate(cells):
        canon = match_column(c)
        if canon and canon not in mapping:
            mapping[canon] = i
    # дата и товар весят больше: без них таблица бесполезна
    score = len(mapping) + (1 if DATE in mapping else 0) + (1 if SKU in mapping else 0)
    return score, mapping


# --- парсеры значений ---------------------------------------------------------

def parse_number(value: object) -> float:
    """«1 234,56» / «1,234.56» / «12 шт» / «—» -> float или NaN"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)

    s = str(value).strip().replace("\u00a0", " ")
    if s.lower() in _NA_TOKENS:
        return np.nan

    # выкидываем единицы измерения и прочий хвост
    s = re.sub(r"(шт|ед|кг|л|уп|pcs|units?)\.?$", "", s, flags=re.IGNORECASE).strip()
    s = s.replace(" ", "")

    # «1,234.56» -> точка десятичная; «1 234,56» -> запятая десятичная
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")

    s = re.sub(r"[^0-9eE+\-.]", "", s)
    if s in {"", "-", "+", "."}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def excel_serial_to_date(serial: float) -> pd.Timestamp | float:
    """Excel хранит даты числом: 45890 -> 2025-08-21. Учитываем баг 1900 года."""
    if not np.isfinite(serial) or serial < 1 or serial > 2_958_465:
        return np.nan
    return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(serial))


def parse_date(value: object) -> pd.Timestamp | float:
    if value is None:
        return np.nan
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, (_dt.datetime, _dt.date)):
        return pd.Timestamp(value).normalize()
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if not np.isfinite(float(value)):
            return np.nan
        v = float(value)
        # 20250821 — встречается в выгрузках как «число-дата»
        if 19000101 <= v <= 29991231:
            try:
                return pd.Timestamp(str(int(v))).normalize()
            except Exception:
                return np.nan
        return excel_serial_to_date(v)

    s = str(value).strip()
    if s.lower() in _NA_TOKENS:
        return np.nan

    # «31 декабря 2025»
    m = re.match(r"^(\d{1,2})\s+([а-яё]+)\.?\s+(\d{4})", s.lower())
    if m:
        mon = _MONTHS_RU.get(m.group(2)[:3])
        if mon:
            try:
                return pd.Timestamp(int(m.group(3)), mon, int(m.group(1)))
            except ValueError:
                return np.nan

    # чистое число в строке — тот же Excel serial
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return parse_date(float(s))

    # сначала явные форматы — так pandas не гадает и не сыплет warning'ами
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y", "%d.%m.%y",
                "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return pd.Timestamp(_dt.datetime.strptime(s[:19], fmt)).normalize()
        except ValueError:
            continue
    for dayfirst in (True, False):
        try:
            ts = pd.to_datetime(s, dayfirst=dayfirst, errors="raise", format="mixed")
            return pd.Timestamp(ts).normalize()
        except Exception:
            continue
    return np.nan


def clean_sku(value: object) -> str:
    s = str(value if value is not None else "").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sku_key(value: str) -> str:
    """Ключ для склейки «Молоко (1 литр)» и «Молоко(1 л)»."""
    s = value.replace("ё", "е").lower()
    s = re.sub(r"[\s\u00a0_\-.,;:()\[\]{}/\\'\"«»]+", "", s)
    s = s.replace("литр", "л").replace("килограмм", "кг").replace("грамм", "г")
    s = s.replace("штук", "шт").replace("упаковка", "уп")
    return s


# --- чтение файлов ------------------------------------------------------------

def _read_raw_tables(source: str | Path | bytes, filename: str | None = None
                     ) -> list[tuple[str | None, pd.DataFrame]]:
    """Возвращает список (имя листа, сырой DataFrame без шапки)."""
    name = (filename or (str(source) if isinstance(source, (str, Path)) else "")).lower()
    data = source

    if name.endswith((".xlsx", ".xlsm", ".xltx")) or (
        isinstance(data, bytes) and data[:2] == b"PK"
    ):
        buf = io.BytesIO(data) if isinstance(data, bytes) else data
        sheets = pd.read_excel(buf, sheet_name=None, header=None, dtype=object)
        return list(sheets.items())

    if isinstance(data, bytes):
        text = None
        for enc in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise IngestError("Не удалось определить кодировку файла.")
        handle: io.StringIO | str | Path = io.StringIO(text)
    else:
        handle = data

    last_err: Exception | None = None
    for sep in (None, ";", ",", "\t", "|"):
        try:
            if isinstance(handle, io.StringIO):
                handle.seek(0)
            df = pd.read_csv(handle, header=None, dtype=object, sep=sep,
                             engine="python", skip_blank_lines=False)
            if df.shape[1] >= 2:
                return [(None, df)]
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise IngestError(f"Не удалось прочитать файл как таблицу. {last_err}")


def read_table(source: str | Path | bytes, filename: str | None = None) -> IngestResult:
    """Читает файл, находит шапку и приводит колонки к каноническим именам."""
    tables = _read_raw_tables(source, filename)
    notes: list[str] = []

    best: tuple[int, str | None, pd.DataFrame, int, dict[str, int]] | None = None
    for sheet_name, raw in tables:
        if raw.empty:
            continue
        limit = min(HEADER_SCAN_ROWS, len(raw))
        for r in range(limit):
            score, mapping = score_header_row(list(raw.iloc[r].values))
            if best is None or score > best[0]:
                best = (score, sheet_name, raw, r, mapping)

    if best is None or best[0] == 0:
        raise IngestError(
            "В файле не найдены колонки с датой, товаром и количеством. "
            "Ожидаются заголовки вида «Дата», «Товар», «Продано», «Остаток» "
            "(допустимы синонимы и любой порядок)."
        )

    score, sheet_name, raw, header_row, idx_map = best

    missing = [c for c in REQUIRED_COLUMNS if c not in idx_map]
    if missing:
        human = {DATE: "дата", SKU: "товар", QTY: "количество проданного"}
        found = [str(v) for v in raw.iloc[header_row].values if str(v) != "nan"]
        raise IngestError(
            "Не найдены обязательные колонки: "
            + ", ".join(human.get(m, m) for m in missing)
            + ". В шапке файла распознано: " + ", ".join(found[:12])
        )

    if header_row > 0:
        notes.append(f"Шапка таблицы найдена в строке {header_row + 1}, "
                     f"{header_row} строк(и) выше пропущены как служебные.")
    if len(tables) > 1:
        notes.append(f"В файле {len(tables)} листов, выбран «{sheet_name}» "
                     f"как наиболее подходящий.")

    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    header_cells = list(raw.iloc[header_row].values)

    out = pd.DataFrame()
    mapping: dict[str, str] = {}
    for canon, col_idx in idx_map.items():
        mapping[canon] = str(header_cells[col_idx])
        out[canon] = body.iloc[:, col_idx]

    dropped = [str(c) for i, c in enumerate(header_cells)
               if i not in idx_map.values() and str(c) not in ("nan", "None", "")]
    if dropped:
        notes.append("Лишние колонки проигнорированы: " + ", ".join(dropped[:8]))

    # --- слой 2: типы -------------------------------------------------------
    out[DATE] = out[DATE].map(parse_date)
    out[SKU] = out[SKU].map(clean_sku)
    out[QTY] = out[QTY].map(parse_number)
    if STOCK in out.columns:
        out[STOCK] = out[STOCK].map(parse_number)
    else:
        out[STOCK] = np.nan
        notes.append("Колонка с остатком не найдена — расчёт заказа будет "
                     "показывать потребность без вычета текущего запаса.")

    out[DATE] = pd.to_datetime(out[DATE], errors="coerce")

    return IngestResult(frame=out, sheet_name=sheet_name, header_row=header_row,
                        mapping=mapping, dropped_columns=dropped, notes=notes)
