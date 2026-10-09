"""
Формирование заявки на закупку в формате Excel.

ЗАЧЕМ ЭТО НУЖНО
---------------
Система считает, что и сколько заказать, но этим работа менеджера не
заканчивается: рекомендацию нужно передать дальше — в отдел снабжения,
руководителю на согласование, поставщику. Экран сайта для этого не
годится: его нельзя приложить к служебной записке и нельзя подписать.

Поэтому выгрузка сделана не дампом таблицы с экрана, а документом,
который можно отдать человеку, не открывавшему систему:

    Лист 1. Заявка на закупку   только то, что нужно заказать,
                                с ценами, суммой и сроками
    Лист 2. Полный план         все товары со статусами — для справки
    Лист 3. Параметры расчёта   на чём основаны цифры

Третий лист принципиален. Документ, который уходит из системы, должен
быть самодостаточным: через месяц никто не вспомнит, при каком сроке
поставки считалась заявка и какая модель её строила. Если этого нет,
цифрам невозможно доверять, и заявку будут пересчитывать вручную.

СУММЫ СЧИТАЮТСЯ ФОРМУЛАМИ
-------------------------
Стоимость позиции и итог по заявке записаны формулами Excel, а не
готовыми числами. Так снабженец может поправить количество прямо в
файле — например, округлить до кратности упаковки, — и итоговая сумма
пересчитается сама. Файл с зашитыми числами пришлось бы пересчитывать
на калькуляторе, и это первый шаг к ошибке в заказе.
"""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# --- оформление ---------------------------------------------------------------
FONT = "Arial"
INK = "1F2D3A"
HEAD_FILL = PatternFill("solid", fgColor="1F5F5B")
HEAD_FONT = Font(name=FONT, color="FFFFFF", bold=True, size=10)
TITLE_FONT = Font(name=FONT, bold=True, size=14, color=INK)
BASE_FONT = Font(name=FONT, size=10, color=INK)
MUTED_FONT = Font(name=FONT, size=9, color="5A6B7B")
TOTAL_FONT = Font(name=FONT, bold=True, size=11, color=INK)

# Срочность заметна цветом: снабженец должен видеть, что горит, не читая
# таблицу целиком.
URGENCY_FILL = {
    "critical": PatternFill("solid", fgColor="FBE3E3"),
    "alert": PatternFill("solid", fgColor="FDF3E3"),
}
# Статус позиции и её срочность в заявке — разные вещи, и подписи у них
# тоже должны быть разными. Статус «Норма» означает, что остаток выше
# точки заказа; но при периодическом пересмотре до целевого уровня его
# всё равно дозаказывают. Строка «Норма» рядом с «заказать 73 шт» читается
# как противоречие, поэтому в заявке используется своя шкала.
STATUS_TEXT = {
    "critical": "Критично",
    "alert": "Пополнить",
    "ok": "Норма",
    "excess": "Избыток",
}

URGENCY_TEXT = {
    "critical": "Критично",
    "alert": "Срочно",
    "ok": "Плановое",
    "excess": "Плановое",
}

THIN = Side(style="thin", color="C9D4DE")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

MONEY = '# ##0.00\\ "₽"'
QTY = "# ##0"


def _num(v, default=None):
    """Приводит значение к числу; None и NaN превращает в default."""
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(f) else f


def _fmt_date(v) -> str:
    """ГГГГ-ММ-ДД -> ДД.ММ.ГГГГ; пусто -> прочерк."""
    if not v:
        return "—"
    try:
        return pd.Timestamp(v).strftime("%d.%m.%Y")
    except (ValueError, TypeError):
        return str(v)


def _header_row(ws, row: int, titles: list[str]) -> None:
    for i, t in enumerate(titles, start=1):
        c = ws.cell(row=row, column=i, value=t)
        c.fill = HEAD_FILL
        c.font = HEAD_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER
    ws.row_dimensions[row].height = 30


def _widths(ws, widths: dict[str, float]) -> None:
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


# =============================================================================
#  Лист 1 — заявка на закупку
# =============================================================================
def _sheet_request(wb: Workbook, items: list[dict], costs: dict[str, float],
                   meta: dict) -> int:
    ws = wb.active
    ws.title = "Заявка на закупку"

    ws["A1"] = "ЗАЯВКА НА ЗАКУПКУ"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:K1")

    ws["A2"] = ("Сформирована автоматически системой управления складскими "
                "запасами на основании прогноза спроса")
    ws["A2"].font = MUTED_FONT
    ws.merge_cells("A2:K2")

    lead_note = f"{meta.get('lead_time', '—')} дн. (общий)"
    if meta.get("lead_custom"):
        lead_note += f"; у {meta['lead_custom']} товаров свой срок — указан в таблице"
    info = [
        ("Дата формирования:", meta.get("generated_at", "")),
        ("Источник данных:", meta.get("source_file", "—")),
        ("Горизонт планирования:", f"{meta.get('horizon', '—')} дн."),
        ("Срок поставки:", lead_note),
    ]
    for i, (label, value) in enumerate(info, start=4):
        ws.cell(row=i, column=1, value=label).font = Font(name=FONT, size=10, bold=True)
        ws.cell(row=i, column=3, value=value).font = BASE_FONT
        ws.merge_cells(start_row=i, start_column=3, end_row=i, end_column=8)

    head = 9
    _header_row(ws, head, [
        "№", "Товар", "Что делать", "Остаток,\nшт", "В пути,\nшт",
        "Срок\nпоставки, дн.", "Заказать\nдо", "К заказу,\nшт",
        "Цена закупки,\n₽", "Сумма,\n₽", "Хватит\nдо",
    ])

    # только то, что действительно нужно заказать; сначала самое срочное
    order_rank = {"critical": 0, "alert": 1, "ok": 2, "excess": 3}
    need = [it for it in items if (_num(it.get("recommended_order"), 0) or 0) > 0]
    need.sort(key=lambda it: (order_rank.get(it.get("status"), 9),
                              _num(it.get("order_in_days"), 9999),
                              -(_num(it.get("recommended_order"), 0) or 0)))

    def _d(v):
        """ГГГГ-ММ-ДД -> ДД.ММ.ГГГГ для документа."""
        if not v:
            return "—"
        try:
            return pd.Timestamp(v).strftime("%d.%m.%Y")
        except (ValueError, TypeError):
            return str(v)

    r = head + 1
    for n, it in enumerate(need, start=1):
        sku = str(it.get("sku", ""))
        status = it.get("status", "ok")
        qty = _num(it.get("recommended_order"), 0) or 0
        cost = _num(costs.get(sku))
        stock = _num(it.get("stock"))
        transit = _num(it.get("in_transit"), 0) or 0
        lead = _num(it.get("lead_time_days"))
        gap = int(_num(it.get("stockout_gap_days"), 0) or 0)
        oid = _num(it.get("order_in_days"))

        if gap > 0:
            todo = f"Срочно (без товара ~{gap} дн.)"
        elif oid is not None and oid <= 0:
            todo = "Заказать сегодня"
        else:
            todo = it.get("action") or URGENCY_TEXT.get(status, status)

        ws.cell(row=r, column=1, value=n)
        ws.cell(row=r, column=2, value=sku)
        ws.cell(row=r, column=3, value=todo)
        ws.cell(row=r, column=4, value=round(stock) if stock is not None else "нет данных")
        ws.cell(row=r, column=5, value=round(transit) if transit else "—")
        ws.cell(row=r, column=6, value=round(lead) if lead is not None else "—")
        ws.cell(row=r, column=7, value=_d(it.get("order_by_date")))
        ws.cell(row=r, column=8, value=round(qty))
        ws.cell(row=r, column=9, value=round(cost, 2) if cost is not None else None)
        # Сумма — формула, а не число: снабженец правит количество,
        # и стоимость пересчитывается сама.
        ws.cell(row=r, column=10, value=f"=H{r}*I{r}")
        ws.cell(row=r, column=11, value=_d(it.get("covered_until")))

        for col in range(1, 12):
            c = ws.cell(row=r, column=col)
            c.font = BASE_FONT
            c.border = BORDER
            c.alignment = Alignment(
                horizontal="left" if col in (2, 3) else "center",
                vertical="center", wrap_text=(col == 3))
        ws.cell(row=r, column=4).number_format = QTY
        ws.cell(row=r, column=8).number_format = QTY
        ws.cell(row=r, column=9).number_format = MONEY
        ws.cell(row=r, column=10).number_format = MONEY

        if gap > 0:
            ws.cell(row=r, column=3).fill = URGENCY_FILL["critical"]
        elif status in URGENCY_FILL:
            ws.cell(row=r, column=3).fill = URGENCY_FILL[status]
        r += 1

    if not need:
        ws.cell(row=r, column=1,
                value="Заказ не требуется: текущих остатков хватает на весь "
                      "горизонт планирования по всем позициям.").font = BASE_FONT
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=11)
        r += 1
        total_row = None
    else:
        total_row = r
        ws.cell(row=r, column=2, value=f"ИТОГО позиций: {len(need)}").font = TOTAL_FONT
        ws.cell(row=r, column=9, value="Сумма заказа:").font = TOTAL_FONT
        ws.cell(row=r, column=9).alignment = Alignment(horizontal="right")
        total = ws.cell(row=r, column=10, value=f"=SUM(J{head + 1}:J{r - 1})")
        total.font = TOTAL_FONT
        total.number_format = MONEY
        total.border = Border(top=Side(style="double", color="1F5F5B"))
        r += 2

    # место для подписей: документ уходит на согласование
    ws.cell(row=r + 1, column=2, value="Составил: ______________________").font = BASE_FONT
    ws.cell(row=r + 1, column=6, value="Согласовал: ______________________").font = BASE_FONT

    note = ws.cell(row=r + 3, column=1, value=(
        "Количество к заказу рассчитано как прогноз спроса на срок поставки "
        "и период пересмотра плюс страховой запас, за вычетом текущего "
        "остатка. Срочность: «Критично» — запаса не хватит до прихода "
        "поставки; «Срочно» — остаток ниже точки заказа; «Плановое» — "
        "остаток ещё выше точки заказа, это дозаказ до целевого уровня. "
        "«Заказать до» — последняя дата, когда заказ ещё успеет прийти до "
        "исчерпания остатка с учётом срока поставки именно этого товара; "
        "«Хватит до» — до какой даты хватит товара, если заказать указанное "
        "количество. Цены закупки взяты из загруженной выгрузки продаж. "
        "Столбец «Сумма» и итог — формулы: при изменении количества "
        "пересчитываются автоматически."))
    note.font = MUTED_FONT
    note.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=r + 3, start_column=1, end_row=r + 4, end_column=11)

    _widths(ws, {"A": 5, "B": 40, "C": 24, "D": 10, "E": 9, "F": 12,
                 "G": 13, "H": 11, "I": 14, "J": 16, "K": 13})
    ws.freeze_panes = f"A{head + 1}"
    if need:
        ws.auto_filter.ref = f"A{head}:K{head + len(need)}"
    ws.print_title_rows = f"{head}:{head}"
    return len(need)


# =============================================================================
#  Лист 2 — полный план
# =============================================================================
def _sheet_plan(wb: Workbook, items: list[dict]) -> None:
    ws = wb.create_sheet("Полный план")

    ws["A1"] = "ПЛАН ПОПОЛНЕНИЯ ПО ВСЕМ ТОВАРАМ"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:N1")
    ws["A2"] = ("Включая позиции, по которым заказ сейчас не требуется, — "
                "для контроля и планирования")
    ws["A2"].font = MUTED_FONT
    ws.merge_cells("A2:K2")

    head = 4
    _header_row(ws, head, [
        "Товар", "Статус", "Остаток,\nшт", "Прогноз спроса\nза горизонт",
        "Верхняя\nграница", "Страховой\nзапас", "Точка\nзаказа",
        "К заказу,\nшт", "Хватит\nна, дней", "Срок\nпоставки, дн.",
        "Заказать\nдо", "Хватит после\nзаказа до", "Режим\nпрогноза", "Примечание",
    ])

    mode_ru = {"fine_tune": "дообучение", "global": "общая модель",
               "stats": "статистика", "reject": "отказ"}

    r = head + 1
    for it in items:
        dos = _num(it.get("days_of_supply"))
        stock = _num(it.get("stock"))
        vals = [
            str(it.get("sku", "")),
            STATUS_TEXT.get(it.get("status"), it.get("status")),
            round(stock) if stock is not None else "нет данных",
            round(_num(it.get("forecast_horizon_q50"), 0) or 0),
            round(_num(it.get("forecast_horizon_q90"), 0) or 0),
            round(_num(it.get("safety_stock"), 0) or 0),
            round(_num(it.get("reorder_point"), 0) or 0),
            round(_num(it.get("recommended_order"), 0) or 0),
            round(dos) if dos is not None and dos >= 0 else "—",
            round(_num(it.get("lead_time_days"))) if _num(it.get("lead_time_days")) is not None else "—",
            _fmt_date(it.get("order_by_date")),
            _fmt_date(it.get("covered_until")),
            mode_ru.get(it.get("mode"), it.get("mode", "")),
            it.get("note", ""),
        ]
        for col, v in enumerate(vals, start=1):
            c = ws.cell(row=r, column=col, value=v)
            c.font = BASE_FONT
            c.border = BORDER
            c.alignment = Alignment(
                horizontal="left" if col in (1, 13, 14) else "center",
                vertical="center", wrap_text=(col == 14))
            if col in (3, 4, 5, 6, 7, 8):
                c.number_format = QTY
        if it.get("status") in URGENCY_FILL:
            ws.cell(row=r, column=2).fill = URGENCY_FILL[it["status"]]
        r += 1

    _widths(ws, {"A": 42, "B": 12, "C": 11, "D": 15, "E": 12, "F": 12,
                 "G": 11, "H": 11, "I": 11, "J": 12, "K": 13, "L": 15,
                 "M": 15, "N": 44})
    ws.freeze_panes = f"A{head + 1}"
    if len(items):
        ws.auto_filter.ref = f"A{head}:N{head + len(items)}"


# =============================================================================
#  Лист 3 — параметры расчёта
# =============================================================================
def _sheet_params(wb: Workbook, meta: dict, n_order: int, n_total: int) -> None:
    ws = wb.create_sheet("Параметры расчёта")

    ws["A1"] = "НА ЧЁМ ОСНОВАНА ЗАЯВКА"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:C1")
    ws["A2"] = ("Документ самодостаточен: по этим данным расчёт можно "
                "воспроизвести или проверить")
    ws["A2"].font = MUTED_FONT
    ws.merge_cells("A2:C2")

    rep = meta.get("clean_report") or {}
    fit = meta.get("model_fit") or {}

    blocks: list[tuple[str, list[tuple[str, object]]]] = [
        ("Исходные данные", [
            ("Файл", meta.get("source_file", "—")),
            ("Период истории", meta.get("period", "—")),
            ("Строк прочитано", rep.get("rows_read", "—")),
            ("Товаров в файле", rep.get("sku_count", "—")),
        ]),
        ("Очистка данных", [
            ("Дубликатов объединено", rep.get("duplicates_merged", "—")),
            ("Выбросов обрезано", rep.get("outliers_winsorized", "—")),
            ("Пропущенных дат дозаполнено", rep.get("dates_filled", "—")),
            ("Дней дефицита выявлено", rep.get("censored_days", "—")),
            ("Возвратов обнулено", rep.get("returns_zeroed", "—")),
        ]),
        ("Параметры планирования", [
            ("Горизонт планирования, дней", meta.get("horizon", "—")),
            ("Срок поставки, дней", meta.get("lead_time", "—")),
            ("Период пересмотра, дней", meta.get("review_period", "—")),
            ("Позиций к заказу", n_order),
            ("Всего позиций в плане", n_total),
        ]),
        ("Модель прогноза", [
            ("Движок", meta.get("model_backend") or "статистический расчёт"),
            ("Проверена на этих данных", "да" if fit.get("checked") else "нет"),
            ("Ошибка прогноза (WAPE), %", fit.get("wape_after") or fit.get("wape") or "—"),
            ("Смещение прогноза, %", fit.get("bias_after") or fit.get("bias") or "—"),
            ("Покрытие интервала", fit.get("coverage_after") or fit.get("coverage") or "—"),
            ("Что сделано с моделью", {
                "fine_tuned": "дообучена на ваших данных",
                "verified": "проверена, дообучение не потребовалось",
                "none": "использована как есть",
            }.get(fit.get("action"), "—")),
        ]),
    ]

    r = 4
    for title, rows in blocks:
        c = ws.cell(row=r, column=1, value=title)
        c.font = Font(name=FONT, bold=True, size=11, color=INK)
        c.fill = PatternFill("solid", fgColor="EDF2F5")
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=3)
        r += 1
        for label, value in rows:
            ws.cell(row=r, column=1, value=label).font = BASE_FONT
            v = ws.cell(row=r, column=2, value=value)
            v.font = BASE_FONT
            v.alignment = Alignment(horizontal="left")
            r += 1
        r += 1

    method = ws.cell(row=r + 1, column=1, value=(
        "Как считается количество к заказу. Модель прогнозирует спрос не "
        "одним числом, а двумя: медианой и верхней границей. Разница между "
        "ними и есть страховой запас — он берётся из наблюдавшегося разброса "
        "продаж, а не из предположения о нормальном распределении. "
        "Множитель страхового запаса подбирается по каждому товару отдельно, "
        "исходя из его наценки и стоимости хранения: товар с высокой "
        "наценкой и дешёвым хранением получает больший запас, товар с низкой "
        "наценкой — меньший. К заказу назначается разница между целевым "
        "уровнем запаса и тем, что есть на складе и в пути."))
    method.font = MUTED_FONT
    method.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 6, end_column=3)

    _widths(ws, {"A": 38, "B": 46, "C": 24})


# =============================================================================
def build_purchase_order(items: list[dict], costs: dict[str, float],
                         meta: dict) -> bytes:
    """
    Собирает заявку на закупку и возвращает готовый xlsx в виде байтов.

    items — строки плана закупки (build_order_plan), costs — закупочная
    цена по каждому товару, meta — сведения о файле, параметрах расчёта
    и модели для третьего листа.
    """
    meta = dict(meta or {})
    meta.setdefault("generated_at", datetime.now().strftime("%d.%m.%Y %H:%M"))

    wb = Workbook()
    n_order = _sheet_request(wb, items, costs or {}, meta)
    _sheet_plan(wb, items)
    _sheet_params(wb, meta, n_order, len(items))

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def order_filename(prefix: str = "Заявка_на_закупку") -> str:
    return f"{prefix}_{datetime.now():%Y-%m-%d}.xlsx"
