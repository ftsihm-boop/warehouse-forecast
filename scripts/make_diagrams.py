"""
Генератор диаграмм для пояснительной записки.

ЗАЧЕМ СКРИПТ, А НЕ РИСОВАЛКА
----------------------------
Диаграммы в курсовой работе быстро расходятся с кодом: переименовали
модуль — картинка врёт. Здесь они собираются из одного файла, поэтому
после изменений в проекте достаточно перезапустить скрипт, и все
рисунки снова соответствуют реальности.

Второй довод — воспроизводимость. Скрипт сам по себе является
приложением к записке: любой проверяющий может запустить его и
получить те же изображения, а не верить картинкам на слово.

ЗАПУСК
------
    python scripts/make_diagrams.py

Результат кладётся в docs/diagrams/ в двух форматах: SVG (векторный,
для вставки в Word без потери качества) и PNG (на случай, если
редактор не принимает SVG).

Зависимости: только стандартная библиотека для SVG. PNG получается
рендером в Chromium через Playwright — если его нет, скрипт честно
напишет об этом и оставит только SVG.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "diagrams"

# --- единое оформление всех диаграмм -----------------------------------------
FONT = "'Segoe UI', 'PT Sans', Arial, sans-serif"
INK = "#1F2D3A"          # основной текст
MUTED = "#5A6B7B"        # подписи
LINE = "#8FA3B5"         # стрелки и рамки
BG_PLAIN = "#FFFFFF"

# палитра слоёв: холодная, различимая и в чёрно-белой печати
FILLS = {
    "ui": ("#EAF2FA", "#9CC0DE"),        # интерфейс
    "api": ("#E7F0E9", "#9BC0A6"),       # сервер и API
    "core": ("#F2EDF7", "#B7A3CC"),      # бизнес-логика
    "data": ("#FBF3E6", "#D9BC86"),      # данные и файлы
    "ext": ("#F1F3F5", "#B9C3CC"),       # внешнее окружение
    "note": ("#FFFFFF", "#C9D4DE"),
}


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


DEFS = [
    '<defs>',
    '<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"'
    ' markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>',
    '<marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8"'
    ' markerHeight="8" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10" fill="none" stroke="{LINE}"'
    ' stroke-width="1.4"/></marker>',
    '</defs>',
]


# По ГОСТ 2.105 подпись к рисунку ставится ПОД рисунком, в тексте записки,
# а не внутри изображения. Поэтому заголовок в картинку не вписывается:
# он был бы вторым и дублировал бы подпись. Для презентации заголовок
# можно вернуть — EMBED_TITLE = True.
EMBED_TITLE = False
TITLE_CROP = 32          # столько пустоты сверху срезается вместе с заголовком


def head(w: int, h: int, title: str) -> list[str]:
    if not EMBED_TITLE:
        return [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}"'
            f' height="{h - TITLE_CROP}" viewBox="0 {TITLE_CROP} {w}'
            f' {h - TITLE_CROP}" font-family="{FONT}">',
            f'<rect y="{TITLE_CROP}" width="{w}" height="{h}" fill="{BG_PLAIN}"/>',
            *DEFS,
        ]
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">',
        f'<rect width="{w}" height="{h}" fill="{BG_PLAIN}"/>',
        *DEFS,
        f'<text x="{w // 2}" y="34" text-anchor="middle" font-size="21"'
        f' font-weight="600" fill="{INK}">{esc(title)}</text>',
    ]


def box(x: int, y: int, w: int, h: int, title: str, lines: list[str] | None = None,
        kind: str = "core", r: int = 9, bold: bool = True,
        title_size: int = 14, line_size: int = 11.5, dash: bool = False) -> list[str]:
    fill, stroke = FILLS[kind]
    d = ' stroke-dasharray="6 4"' if dash else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"'
           f' stroke="{stroke}" stroke-width="1.6"{d}/>']
    lines = lines or []
    cx = x + w // 2
    if lines:
        ty = y + 24
    else:
        ty = y + h // 2 + 5
    out.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{title_size}"'
               f' font-weight="{"600" if bold else "400"}" fill="{INK}">{esc(title)}</text>')
    for i, ln in enumerate(lines):
        out.append(f'<text x="{cx}" y="{ty + 19 + i * 16}" text-anchor="middle"'
                   f' font-size="{line_size}" fill="{MUTED}">{esc(ln)}</text>')
    return out


def lbox(x: int, y: int, w: int, h: int, title: str, lines: list[str],
         kind: str = "core", title_size: int = 13, line_size: int = 11) -> list[str]:
    """Блок с выравниванием текста по левому краю — для классов и сущностей."""
    fill, stroke = FILLS[kind]
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}"'
           f' stroke="{stroke}" stroke-width="1.6"/>',
           f'<line x1="{x}" y1="{y + 30}" x2="{x + w}" y2="{y + 30}"'
           f' stroke="{stroke}" stroke-width="1.3"/>',
           f'<text x="{x + w // 2}" y="{y + 21}" text-anchor="middle"'
           f' font-size="{title_size}" font-weight="600" fill="{INK}">{esc(title)}</text>']
    for i, ln in enumerate(lines):
        out.append(f'<text x="{x + 11}" y="{y + 49 + i * 16}" font-size="{line_size}"'
                   f' fill="{MUTED}">{esc(ln)}</text>')
    return out


def halo(x: int, y: int, text: str, size: float = 11,
         anchor: str = "middle") -> str:
    """Белая подложка под текстом, чтобы он не перечёркивался линиями."""
    w = len(text) * size * 0.54 + 12
    h = size * 1.55
    if anchor == "middle":
        rx = x - w / 2
    elif anchor == "end":
        rx = x - w
    else:
        rx = x - 6
    return (f'<rect x="{rx:.0f}" y="{y - size:.0f}" width="{w:.0f}"'
            f' height="{h:.0f}" fill="{BG_PLAIN}"/>')


def arrow(x1: int, y1: int, x2: int, y2: int, label: str = "",
          dash: bool = False, marker: str = "a", lx: int | None = None,
          ly: int | None = None, anchor: str = "middle",
          size: float = 11, plate: bool = True) -> list[str]:
    d = ' stroke-dasharray="5 4"' if dash else ""
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}"'
           f' stroke-width="1.6" marker-end="url(#{marker})"{d}/>']
    if label:
        mx = lx if lx is not None else (x1 + x2) // 2
        my = ly if ly is not None else (y1 + y2) // 2 - 7
        if plate:
            out.append(halo(mx, my, label, size, anchor))
        out.append(f'<text x="{mx}" y="{my}" text-anchor="{anchor}" font-size="{size}"'
                   f' fill="{MUTED}">{esc(label)}</text>')
    return out


def caption(x: int, y: int, text: str, size: float = 11.5,
            anchor: str = "start", italic: bool = True) -> list[str]:
    st = ' font-style="italic"' if italic else ""
    return [f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}"'
            f' fill="{MUTED}"{st}>{esc(text)}</text>']


def band(x: int, y: int, w: int, h: int, label: str) -> list[str]:
    """Полоса-слой с подписью сбоку."""
    return [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none"'
            f' stroke="{LINE}" stroke-width="1.1" stroke-dasharray="7 5"/>',
            f'<text x="{x + 12}" y="{y + 18}" font-size="11.5" font-weight="600"'
            f' fill="{MUTED}" letter-spacing="0.6">{esc(label.upper())}</text>']


# =============================================================================
# 1. Архитектура системы
# =============================================================================
def d_architecture() -> tuple[str, int, int]:
    W, H = 1240, 810
    s = head(W, H, "Рисунок 1 — Архитектура информационной системы")
    s += caption(W // 2, 56, "Трёхуровневая клиент-серверная архитектура; "
                 "обмен по HTTP, формат JSON", anchor="middle")

    s += band(40, 78, W - 80, 118, "уровень представления")
    s += box(80, 104, 330, 74, "Веб-интерфейс (web/index.html)",
             ["одностраничное приложение, без сборки"], "ui")
    s += box(450, 104, 300, 74, "Загрузка файла",
             ["drag-and-drop, индикатор этапов"], "ui")
    s += box(790, 104, 370, 74, "Панели результата",
             ["план закупки, очистка, эффект"], "ui")

    s += band(40, 214, W - 80, 150, "уровень приложения")
    s += box(80, 240, 250, 104, "HTTP-сервер",
             ["scripts/serve.py", "стандартная библиотека"], "api")
    s += box(360, 240, 400, 104, "REST API",
             ["/upload  /history  /forecast", "/order-plan  /economic-impact",
              "/clean-report  /upload-progress"], "api")
    s += box(790, 240, 370, 104, "Сессии пользователей",
             ["очищенные данные в памяти", "модель сессии, TTL 4 часа"], "api")

    s += band(40, 382, W - 80, 236, "уровень бизнес-логики (пакет src/)")
    row1 = [("ingest.py", "распознавание", "колонок"),
            ("cleaning.py", "очистка", "и дозаполнение"),
            ("demand_classes.py", "классификация", "спроса"),
            ("features.py", "признаки", "для модели")]
    for i, (n, a, b) in enumerate(row1):
        s += box(80 + i * 273, 408, 245, 88, n, [a, b], "core")
    row2 = [("model.py", "градиентный бустинг", "квантили 0.5 и 0.9"),
            ("model_fit.py", "проверка и", "дообучение"),
            ("inventory.py", "точка заказа,", "страховой запас"),
            ("backtest.py", "бэктест и", "экономический эффект")]
    for i, (n, a, b) in enumerate(row2):
        s += box(80 + i * 273, 512, 245, 88, n, [a, b], "core")

    s += band(40, 636, W - 80, 136, "уровень данных")
    s += box(80, 662, 300, 86, "Файл пользователя",
             ["CSV / XLSX / выгрузка 1С"], "data")
    s += box(420, 662, 300, 86, "Обученная модель",
             ["models/global/"], "data")
    s += box(760, 662, 400, 86, "Результаты расчёта",
             ["план закупки, отчёты — в ответе API"], "data")

    s += arrow(620, 178, 620, 240, "HTTP, multipart и JSON")
    s += arrow(620, 344, 620, 408, "вызовы функций")
    s += arrow(620, 600, 620, 662, "чтение и запись")
    return "\n".join(s + ["</svg>"]), W, H


# =============================================================================
# 2. Поток обработки данных
# =============================================================================
def d_dataflow() -> tuple[str, int, int]:
    W, H = 1240, 760
    s = head(W, H, "Рисунок 2 — Поток обработки данных")
    s += caption(W // 2, 56, "От файла пользователя до плана закупки "
                 "и экономического обоснования", anchor="middle")

    s += box(60, 92, 210, 80, "Файл продаж", ["CSV, XLSX, 1С"], "data")
    s += box(320, 92, 230, 80, "Распознавание", ["шапка, колонки,", "форматы дат"], "core")
    s += box(600, 92, 230, 80, "Очистка", ["дубли, выбросы,", "пропуски дат"], "core")
    s += box(880, 92, 300, 80, "Разметка дефицита",
             ["продажи 0 при остатке 0 —", "спрос был выше факта"], "core")
    s += arrow(270, 132, 320, 132)
    s += arrow(550, 132, 600, 132)
    s += arrow(830, 132, 880, 132)

    s += box(880, 222, 300, 84, "Классификация спроса",
             ["ADI и CV²: SMOOTH, ERRATIC,", "INTERMITTENT, LUMPY + ABC"], "core")
    s += arrow(1030, 172, 1030, 222)

    s += box(600, 222, 230, 84, "Признаки",
             ["лаги, скользящие,", "календарь РФ"], "core")
    s += arrow(880, 264, 830, 264)

    s += box(600, 356, 230, 96, "Модель",
             ["градиентный бустинг", "квантили 0.5 и 0.9"], "core")
    s += arrow(715, 306, 715, 356)

    s += box(250, 356, 270, 96, "Дообучение",
             ["проверка на данных", "пользователя, затем", "обучение"], "core")
    s += arrow(600, 386, 520, 386, "проверка на данных", ly=378)
    s += arrow(520, 424, 600, 424, "обновлённая модель", dash=True, ly=416)

    s += box(880, 356, 300, 96, "Прогноз спроса",
             ["на горизонт планирования", "медиана и верхняя граница"], "core")
    s += arrow(830, 404, 880, 404)

    s += box(880, 502, 300, 96, "План закупки",
             ["точка заказа, страховой", "запас, объём заказа"], "api")
    s += arrow(1030, 452, 1030, 502)

    s += box(510, 502, 320, 96, "Экономический эффект",
             ["бэктест: ручное планирование", "против системы"], "api")
    s += arrow(880, 550, 830, 550)

    s += box(160, 502, 300, 96, "Подбор запаса",
             ["по каждому товару свой", "коэффициент; убыточные —", "на ручном правиле"], "core")
    s += arrow(510, 550, 460, 550)

    s += box(510, 640, 320, 74, "Отчёт пользователю",
             ["таблица, графики, выгрузка"], "ui")
    s += arrow(670, 598, 670, 640)
    return "\n".join(s + ["</svg>"]), W, H


# =============================================================================
# 3. Диаграмма вариантов использования
# =============================================================================
def d_usecase() -> tuple[str, int, int]:
    W, H = 1180, 780
    s = head(W, H, "Рисунок 3 — Диаграмма вариантов использования")

    def actor(x: int, y: int, name: str, role: str) -> list[str]:
        return [f'<circle cx="{x}" cy="{y}" r="17" fill="#FFFFFF" stroke="{INK}"'
                ' stroke-width="1.8"/>',
                f'<line x1="{x}" y1="{y + 17}" x2="{x}" y2="{y + 60}" stroke="{INK}"'
                ' stroke-width="1.8"/>',
                f'<line x1="{x - 24}" y1="{y + 32}" x2="{x + 24}" y2="{y + 32}"'
                f' stroke="{INK}" stroke-width="1.8"/>',
                f'<line x1="{x}" y1="{y + 60}" x2="{x - 20}" y2="{y + 92}"'
                f' stroke="{INK}" stroke-width="1.8"/>',
                f'<line x1="{x}" y1="{y + 60}" x2="{x + 20}" y2="{y + 92}"'
                f' stroke="{INK}" stroke-width="1.8"/>',
                f'<text x="{x}" y="{y + 116}" text-anchor="middle" font-size="13.5"'
                f' font-weight="600" fill="{INK}">{esc(name)}</text>',
                f'<text x="{x}" y="{y + 133}" text-anchor="middle" font-size="11"'
                f' fill="{MUTED}">{esc(role)}</text>']

    def uc(cx: int, cy: int, label: str, sub: str = "", w: int = 250,
           h: int = 62) -> list[str]:
        o = [f'<ellipse cx="{cx}" cy="{cy}" rx="{w // 2}" ry="{h // 2}"'
             f' fill="{FILLS["ui"][0]}" stroke="{FILLS["ui"][1]}" stroke-width="1.6"/>']
        if sub:
            o.append(f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="12.5"'
                     f' fill="{INK}">{esc(label)}</text>')
            o.append(f'<text x="{cx}" y="{cy + 15}" text-anchor="middle" font-size="10.5"'
                     f' fill="{MUTED}">{esc(sub)}</text>')
        else:
            o.append(f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-size="12.5"'
                     f' fill="{INK}">{esc(label)}</text>')
        return o

    s += [f'<rect x="300" y="86" width="600" height="600" rx="12" fill="none"'
          f' stroke="{LINE}" stroke-width="1.6"/>',
          f'<text x="600" y="112" text-anchor="middle" font-size="13"'
          f' font-weight="600" fill="{MUTED}">Информационная система '
          f'управления складскими запасами</text>']

    s += actor(140, 175, "Менеджер", "по закупкам")
    s += actor(140, 500, "Аналитик", "сопровождение модели")

    ys = [168, 250, 332, 414, 496, 570, 645]
    s += uc(600, ys[0], "Загрузить данные о продажах", "CSV, XLSX, выгрузка 1С", 300)
    s += uc(600, ys[1], "Просмотреть отчёт об очистке", "", 300)
    s += uc(600, ys[2], "Получить план закупки", "точка заказа и объём", 300)
    s += uc(600, ys[3], "Рассчитать экономический эффект", "", 320)
    s += uc(600, ys[4], "Задать срок поставки и горизонт", "", 320)
    s += uc(600, ys[5], "Выгрузить отчёт", "", 240)
    s += uc(600, ys[6], "Обучить модель на истории", "", 300)

    # менеджер работает со всеми сценариями, кроме обучения модели
    for y in ys[:6]:
        s += arrow(176, 211, 452, y)
    # аналитик отвечает только за обучение
    s += arrow(176, 536, 452, ys[6])

    s += uc(950, 190, "Дообучить под ассортимент", "", 236, 56)
    s += uc(950, 300, "Классифицировать спрос", "", 236, 56)
    s += uc(950, 440, "Подобрать страховой запас", "", 236, 56)
    s += arrow(748, 158, 838, 184, "«include»", dash=True, marker="ao", lx=793, ly=156)
    s += arrow(742, 186, 838, 292, "«include»", dash=True, marker="ao", lx=806, ly=250)
    s += arrow(758, 420, 838, 434, "«include»", dash=True, marker="ao", lx=798, ly=414)

    s += caption(40, 740, "Пунктирные связи «include» — обязательные части "
                 "сценария, выполняемые системой автоматически.")
    return "\n".join(s + ["</svg>"]), W, H


# =============================================================================
# 4. Диаграмма последовательности
# =============================================================================
def d_sequence() -> tuple[str, int, int]:
    W, H = 1300, 830
    s = head(W, H, "Рисунок 4 — Диаграмма последовательности: загрузка файла и расчёт")

    cols = [("Менеджер", 110), ("Браузер", 300), ("HTTP-сервер", 520),
            ("Очистка", 730), ("Модель", 940), ("Расчёт эффекта", 1170)]
    for name, x in cols:
        s += [f'<rect x="{x - 88}" y="66" width="176" height="42" rx="7"'
              f' fill="{FILLS["ext"][0]}" stroke="{FILLS["ext"][1]}" stroke-width="1.5"/>',
              f'<text x="{x}" y="93" text-anchor="middle" font-size="12.5"'
              f' font-weight="600" fill="{INK}">{esc(name)}</text>',
              f'<line x1="{x}" y1="108" x2="{x}" y2="782" stroke="{LINE}"'
              ' stroke-width="1.1" stroke-dasharray="5 5"/>']
    X = {n: x for n, x in cols}

    def msg(a: str, b: str, y: int, text: str, dash: bool = False) -> list[str]:
        x1, x2 = X[a], X[b]
        right = x2 > x1
        return arrow(x1 + (9 if right else -9), y, x2 + (-9 if right else 9), y,
                     text, dash=dash, ly=y - 9)

    s += msg("Менеджер", "Браузер", 142, "выбирает файл продаж")
    s += msg("Браузер", "HTTP-сервер", 180, "POST /upload (файл, uid)")

    # опрос прогресса идёт параллельно обработке, поэтому вынесен в рамку,
    # которая заканчивается на линии жизни сервера и не задевает правую часть
    s += [f'<rect x="{X["Браузер"] - 76}" y="204" width="{X["HTTP-сервер"] - X["Браузер"] + 76}"'
          f' height="122" rx="8" fill="none" stroke="{FILLS["api"][1]}"'
          ' stroke-width="1.4" stroke-dasharray="6 4"/>',
          halo(X["Браузер"] - 66, 223, "loop [каждые 400 мс, параллельно]", 11, "start"),
          f'<text x="{X["Браузер"] - 66}" y="223" font-size="11" font-weight="600"'
          f' fill="{MUTED}">loop [каждые 400 мс, параллельно]</text>']
    s += msg("Браузер", "HTTP-сервер", 262, "GET /upload-progress")
    s += msg("HTTP-сервер", "Браузер", 302, "этап обработки и процент", dash=True)

    s += msg("HTTP-сервер", "Очистка", 226, "read_table() + clean()")
    s += msg("Очистка", "HTTP-сервер", 266, "канонический набор + отчёт", dash=True)
    s += msg("HTTP-сервер", "Модель", 306, "adapt_to_data(): проверка и дообучение")
    s += msg("Модель", "HTTP-сервер", 346, "модель под этот ассортимент", dash=True)

    s += msg("HTTP-сервер", "Браузер", 390, "session_id, отчёт об очистке", dash=True)

    s += msg("Менеджер", "Браузер", 442, "нажимает «Рассчитать»")
    s += msg("Браузер", "HTTP-сервер", 482, "GET /economic-impact (охват, срок поставки)")
    s += msg("HTTP-сервер", "Расчёт эффекта", 522, "tune_portfolio()")

    s += [f'<rect x="{X["Модель"] - 150}" y="548" width="{X["Расчёт эффекта"] - X["Модель"] + 180}"'
          f' height="128" rx="8" fill="none" stroke="{FILLS["core"][1]}"'
          ' stroke-width="1.4" stroke-dasharray="6 4"/>',
          halo(X["Модель"] - 140, 567, "loop [по каждому товару × сетка коэффициентов]",
               11, "start"),
          f'<text x="{X["Модель"] - 140}" y="567" font-size="11" font-weight="600"'
          f' fill="{MUTED}">loop [по каждому товару × сетка коэффициентов]</text>']
    s += msg("Расчёт эффекта", "Модель", 606, "прогноз на окне бэктеста")
    s += msg("Модель", "Расчёт эффекта", 646, "квантили 0.5 и 0.9", dash=True)

    s += msg("Расчёт эффекта", "HTTP-сервер", 706, "эффект, охват управления, диагностика", dash=True)
    s += msg("HTTP-сервер", "Браузер", 744, "JSON", dash=True)
    s += msg("Браузер", "Менеджер", 782, "карточки эффекта и пояснение", dash=True)
    return "\n".join(s + ["</svg>"]), W, H


# =============================================================================
# 5. Диаграмма классов
# =============================================================================
def d_classes() -> tuple[str, int, int]:
    W, H = 1230, 800
    s = head(W, H, "Рисунок 5 — Диаграмма классов основных модулей")
    s += caption(W // 2, 56, "Система построена функционально, классы-dataclass "
                 "служат контейнерами результата", anchor="middle")

    def hh(n: int) -> int:
        return 30 + n * 16 + 26

    C1, C2, C3, WID = 60, 445, 830, 310
    R1, R2, R3 = 84, 330, 590

    s += lbox(C1, R1, WID, hh(5), "IngestResult", [
        "+ frame: DataFrame", "+ mapping: dict — колонки",
        "+ header_row: int", "+ sheet: str", "+ warnings: list[str]"], "data")
    s += lbox(C2, R1, WID, hh(6), "CleaningReport", [
        "+ rows_read: int", "+ duplicates_merged: int",
        "+ outliers_winsorized: int", "+ censored_days: int",
        "+ dates_filled: int", "+ sku_count: int"], "note")
    s += lbox(C3, R1, WID, hh(5), "DemandProfile", [
        "+ sku: str", "+ adi: float — интервал", "+ cv2: float — вариация",
        "+ demand_class: str", "+ forecastable: bool"], "core")

    s += lbox(C1, R2, WID, hh(6), "QuantileModel", [
        "+ features: list[str]", "+ backend: str", "+ fit(X, y)",
        "+ predict(X) -> (q50, q90)", "+ fine_tune(X, y)",
        "+ save(path) / load(path)"], "core")
    s += lbox(C2, R2, WID, hh(6), "tune_portfolio()", [
        "подбирает коэффициент запаса", "по каждому товару отдельно",
        "и выбирает политику закупки:", "прогноз или ручное правило",
        "", "возвращает сводку и охват"], "api")
    s += lbox(C3, R2, WID, hh(6), "PolicyResult", [
        "+ name: str — политика", "+ avg_stock_units / value",
        "+ stockout_days: int", "+ lost_units / lost_revenue",
        "+ holding_cost: float", "+ total_cost: float"], "api")

    s += lbox(C1, R3, WID, hh(5), "FitReport", [
        "+ checked / suitable: bool", "+ wape / bias / coverage",
        "+ action: str — что сделано", "+ message: str",
        "+ improvement_pct: float"], "note")
    s += lbox(C2, R3, WID, hh(5), "Economics", [
        "+ unit_cost / margin", "+ holding_rate_year: float",
        "+ order_cost: float", "+ price / profit_per_unit",
        "+ from_data(df) — из файла"], "api")
    s += lbox(C3, R3, WID, hh(5), "SkuEconomics", [
        "+ sku: str", "+ unit_cost / unit_price",
        "+ margin_rate: float", "+ service_level: float",
        "+ service_factor: float"], "api")

    y1 = R1 + hh(5) // 2
    s += arrow(C1 + WID, y1, C2, y1, "clean()", dash=True, marker="ao", ly=y1 - 9)
    s += arrow(C2 + WID, y1, C3, y1, "profile_all()", dash=True, marker="ao", ly=y1 - 9)

    y2 = R2 + hh(6) // 2
    s += arrow(C1 + WID, y2, C2, y2, "прогноз", dash=True, marker="ao", ly=y2 - 9)
    s += arrow(C2 + WID, y2, C3, y2, "создаёт", dash=True, marker="ao", ly=y2 - 9)

    y3 = R3 + hh(5) // 2
    s += arrow(C3, y3, C2 + WID, y3, "агрегируется", dash=True, marker="ao", ly=y3 - 9)
    s += arrow(C2 + 160, R3, C2 + 160, R2 + hh(6), "экономика товара",
               dash=True, marker="ao", lx=C2 + 160, ly=R3 - 14)
    s += arrow(C1 + 155, R3, C1 + 155, R2 + hh(6), "проверка создаёт",
               dash=True, marker="ao", lx=C1 + 155, ly=R3 - 14)

    s += caption(60, 772, "Пунктир с открытой стрелкой — зависимость по UML; "
                 "подпись указывает функцию, которая её реализует.")
    return "\n".join(s + ["</svg>"]), W, H


# =============================================================================
# 6. Логическая модель данных
# =============================================================================
def d_er() -> tuple[str, int, int]:
    W, H = 1240, 800
    s = head(W, H, "Рисунок 6 — Логическая модель данных")
    s += caption(W // 2, 56, "Сущности канонического формата; хранение файловое, "
                 "модель приведена для переноса в СУБД", anchor="middle")

    def hh(n: int) -> int:
        return 30 + n * 16 + 24

    s += lbox(60, 96, 290, hh(4), "ТОВАР", [
        "PK  sku — наименование", "     класс спроса (ADI, CV²)",
        "     группа ABC", "     признак прогнозируемости"], "data")
    s += lbox(490, 96, 320, hh(6), "ПРОДАЖА ЗА ДЕНЬ", [
        "PK  date", "PK  FK sku", "     qty — продано, шт",
        "     stock — остаток на конец дня", "     флаги: дефицит, выброс,",
        "     дозаполнение, возврат"], "data")
    s += lbox(950, 96, 250, hh(4), "ЦЕНА ТОВАРА", [
        "PK  FK sku", "     price — цена продажи",
        "     cost — закупочная цена", "     источник: файл или оценка"], "data")

    s += lbox(60, 330, 290, hh(5), "СЕССИЯ РАСЧЁТА", [
        "PK  session_id", "     момент создания",
        "     имя загруженного файла", "     отчёт об очистке",
        "     модель сессии"], "api")
    s += lbox(490, 330, 320, hh(5), "ПРОГНОЗ", [
        "PK  FK sku", "PK  дата начала, горизонт",
        "     q50 — медиана спроса", "     q90 — верхняя граница",
        "     режим: модель или статистика"], "core")
    s += lbox(950, 330, 250, hh(5), "ПЛАН ЗАКУПКИ", [
        "PK  FK sku", "     страховой запас", "     точка заказа",
        "     рекомендуемый объём", "     статус позиции"], "core")

    s += lbox(490, 566, 320, hh(5), "РЕЗУЛЬТАТ БЭКТЕСТА", [
        "PK  FK sku", "     коэффициент запаса",
        "     политика: прогноз или ручная", "     затраты as-is и to-be",
        "     экономический эффект"], "api")

    def rel_h(x1, x2, y, left, right, label):
        """Горизонтальная связь: мощности у концов, подпись над серединой."""
        o = [f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{LINE}"'
             ' stroke-width="1.6"/>']
        mid = (x1 + x2) // 2
        o.append(halo(mid, y - 12, label, 10.5))
        o.append(f'<text x="{mid}" y="{y - 12}" text-anchor="middle" font-size="10.5"'
                 f' font-style="italic" fill="{MUTED}">{esc(label)}</text>')
        for x, c, d in ((x1, left, 1), (x2, right, -1)):
            o.append(f'<text x="{x + 16 * d}" y="{y + 19}" text-anchor="middle"'
                     f' font-size="12.5" font-weight="700" fill="{INK}">{esc(c)}</text>')
        return o

    def rel_v(x, y1, y2, top, bottom, label):
        o = [f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{LINE}"'
             ' stroke-width="1.6"/>']
        mid = (y1 + y2) // 2
        o.append(f'<text x="{x + 12}" y="{mid + 4}" font-size="10.5"'
                 f' font-style="italic" fill="{MUTED}">{esc(label)}</text>')
        for yy, c, d in ((y1, top, 1), (y2, bottom, -1)):
            o.append(f'<text x="{x - 16}" y="{yy + 20 * d}" text-anchor="middle"'
                     f' font-size="12.5" font-weight="700" fill="{INK}">{esc(c)}</text>')
        return o

    s += rel_h(350, 490, 175, "1", "N", "имеет историю")
    s += rel_h(810, 950, 175, "1", "1", "цена выводится")
    s += rel_v(205, 96 + hh(4), 330, "N", "1", "обрабатывается в")
    s += rel_h(350, 490, 400, "1", "N", "порождает")
    s += rel_h(810, 950, 400, "1", "1", "переходит в")
    s += rel_v(650, 330 + hh(5), 566, "1", "N", "проверяется бэктестом")

    s += caption(60, 770, "Данные хранятся в файлах и в оперативной памяти сессии; "
                 "модель показывает, какие таблицы будут созданы при переходе "
                 "на промышленную СУБД.")
    return "\n".join(s + ["</svg>"]), W, H


DIAGRAMS = {
    "01_architecture": d_architecture,
    "02_dataflow": d_dataflow,
    "03_usecase": d_usecase,
    "04_sequence": d_sequence,
    "05_classes": d_classes,
    "06_er_model": d_er,
}


def render_png(items: list[tuple[Path, int, int]]) -> bool:
    """Рендерит SVG в PNG через Chromium. Без него остаются только SVG."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Playwright не установлен — PNG не создаются, SVG на месте.")
        return False

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(device_scale_factor=2)
        for svg_path, w, h in items:
            page.set_viewport_size({"width": w, "height": h})
            page.goto(svg_path.as_uri())
            page.screenshot(path=str(svg_path.with_suffix(".png")))
        browser.close()
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    made: list[tuple[Path, int, int]] = []
    for name, fn in DIAGRAMS.items():
        svg, w, h = fn()
        if not EMBED_TITLE:
            h -= TITLE_CROP      # заголовок срезан — окно рендера тоже ниже
        p = OUT / f"{name}.svg"
        p.write_text(svg, encoding="utf-8")
        made.append((p, w, h))
        print(f"  + {p.relative_to(ROOT)}  ({w}×{h})")

    if render_png(made):
        print(f"\nPNG записаны туда же. Всего диаграмм: {len(made)}")
    print(f"Каталог: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
