"""
Тесты пайплайна.

    pytest -q                 (или python tests/test_pipeline.py без pytest)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import clean, pick_mode  # noqa: E402
from src.features import build_features, build_training_set, time_split  # noqa: E402
from src.ingest import (  # noqa: E402
    IngestResult, excel_serial_to_date, match_column, parse_date, parse_number,
    read_table, sku_key,
)
from src.inventory import STATUS_ALERT, STATUS_CRITICAL, compute_order  # noqa: E402
from src.schema import (  # noqa: E402
    DATE, FLAG_CENSORED, MODE_FINE_TUNE, MODE_GLOBAL, MODE_REJECT, MODE_STATS,
    QTY, SKU, STOCK,
)


# --- слой 2: парсинг значений -------------------------------------------------

def test_parse_number_formats():
    assert parse_number("1 234,56") == 1234.56
    assert parse_number("1\u00a0234,56") == 1234.56
    assert parse_number("1,234.56") == 1234.56
    assert parse_number("12 шт") == 12
    assert parse_number(42) == 42.0
    for bad in ("—", "н/д", "", None, "abc"):
        assert np.isnan(parse_number(bad)), bad


def test_parse_date_formats():
    target = pd.Timestamp("2025-08-21")
    assert parse_date(45890) == target                    # Excel serial
    assert parse_date("21.08.2025") == target
    assert parse_date("2025-08-21") == target
    assert parse_date("2025-08-21 00:00:00") == target
    assert parse_date("21 августа 2025") == target
    assert np.isnan(parse_date("ИТОГО"))
    assert excel_serial_to_date(45890) == target


def test_column_synonyms():
    assert match_column("Дата") == DATE
    assert match_column("Номенклатура") == SKU
    assert match_column("Продано, шт.") == QTY
    assert match_column("Остаток на складе") == STOCK
    assert match_column("item_cnt_day") == QTY
    assert match_column("Комментарий менеджера") is None


def test_sku_key_merges_variants():
    assert sku_key("Молоко (1 литр)") == sku_key("Молоко(1 л)")
    assert sku_key(" Бананы  (1 кг) ") == sku_key("Бананы (1 кг)")
    assert sku_key("Хлеб") != sku_key("Пиво")


# --- слой 1: поиск шапки ------------------------------------------------------

def test_header_below_service_rows(tmp_path: Path):
    """Шапка в 5-й строке, как в выгрузке 1С."""
    rows = [["Отчёт о продажах", "", "", ""],
            ["Период: 01.01.2025 - 31.01.2025", "", "", ""],
            ["", "", "", ""],
            ["", "", "", ""],
            ["Дата", "Товар", "Продано", "Остаток на складе"]]
    for i in range(20):
        rows.append([f"{(i % 28) + 1:02d}.01.2025", "Молоко", i % 7 + 1, 50 - i])
    p = tmp_path / "report.csv"
    pd.DataFrame(rows).to_csv(p, index=False, header=False, encoding="utf-8-sig")

    res = read_table(p)
    assert res.header_row == 4
    assert set(res.mapping) == {DATE, SKU, QTY, STOCK}


# --- слои 3-4: логика ---------------------------------------------------------

def _frame(rows) -> IngestResult:
    df = pd.DataFrame(rows, columns=[DATE, SKU, QTY, STOCK])
    df[DATE] = pd.to_datetime(df[DATE])
    return IngestResult(frame=df, sheet_name=None, header_row=0, mapping={})


def test_duplicates_and_returns():
    res = _frame([
        ("2025-01-01", "A", 5, 10), ("2025-01-01", "A", 3, 10),   # дубль
        ("2025-01-02", "A", -2, 8),                                # возврат
        ("2025-01-03", "A", 4, 6),
    ])
    df, rep = clean(res, winsorize=False)
    assert rep.duplicates_merged == 1
    assert rep.returns_zeroed == 1
    assert df.loc[df[DATE] == "2025-01-01", QTY].iloc[0] == 8   # 5 + 3
    assert (df[QTY] >= 0).all()


def test_missing_dates_filled_with_zero():
    res = _frame([("2025-01-01", "A", 5, 10),
                  ("2025-01-05", "A", 4, 6)])          # пропущены 2, 3, 4
    df, rep = clean(res, winsorize=False)
    assert rep.dates_filled == 3
    assert len(df) == 5
    assert df[QTY].sum() == 9


def test_censored_demand_detected():
    """Продажи 0 при нулевом остатке — это дефицит, а не отсутствие спроса."""
    res = _frame([("2025-01-01", "A", 5, 10),
                  ("2025-01-02", "A", 0, 0),           # дефицит
                  ("2025-01-03", "A", 0, 7)])          # честный ноль
    df, rep = clean(res, winsorize=False)
    assert rep.censored_days == 1
    flags = df.sort_values(DATE)[FLAG_CENSORED].tolist()
    assert flags == [False, True, False]


def test_outlier_winsorized_not_dropped():
    rows = [(f"2025-01-{d:02d}", "A", 10, 100) for d in range(1, 29)]
    rows.append(("2025-01-29", "A", 5000, 100))        # выброс
    df, rep = clean(_frame(rows), winsorize=True)
    assert rep.outliers_winsorized == 1
    assert len(df) == 29                               # строку не удалили
    assert df[QTY].max() < 5000                        # значение обрезали


def test_similar_sku_merged():
    rows = ([("2025-01-01", "Молоко (1 литр)", 5, 10)] * 3
            + [("2025-01-02", "Молоко(1 л)", 4, 8)])
    df, rep = clean(_frame(rows), winsorize=False)
    assert df[SKU].nunique() == 1
    assert rep.sku_merged


# --- выбор режима (вариант C) -------------------------------------------------

def test_mode_thresholds():
    assert pick_mode(5) == MODE_REJECT
    assert pick_mode(30) == MODE_STATS
    assert pick_mode(120) == MODE_GLOBAL
    assert pick_mode(400) == MODE_FINE_TUNE


# --- признаки -----------------------------------------------------------------

def _synthetic(days: int = 200, sku: str = "A") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    d = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame({
        DATE: d, SKU: sku,
        QTY: np.abs(rng.normal(20, 4, days)).round(),
        STOCK: 200.0, FLAG_CENSORED: False,
    })


def test_features_have_no_leakage():
    """Признаки на дату T не должны зависеть от продаж в день T."""
    df = _synthetic(120)
    f1 = build_features(df, horizon=7, with_target=False)
    df2 = df.copy()
    df2.loc[df2.index[-1], QTY] = 9999          # меняем ТОЛЬКО последний день
    f2 = build_features(df2, horizon=7, with_target=False)

    cols = [c for c in f1.columns if c.startswith(("lag_", "mean_", "std_"))]
    last1 = f1.iloc[-1][cols].fillna(-1).to_numpy(dtype=float)
    last2 = f2.iloc[-1][cols].fillna(-1).to_numpy(dtype=float)
    assert np.allclose(last1, last2), "признаки подглядывают в текущий день"


def test_target_is_forward_sum():
    df = _synthetic(60)
    f = build_features(df, horizon=7, with_target=True).sort_values(DATE)
    qty = df.sort_values(DATE)[QTY].to_numpy()
    row = f.iloc[10]
    assert np.isclose(row["target_abs"], qty[11:18].sum())


def test_time_split_has_no_overlap():
    df = build_training_set(_synthetic(300), [7, 14])
    tr, te = time_split(df, test_days=60)
    assert tr[DATE].max() < te[DATE].min()


# --- расчёт закупки -----------------------------------------------------------

def test_safety_stock_from_quantiles():
    p = compute_order("A", q50_horizon=300, q90_horizon=420, horizon_days=30,
                      stock=500, lead_time_days=7, review_period_days=7)
    # спрос за 7 дней: q50=70, q90=98 -> SS=28, ROP=98
    assert np.isclose(p.demand_lead_q50, 70, atol=0.5)
    assert np.isclose(p.safety_stock, 28, atol=0.5)
    assert np.isclose(p.reorder_point, 98, atol=0.5)


def test_critical_status_when_stock_below_lead_demand():
    p = compute_order("A", 300, 420, 30, stock=10, lead_time_days=7)
    assert p.status == STATUS_CRITICAL
    assert p.recommended_order > 0


def test_alert_when_below_reorder_point():
    p = compute_order("A", 300, 420, 30, stock=90, lead_time_days=7)
    assert p.status == STATUS_ALERT


def test_no_order_when_stock_is_enough():
    p = compute_order("A", 300, 320, 30, stock=400, lead_time_days=7,
                      review_period_days=7)
    assert p.recommended_order == 0


def test_eoq_raises_small_order():
    small = compute_order("A", 300, 320, 30, stock=130, lead_time_days=7)
    with_eoq = compute_order("A", 300, 320, 30, stock=130, lead_time_days=7,
                             order_cost=500, holding_cost_per_unit_year=20)
    assert with_eoq.recommended_order >= small.recommended_order
    assert with_eoq.eoq is not None


def test_order_multiple_rounds_up():
    p = compute_order("A", 300, 420, 30, stock=0, lead_time_days=7,
                      order_multiple=12)
    assert p.recommended_order % 12 == 0


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    import inspect
    import tempfile

    failed = 0
    for name, fn in fns:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  OK    {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} тестов пройдено")
    sys.exit(1 if failed else 0)
