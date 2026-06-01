"""Регрессия фикса #2: SqlSelfCorrector не должен сужать месячный диапазон дат
до точечной даты.

Планировщик намеренно строит календарный диапазон (col >= start AND col < end)
и выбрасывает точечный `col = 'YYYY-MM-DD'` из QuerySpec. LLM SqlWriter /
self-corrector иногда возвращают точечный фильтр (и даже выкидывают диапазон),
схлопывая «в феврале» до одного дня. `_enforce_blueprint_date_range` это лечит.
"""

from graph.nodes.sql_pipeline import (
    _enforce_blueprint_date_range,
    _blueprint_date_ranges,
)


_BP = {
    "where_conditions": [
        "report_dt >= '2026-02-01'::date",
        "report_dt < '2026-03-01'::date",
        "is_task = TRUE",
    ]
}


def test_blueprint_date_ranges_groups_by_column():
    ranges = _blueprint_date_ranges(_BP)
    assert set(ranges) == {"report_dt"}
    assert ranges["report_dt"] == [
        "report_dt >= '2026-02-01'::date",
        "report_dt < '2026-03-01'::date",
    ]


def test_point_only_restored_to_range():
    """Self-corrector выкинул диапазон, оставил точечную дату → диапазон возвращается."""
    sql = (
        "SELECT count(*) AS count_all FROM t "
        "WHERE report_dt = '2026-02-01'::date AND is_task = TRUE"
    )
    out, note = _enforce_blueprint_date_range(sql, _BP)
    assert "report_dt >= '2026-02-01'::date" in out
    assert "report_dt < '2026-03-01'::date" in out
    assert "report_dt = '2026-02-01'" not in out
    assert "is_task = TRUE" in out
    assert note


def test_redundant_point_alongside_range_dropped():
    """SqlWriter добавил точечный фильтр поверх диапазона → точечный удаляется."""
    sql = (
        "SELECT count(*) FROM t WHERE report_dt >= '2026-02-01'::date "
        "AND report_dt < '2026-03-01'::date AND is_task = TRUE "
        "AND report_dt = '2026-02-01'::date"
    )
    out, note = _enforce_blueprint_date_range(sql, _BP)
    assert out.count("report_dt = '2026-02-01'") == 0
    assert "report_dt >= '2026-02-01'::date" in out
    assert "report_dt < '2026-03-01'::date" in out
    assert note


def test_correct_range_untouched():
    sql = (
        "SELECT count(*) FROM t WHERE report_dt >= '2026-02-01'::date "
        "AND report_dt < '2026-03-01'::date"
    )
    out, note = _enforce_blueprint_date_range(sql, _BP)
    assert out == sql
    assert note == ""


def test_no_blueprint_range_is_noop():
    sql = "SELECT count(*) FROM t WHERE report_dt = '2026-02-01'::date"
    out, note = _enforce_blueprint_date_range(sql, {"where_conditions": ["is_task = TRUE"]})
    assert out == sql
    assert note == ""


def test_qualified_column_point_restored():
    """Точечный фильтр на квалифицированной колонке тоже распознаётся."""
    bp = {"where_conditions": ["t.report_dt >= '2026-02-01'::date", "t.report_dt < '2026-03-01'::date"]}
    sql = "SELECT count(*) FROM tbl t WHERE t.report_dt = '2026-02-01'::date"
    out, note = _enforce_blueprint_date_range(sql, bp)
    assert ">= '2026-02-01'::date" in out
    assert "< '2026-03-01'::date" in out
    assert note
