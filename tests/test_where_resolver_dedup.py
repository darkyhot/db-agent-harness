"""Дедупликация фильтров в WhereResolver: bool/strring + system_timestamp."""

import pandas as pd

from core.query_ir import FilterSpec
from core.schema_loader import SchemaLoader
from core.where_resolver import (
    _add_unique,
    _apply_exact_filter_specs,
    _column_has_range_predicate,
    _drop_system_timestamp_when_time_axis_present,
    _is_categorical_filter_column,
    _parse_condition,
)


def test_add_unique_dedupes_ilike_case_insensitive():
    """Fix F: ILIKE регистронезависим — '%фактический отток%' и '%Фактический
    отток%' это один фильтр, не два."""
    conditions: list[str] = []
    _add_unique(conditions, "task_subtype ILIKE '%фактический отток%'")
    _add_unique(conditions, "task_subtype ILIKE '%Фактический отток%'")
    assert len(conditions) == 1


def _funnel_loader_for_categorical(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel_task"],
        "description": ["Воронка задач"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 3,
        "table_name": ["sale_funnel_task"] * 3,
        "column_name": ["report_dt", "task_subtype", "is_task_closed"],
        "dType": ["date", "varchar", "boolean"],
        "description": ["Отчётная дата", "Подтип задачи", "Признак закрытия задачи"],
        "is_primary_key": [False, False, False],
        "unique_perc": [1.0, 5.0, 2.0],
        "not_null_perc": [100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_is_categorical_filter_column(tmp_path):
    """Fix G: только text-колонка «покрывает» таблицу для F6; дата и булев флаг —
    нет (иначе report_dt/is_task затеняют task_subtype)."""
    loader = _funnel_loader_for_categorical(tmp_path)
    assert _is_categorical_filter_column(loader, "dm.sale_funnel_task", "task_subtype") is True
    assert _is_categorical_filter_column(loader, "dm.sale_funnel_task", "report_dt") is False
    assert _is_categorical_filter_column(loader, "dm.sale_funnel_task", "is_task_closed") is False


def test_exact_point_date_skipped_when_same_column_range_exists():
    """Регрессия agent(20): точечный `report_dt = '2026-02-01'` поверх уже
    построенного месячного диапазона редундантен и сужает месяц до одного дня —
    пропускаем его в пользу диапазона (даже когда time_range=None, т.к. диапазон
    пришёл из текста в base_conditions)."""
    conditions = [
        "report_dt >= '2026-02-01'::date",
        "report_dt < '2026-03-01'::date",
    ]
    selected_columns = {"dm.sale_funnel_task": {"filter": ["report_dt"]}}
    specs = [FilterSpec(target="report_dt", operator="=", value="2026-02-01")]
    applied = _apply_exact_filter_specs(
        conditions, selected_columns, specs, schema_loader=None, time_range=None,
    )
    assert applied == []  # точка пропущена
    assert "report_dt = '2026-02-01'" not in " ".join(conditions)
    assert _column_has_range_predicate(conditions, "report_dt") is True


def test_exact_point_date_applied_when_no_range():
    """Без диапазона и без time_range точечный date-фильтр остаётся (никакого
    over-suppress)."""
    conditions: list[str] = []
    selected_columns = {"dm.sale_funnel_task": {"filter": ["report_dt"]}}
    specs = [FilterSpec(target="report_dt", operator="=", value="2026-02-01")]
    applied = _apply_exact_filter_specs(
        conditions, selected_columns, specs, schema_loader=None, time_range=None,
    )
    assert applied == ["query_spec:0"]
    assert any("report_dt = '2026-02-01'" in c for c in conditions)


def _loader_with_outflow(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_dwh_fact_outflow"],
        "description": ["Факт оттока"],
        "grain": ["day"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 3,
        "table_name": ["uzp_dwh_fact_outflow"] * 3,
        "column_name": ["report_dt", "inserted_dttm", "is_task"],
        "dType": ["date", "timestamp", "bool"],
        "description": [
            "Отчетная дата",
            "Время вставки в систему",
            "Признак задачи",
        ],
        "is_primary_key": [False, False, False],
        "unique_perc": [0.5, 99.0, 0.01],
        "not_null_perc": [99.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_parse_condition_normalizes_bool_literal():
    assert _parse_condition("is_task = 'True'") == ("is_task", "=", "true")
    assert _parse_condition("is_task = true") == ("is_task", "=", "true")
    assert _parse_condition("is_task = 1") == ("is_task", "=", "true")
    assert _parse_condition("is_task = false") == ("is_task", "=", "false")


def test_parse_condition_strips_date_cast():
    assert _parse_condition("report_dt >= '2026-02-01'::date") == (
        "report_dt", ">=", "2026-02-01",
    )


def test_add_unique_dedupes_string_vs_bool_literal():
    conditions: list[str] = []
    _add_unique(conditions, "is_task = true")
    _add_unique(conditions, "is_task = 'True'")
    _add_unique(conditions, "is_task = 1")
    assert conditions == ["is_task = true"]


def test_drop_system_timestamp_when_time_axis_present(tmp_path):
    loader = _loader_with_outflow(tmp_path)
    selected = {
        "dm.uzp_dwh_fact_outflow": {
            "filter": ["report_dt", "inserted_dttm"],
        }
    }
    conditions = [
        "report_dt >= '2026-02-01'",
        "inserted_dttm >= '2026-02-01'",
        "report_dt < '2026-03-01'",
        "inserted_dttm < '2026-03-01'",
    ]
    cleaned = _drop_system_timestamp_when_time_axis_present(
        conditions, selected, schema_loader=loader,
    )
    assert "report_dt >= '2026-02-01'" in cleaned
    assert "report_dt < '2026-03-01'" in cleaned
    assert all("inserted_dttm" not in c for c in cleaned)


def test_drop_system_timestamp_keeps_when_no_time_axis(tmp_path):
    loader = _loader_with_outflow(tmp_path)
    selected = {"dm.uzp_dwh_fact_outflow": {"filter": ["inserted_dttm"]}}
    conditions = ["inserted_dttm >= '2026-02-01'"]
    cleaned = _drop_system_timestamp_when_time_axis_present(
        conditions, selected, schema_loader=loader,
    )
    assert cleaned == conditions
