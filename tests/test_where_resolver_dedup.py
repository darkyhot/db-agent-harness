"""Дедупликация фильтров в WhereResolver: bool/strring + system_timestamp."""

import pandas as pd

from core.schema_loader import SchemaLoader
from core.where_resolver import (
    _add_unique,
    _drop_system_timestamp_when_time_axis_present,
    _parse_condition,
)


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
