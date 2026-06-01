"""Централизованный scrub авто-выведенных «левых» WHERE-условий (round 8).

Path-независимая защита: на финальном blueprint режем grain-флаги и точечные
даты/таймстемпы, которых пользователь не просил, сохраняя явные QuerySpec-фильтры,
диапазоны дат и категориальные ILIKE.
"""

import pandas as pd

from core.schema_loader import SchemaLoader
from graph.nodes.sql_pipeline import _flag_in_grain_set, _scrub_auto_derived_where


def _funnel_loader(tmp_path):
    """sale_funnel_task: зерно = задача, 3 task-флага, категориальные task_subtype/category."""
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel_task"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 6,
        "table_name": ["sale_funnel_task"] * 6,
        "column_name": [
            "report_dt", "task_subtype", "task_category",
            "is_task_closed", "is_task_in_progress", "fact_close_task_dttm",
        ],
        "dType": ["date", "varchar", "varchar", "boolean", "boolean", "timestamp"],
        "description": [
            "Отчётная дата", "Подтип задачи", "Категория задачи",
            "Признак закрытия задачи", "Признак - задача выполняется",
            "Дата фактического закрытия задачи",
        ],
        "is_primary_key": [False] * 6,
        "unique_perc": [1.0, 5.0, 2.0, 2.0, 2.0, 30.0],
        "not_null_perc": [100.0] * 6,
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def _outflow_loader(tmp_path):
    """fact_outflow: зерно = отток, ОДИН флаг is_task (различающий)."""
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["fact_outflow"],
        "description": ["Информация по фактическим оттокам"],
        "grain": ["event"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 3,
        "table_name": ["fact_outflow"] * 3,
        "column_name": ["report_dt", "is_task", "inn"],
        "dType": ["date", "boolean", "varchar"],
        "description": ["Отчётная дата", "Признак выставленной задачи", "ИНН"],
        "is_primary_key": [False] * 3,
        "unique_perc": [1.0, 50.0, 90.0],
        "not_null_perc": [100.0] * 3,
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_flag_in_grain_set(tmp_path):
    loader = _funnel_loader(tmp_path)
    assert _flag_in_grain_set("is_task_in_progress", "dm.sale_funnel_task", loader) is True
    assert _flag_in_grain_set("is_task_closed", "dm.sale_funnel_task", loader) is True
    od = tmp_path / "o"
    od.mkdir()
    out = _outflow_loader(od)
    assert _flag_in_grain_set("is_task", "dm.fact_outflow", out) is False  # одиночный флаг


def test_scrub_drops_grain_flags_and_point_dates_keeps_rest(tmp_path):
    """agent(25): is_task_in_progress / is_task_closed / fact_close_task_dttm='...' —
    убраны; report_dt-диапазон и обе ILIKE — сохранены."""
    loader = _funnel_loader(tmp_path)
    blueprint = {
        "main_table": "dm.sale_funnel_task",
        "selected_columns": {"dm.sale_funnel_task": {"filter": [
            "report_dt", "task_subtype", "task_category",
            "is_task_closed", "is_task_in_progress", "fact_close_task_dttm",
        ]}},
        "where_conditions": [
            "report_dt >= '2026-02-01'::date",
            "report_dt < '2026-03-01'::date",
            "is_task_in_progress = TRUE",
            "is_task_closed = TRUE",
            "fact_close_task_dttm = '2026-02-24 00:00:00.000000'",
            "task_subtype ILIKE '%Фактический отток%'",
            "task_category ILIKE '%Задача%'",
        ],
    }
    # QuerySpec: is_task (нет точной колонки) + report_dt. is_task НЕ совпадает с
    # is_task_in_progress/is_task_closed точно → не защищён.
    query_spec = {"filters": [{"target": "is_task"}, {"target": "report_dt"}]}
    kept = _scrub_auto_derived_where(blueprint, query_spec, loader)
    assert "is_task_in_progress = TRUE" not in kept
    assert "is_task_closed = TRUE" not in kept
    assert not any("fact_close_task_dttm" in c for c in kept)
    assert "report_dt >= '2026-02-01'::date" in kept
    assert "report_dt < '2026-03-01'::date" in kept
    assert any("task_subtype ILIKE" in c for c in kept)
    assert any("task_category ILIKE" in c for c in kept)


def test_scrub_keeps_lone_is_task_on_fact_outflow(tmp_path):
    """fact_outflow: is_task — одиночный различающий флаг и точный QuerySpec-target →
    СОХРАНЁН."""
    loader = _outflow_loader(tmp_path)
    blueprint = {
        "main_table": "dm.fact_outflow",
        "selected_columns": {"dm.fact_outflow": {"filter": ["report_dt", "is_task"]}},
        "where_conditions": [
            "report_dt >= '2026-02-01'::date",
            "report_dt < '2026-03-01'::date",
            "is_task = TRUE",
        ],
    }
    query_spec = {"filters": [{"target": "is_task"}, {"target": "report_dt"}]}
    kept = _scrub_auto_derived_where(blueprint, query_spec, loader)
    assert "is_task = TRUE" in kept
    assert len([c for c in kept if "report_dt" in c]) == 2


def test_scrub_keeps_explicit_point_date(tmp_path):
    """Явный точечный report_dt (точный QuerySpec-target) НЕ режется."""
    loader = _funnel_loader(tmp_path)
    blueprint = {
        "main_table": "dm.sale_funnel_task",
        "selected_columns": {"dm.sale_funnel_task": {"filter": ["report_dt"]}},
        "where_conditions": ["report_dt = '2026-02-15'::date"],
    }
    query_spec = {"filters": [{"target": "report_dt"}]}
    kept = _scrub_auto_derived_where(blueprint, query_spec, loader)
    assert kept == ["report_dt = '2026-02-15'::date"]
