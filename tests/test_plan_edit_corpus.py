"""Behavior corpus для state-based plan_edit resolver."""

from __future__ import annotations

import pandas as pd
import pytest

from core.schema_loader import SchemaLoader
from tests.test_plan_edit import _DummyDB, _DummyLLM, _DummyMemory, _DummyValidator, _Node, _state


@pytest.fixture
def synthetic_loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["schema_a", "schema_a"],
        "table_name": ["fact_x", "dim_y"],
        "description": [
            "Фактовая таблица событий",
            "Справочник клиентов",
        ],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)

    attrs_df = pd.DataFrame({
        "schema_name": ["schema_a"] * 8,
        "table_name": [
            "fact_x", "fact_x", "fact_x", "fact_x", "fact_x",
            "dim_y", "dim_y", "dim_y",
        ],
        "column_name": [
            "task_code", "report_dt", "task_subtype", "task_category", "inn",
            "task_code", "segment_name", "tb_id",
        ],
        "dType": [
            "varchar", "date", "varchar", "varchar", "varchar",
            "varchar", "varchar", "varchar",
        ],
        "description": [
            "Код задачи", "Дата отчёта", "Тип задачи", "Категория задачи", "ИНН",
            "Код задачи", "Сегмент", "ТБ",
        ],
        "is_primary_key": [True, False, False, False, False, True, False, False],
        "unique_perc": [100.0, 30.0, 5.0, 5.0, 80.0, 100.0, 10.0, 20.0],
        "not_null_perc": [100.0] * 8,
    })
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


@pytest.fixture
def node(synthetic_loader):
    return _Node(
        _DummyLLM(),
        _DummyDB(),
        synthetic_loader,
        _DummyMemory(),
        _DummyValidator(),
        [],
    )


@pytest.mark.parametrize(
    ("edit_text", "overrides", "expected_kind", "expected_patch"),
    [
        (
            "ты считаешь только task_code, надо еще инн",
            {
                "selected_columns": {
                    "schema_a.fact_x": {
                        "select": ["task_code"],
                        "filter": ["report_dt", "task_subtype", "task_category"],
                        "aggregate": ["task_code", "inn"],
                        "group_by": [],
                    }
                },
            },
            "patch",
            [{"command": "add_metric", "function": "COUNT", "column": "inn", "distinct": False}],
        ),
        (
            "не по task_code, а по инн",
            {
                "selected_columns": {
                    "schema_a.fact_x": {
                        "select": ["task_code"],
                        "filter": ["report_dt", "task_subtype", "task_category"],
                        "aggregate": ["task_code", "inn"],
                        "group_by": [],
                    }
                },
            },
            "patch",
            [{"command": "replace_primary_metric", "function": "COUNT", "column": "inn", "distinct": False}],
        ),
        (
            "вместо инн",
            {
                "selected_columns": {
                    "schema_a.fact_x": {
                        "select": ["task_code"],
                        "filter": ["report_dt", "task_subtype", "task_category"],
                        "aggregate": ["task_code", "inn"],
                        "group_by": [],
                    }
                },
            },
            "patch",
            [{"command": "replace_primary_metric", "function": "COUNT", "column": "inn", "distinct": False}],
        ),
        (
            "без distinct",
            {
                "sql_blueprint": {
                    "strategy": "simple_select",
                    "main_table": "schema_a.fact_x",
                    "where_conditions": [],
                    "aggregation": {"function": "COUNT", "column": "task_code", "alias": "count_task_code", "distinct": True},
                    "aggregations": [{"function": "COUNT", "column": "task_code", "alias": "count_task_code", "distinct": True}],
                    "group_by": [],
                    "order_by": "count_task_code DESC",
                    "limit": None,
                },
            },
            "patch",
            [{"command": "set_distinct", "value": False, "target": "primary"}],
        ),
        (
            "считай просто строки",
            {},
            "patch",
            [{"command": "set_count_star"}],
        ),
        (
            "ты считаешь только task_code, надо еще task",
            {},
            "clarify",
            [],
        ),
    ],
)
def test_plan_edit_behavior_corpus(node, edit_text, overrides, expected_kind, expected_patch):
    state = _state(plan_edit_text=edit_text, **overrides)
    routed = node.plan_edit_router(state)
    assert routed["plan_edit_kind"] == expected_kind

    if expected_kind == "patch":
        assert routed["plan_edit_resolution"]["edit_goal"] == "patch"
        assert routed["plan_edit_resolution"]["chosen_patch"] == expected_patch
        patched = node.plan_patcher({**state, **routed})
        assert patched["sql_blueprint"]["aggregation"] is not None
    else:
        assert routed["plan_edit_needs_clarification"] is True
        assert routed["clarification_message"]
        assert routed["plan_edit_resolution"]["clarification_reason"]
