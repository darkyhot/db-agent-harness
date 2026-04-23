"""Тесты для plan-edit цикла."""

from __future__ import annotations

import pandas as pd
import pytest

from core.schema_loader import SchemaLoader
from graph.nodes.common import BaseNodeMixin
from graph.nodes.explorer import ExplorerNodes
from graph.nodes.plan_edit import PlanEditNodes
from graph.nodes.sql_pipeline import SqlPipelineNodes


class _DummyDB:
    def get_sample(self, schema: str, table: str, limit: int):
        return pd.DataFrame()


class _DummyLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls: list[tuple[str, str, float | None]] = []

    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
        self.calls.append((system_prompt, user_prompt, temperature))
        if self.responses:
            return self.responses.pop(0)
        raise AssertionError("LLM fallback should not be used in deterministic tests")


class _DummyMemory:
    def add_message(self, role: str, content: str) -> None:
        return None

    def get_all_memory(self):
        return {}


class _DummyValidator:
    pass


class _Node(PlanEditNodes, ExplorerNodes, SqlPipelineNodes, BaseNodeMixin):
    pass


@pytest.fixture
def synthetic_loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["schema_a", "schema_a", "schema_a"],
        "table_name": ["fact_x", "dim_y", "alt_fact"],
        "description": [
            "Фактовая таблица событий",
            "Справочник клиентов с сегментом",
            "Альтернативная фактовая таблица",
        ],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)

    attrs_df = pd.DataFrame({
        "schema_name": ["schema_a"] * 10,
        "table_name": [
            "fact_x", "fact_x", "fact_x", "fact_x",
            "dim_y", "dim_y", "dim_y",
            "alt_fact", "alt_fact", "alt_fact",
        ],
        "column_name": [
            "task_code", "report_dt", "task_subtype", "task_category",
            "task_code", "segment_name", "tb_id",
            "task_code", "report_dt", "task_category",
        ],
        "dType": [
            "varchar", "date", "varchar", "varchar",
            "varchar", "varchar", "varchar",
            "varchar", "date", "varchar",
        ],
        "description": [
            "Код задачи", "Дата отчёта", "Тип задачи", "Категория задачи",
            "Код задачи", "Сегмент", "ТБ",
            "Код задачи", "Дата отчёта", "Категория задачи",
        ],
        "is_primary_key": [
            True, False, False, False,
            True, False, False,
            True, False, False,
        ],
        "unique_perc": [100.0, 30.0, 5.0, 5.0, 100.0, 10.0, 20.0, 100.0, 30.0, 5.0],
        "not_null_perc": [100.0] * 10,
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


def _state(**overrides):
    base = {
        "graph_iterations": 0,
        "plan_edit_text": "",
        "sql_blueprint": {
            "strategy": "simple_select",
            "main_table": "schema_a.fact_x",
            "where_conditions": [
                "report_dt >= '2026-02-01'::date",
                "report_dt < '2026-03-01'::date",
                "task_subtype ILIKE '%фактический отток%'",
            ],
            "aggregation": {"function": "COUNT", "column": "task_code", "alias": "count_task_code"},
            "group_by": [],
            "order_by": "count_task_code DESC",
            "limit": None,
        },
        "selected_tables": [("schema_a", "fact_x")],
        "selected_columns": {
            "schema_a.fact_x": {
                "select": ["task_code"],
                "filter": ["report_dt", "task_subtype", "task_category"],
                "aggregate": ["task_code"],
                "group_by": [],
            }
        },
        "join_spec": [],
        "user_hints": {
            "must_keep_tables": [("schema_a", "fact_x")],
            "join_fields": [],
            "dim_sources": {},
            "having_hints": [],
            "group_by_hints": [],
            "aggregate_hints": [],
            "aggregation_preferences": {},
            "time_granularity": None,
            "negative_filters": [],
        },
        "intent": {"aggregation_hint": "count", "entities": ["задачи"]},
        "semantic_frame": {},
        "where_resolution": {},
        "join_decision": {},
        "planning_confidence": {},
        "evidence_trace": {},
        "fallback_policy": {},
        "plan_edit_history": [],
        "previous_sql_blueprint": {},
        "plan_edit_payload": {},
        "needs_clarification": False,
        "clarification_message": "",
        "allowed_tables": ["schema_a.fact_x"],
        "table_structures": {
            "schema_a.fact_x": "task_code | report_dt | task_subtype | task_category",
        },
        "table_samples": {"schema_a.fact_x": ""},
        "table_types": {"schema_a.fact_x": "fact"},
        "join_analysis_data": {},
        "column_selector_hint": "",
        "messages": [],
        "user_input": "Сколько задач по фактическому оттоку поставили в феврале 2026",
    }
    base.update(overrides)
    return base


def test_router_patch_sort_desc(node):
    result = node.plan_edit_router(_state(plan_edit_text="поменяй порядок сортировки по убыванию"))
    assert result["plan_edit_kind"] == "patch"
    ops = result["plan_edit_payload"]["operations"]
    assert {"op": "replace", "path": "order_by.direction", "value": "DESC"} in ops


def test_router_patch_count_star_russian(node):
    result = node.plan_edit_router(_state(plan_edit_text="посчитай просто количество строк и сортируй по убыванию"))
    assert result["plan_edit_kind"] == "patch"
    ops = result["plan_edit_payload"]["operations"]
    assert {"op": "replace", "path": "aggregation.column", "value": "*"} in ops
    assert {"op": "replace", "path": "aggregation.distinct", "value": False} in ops
    assert {"op": "replace", "path": "order_by.direction", "value": "DESC"} in ops


def test_router_patch_no_distinct_only(node):
    result = node.plan_edit_router(_state(plan_edit_text="не надо считать по уникальной дате"))
    assert result["plan_edit_kind"] == "patch"
    assert result["plan_edit_payload"]["operations"] == [
        {"op": "replace", "path": "aggregation.distinct", "value": False},
    ]


def test_router_count_by_field(node):
    result = node.plan_edit_router(_state(plan_edit_text="посчитай по task_code"))
    assert result["plan_edit_kind"] == "patch"
    ops = result["plan_edit_payload"]["operations"]
    assert {"op": "replace", "path": "aggregation.function", "value": "COUNT"} in ops
    assert {"op": "replace", "path": "aggregation.column", "value": "task_code"} in ops


def test_router_patch_date_shift(node):
    result = node.plan_edit_router(_state(plan_edit_text="поменяй фильтр на дате с 1 числа на 2"))
    assert result["plan_edit_kind"] == "patch"
    ops = result["plan_edit_payload"]["operations"]
    assert any(op["path"] == "where.date.from" and op["value"] == "2026-02-02" for op in ops)


def test_router_rebind_replace_table(node):
    result = node.plan_edit_router(_state(plan_edit_text="я передумал использовать эту таблицу и хочу таблицу schema_a.alt_fact"))
    assert result["plan_edit_kind"] == "rebind"
    assert result["plan_edit_payload"]["operations"][0]["op"] == "replace_main_table"


def test_router_rewrite(node):
    result = node.plan_edit_router(_state(plan_edit_text="я вообще передумал что-то считать, покажи список"))
    assert result["plan_edit_kind"] == "rewrite"
    assert result["plan_edit_payload"]["intent_changes"]["aggregation_hint"] == "list"


def test_patcher_applies_count_distinct(node):
    state = _state(plan_edit_payload={
        "operations": [
            {"op": "replace", "path": "aggregation.function", "value": "COUNT"},
            {"op": "replace", "path": "aggregation.column", "value": "task_code"},
            {"op": "replace", "path": "aggregation.distinct", "value": True},
        ]
    })
    result = node.plan_patcher(state)
    assert result["sql_blueprint"]["aggregation"]["distinct"] is True
    assert result["sql_blueprint"]["aggregation"]["column"] == "task_code"


def test_patcher_count_star_strips_distinct_and_updates_order_by(node):
    state = _state(
        sql_blueprint={
            "strategy": "simple_select",
            "main_table": "schema_a.fact_x",
            "where_conditions": [],
            "aggregation": {
                "function": "COUNT",
                "column": "task_code",
                "alias": "count_task_code",
                "distinct": True,
            },
            "group_by": [],
            "order_by": "count_task_code DESC",
            "limit": None,
        },
        plan_edit_payload={
            "operations": [
                {"op": "replace", "path": "aggregation.function", "value": "COUNT"},
                {"op": "replace", "path": "aggregation.column", "value": "*"},
                {"op": "replace", "path": "aggregation.distinct", "value": False},
            ]
        },
    )
    result = node.plan_patcher(state)
    assert result["sql_blueprint"]["aggregation"] == {
        "function": "COUNT",
        "column": "*",
        "alias": "count_all",
    }
    assert result["sql_blueprint"]["order_by"] == "count_all DESC"


def test_source_rebinder_replace_main_table(node):
    state = _state(plan_edit_payload={
        "operations": [{"op": "replace_main_table", "table": "schema_a.alt_fact"}],
    })
    result = node.source_rebinder(state)
    assert result["selected_tables"][0] == ("schema_a", "alt_fact")
    assert result["sql_blueprint"]["main_table"] == "schema_a.alt_fact"


def test_intent_rewriter_fast_path_list(node):
    state = _state(plan_edit_text="не считаем, покажи список", plan_edit_payload={
        "intent_changes": {"aggregation_hint": "list"},
    })
    result = node.intent_rewriter(state)
    assert result["intent"]["aggregation_hint"] == "list"
    assert result["sql_blueprint"].get("aggregation") is None


def test_validator_rejects_invalid_distinct_star(node):
    state = _state(sql_blueprint={
        "main_table": "schema_a.fact_x",
        "aggregation": {"function": "COUNT", "column": "*", "alias": "count_all", "distinct": True},
        "where_conditions": [],
        "group_by": [],
        "order_by": "",
        "limit": None,
    })
    result = node.plan_edit_validator(state)
    assert result["needs_clarification"] is True


def test_diff_renderer_reports_change(node):
    state = _state(
        previous_sql_blueprint={
            "main_table": "schema_a.fact_x",
            "aggregation": {"function": "COUNT", "column": "task_code", "alias": "count_task_code"},
            "where_conditions": [],
            "group_by": [],
            "order_by": "count_task_code DESC",
            "limit": None,
        },
        sql_blueprint={
            "main_table": "schema_a.fact_x",
            "aggregation": {"function": "COUNT", "column": "task_code", "alias": "count_task_code", "distinct": True},
            "where_conditions": [],
            "group_by": [],
            "order_by": "count_task_code ASC",
            "limit": None,
        },
    )
    result = node.plan_diff_renderer(state)
    changed_fields = {item["field"] for item in result["plan_diff"]["changed"]}
    assert "aggregation" in changed_fields
    assert "order_by" in changed_fields


def test_fallback_llm_emits_operations(synthetic_loader):
    llm = _DummyLLM([
        '{"edit_kind":"patch","confidence":0.4,"payload":{}}',
        '{"operations":[{"op":"replace","path":"aggregation.column","value":"*"},{"op":"replace","path":"aggregation.distinct","value":false}]}',
    ])
    node = _Node(
        llm,
        _DummyDB(),
        synthetic_loader,
        _DummyMemory(),
        _DummyValidator(),
        [],
    )
    result = node.plan_edit_router(_state(plan_edit_text="считай просто строки"))
    assert result["plan_edit_kind"] == "patch"
    assert result["plan_edit_payload"]["operations"] == [
        {"op": "replace", "path": "aggregation.column", "value": "*"},
        {"op": "replace", "path": "aggregation.distinct", "value": False},
    ]
    assert len(llm.calls) == 2
