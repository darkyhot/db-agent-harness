"""Тесты для plan-edit цикла."""

from __future__ import annotations

import json

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
        "schema_name": ["schema_a"] * 11,
        "table_name": [
            "fact_x", "fact_x", "fact_x", "fact_x", "fact_x",
            "dim_y", "dim_y", "dim_y",
            "alt_fact", "alt_fact", "alt_fact",
        ],
        "column_name": [
            "task_code", "report_dt", "task_subtype", "task_category", "inn",
            "task_code", "segment_name", "tb_id",
            "task_code", "report_dt", "task_category",
        ],
        "dType": [
            "varchar", "date", "varchar", "varchar", "varchar",
            "varchar", "varchar", "varchar",
            "varchar", "date", "varchar",
        ],
        "description": [
            "Код задачи", "Дата отчёта", "Тип задачи", "Категория задачи", "ИНН",
            "Код задачи", "Сегмент", "ТБ",
            "Код задачи", "Дата отчёта", "Категория задачи",
        ],
        "is_primary_key": [
            True, False, False, False, False,
            True, False, False,
            True, False, False,
        ],
        "unique_perc": [100.0, 30.0, 5.0, 5.0, 80.0, 100.0, 10.0, 20.0, 100.0, 30.0, 5.0],
        "not_null_perc": [100.0] * 11,
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
            "aggregations": [{"function": "COUNT", "column": "task_code", "alias": "count_task_code"}],
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
        "plan_edit_resolution": {},
        "needs_clarification": False,
        "clarification_message": "",
        "allowed_tables": ["schema_a.fact_x"],
        "excluded_tables": [],
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
    patch = result["plan_edit_resolution"]["chosen_patch"]
    assert {"command": "set_order", "direction": "DESC"} in patch


def test_router_patch_count_star_russian(node):
    result = node.plan_edit_router(_state(plan_edit_text="посчитай просто количество строк и сортируй по убыванию"))
    assert result["plan_edit_kind"] == "patch"
    patch = result["plan_edit_resolution"]["chosen_patch"]
    assert {"command": "set_count_star"} in patch
    assert {"command": "set_order", "direction": "DESC"} in patch


def test_router_patch_no_distinct_only(node):
    result = node.plan_edit_router(_state(plan_edit_text="не надо считать по уникальной дате"))
    assert result["plan_edit_kind"] == "patch"
    assert result["plan_edit_resolution"]["chosen_patch"] == [
        {"command": "set_distinct", "value": False, "target": "primary"},
    ]


def test_router_count_by_field(node):
    result = node.plan_edit_router(_state(plan_edit_text="посчитай по task_code"))
    assert result["plan_edit_kind"] == "patch"
    patch = result["plan_edit_resolution"]["chosen_patch"]
    assert {"command": "replace_primary_metric", "function": "COUNT", "column": "task_code", "distinct": False} in patch


def test_router_count_by_field_supports_count_po_inn(node):
    result = node.plan_edit_router(_state(plan_edit_text="Давай count по инн"))
    assert result["plan_edit_kind"] == "patch"
    patch = result["plan_edit_resolution"]["chosen_patch"]
    assert {"command": "replace_primary_metric", "function": "COUNT", "column": "inn", "distinct": False} in patch


def test_router_count_by_unknown_field_clarifies(node):
    result = node.plan_edit_router(_state(plan_edit_text="Давай count по mystery_field"))
    assert result["plan_edit_kind"] == "clarify"
    assert result["plan_edit_needs_clarification"] is True


def test_router_add_count_metric(node):
    result = node.plan_edit_router(_state(plan_edit_text="и еще по инн"))
    assert result["plan_edit_kind"] == "patch"
    assert {"command": "add_metric", "function": "COUNT", "column": "inn", "distinct": False} in result["plan_edit_resolution"]["chosen_patch"]


def test_router_and_patcher_add_second_count_metric_from_natural_feedback(node):
    state = _state(
        plan_edit_text="ты считаешь только task_code, надо еще инн",
        selected_columns={
            "schema_a.fact_x": {
                "select": ["task_code"],
                "filter": ["report_dt", "task_subtype", "task_category"],
                "aggregate": ["task_code", "inn"],
                "group_by": [],
            }
        },
    )
    routed = node.plan_edit_router(state)
    assert routed["plan_edit_kind"] == "patch"
    assert {"command": "add_metric", "function": "COUNT", "column": "inn", "distinct": False} in routed["plan_edit_resolution"]["chosen_patch"]

    patched = node.plan_patcher({**state, **routed})
    assert patched["sql_blueprint"]["aggregation"]["column"] == "task_code"
    assert patched["sql_blueprint"]["aggregations"] == [
        {"function": "COUNT", "column": "task_code", "alias": "count_task_code"},
        {"function": "COUNT", "column": "inn", "alias": "count_inn"},
    ]
    assert patched["user_hints"]["aggregation_preferences_list"] == [
        {"function": "count", "column": "task_code", "distinct": False},
        {"function": "count", "column": "inn", "distinct": False},
    ]


def test_router_patch_date_shift(node):
    result = node.plan_edit_router(_state(plan_edit_text="поменяй фильтр на дате с 1 числа на 2"))
    assert result["plan_edit_kind"] == "patch"
    patch = result["plan_edit_resolution"]["chosen_patch"]
    assert {"command": "set_date_range", "from": "2026-02-02", "to": "2026-03-02"} in patch


def test_router_rebind_replace_table(node):
    result = node.plan_edit_router(_state(plan_edit_text="я передумал использовать эту таблицу и хочу таблицу schema_a.alt_fact"))
    assert result["plan_edit_kind"] == "rebind"
    assert result["plan_edit_payload"]["operations"][0]["op"] == "replace_main_table"


def test_router_rewrite(node):
    result = node.plan_edit_router(_state(plan_edit_text="я вообще передумал что-то считать, покажи список"))
    assert result["plan_edit_kind"] == "rewrite"
    assert result["plan_edit_payload"]["intent_changes"]["aggregation_hint"] == "list"


def test_patcher_applies_count_distinct(node):
    state = _state(
        plan_edit_payload={
            "chosen_patch": [
                {"command": "replace_primary_metric", "function": "COUNT", "column": "task_code", "distinct": True},
            ],
        },
        plan_edit_resolution={
            "edit_goal": "patch",
            "requested_changes": [],
            "candidate_targets": [],
            "chosen_patch": [
                {"command": "replace_primary_metric", "function": "COUNT", "column": "task_code", "distinct": True},
            ],
            "confidence": 0.95,
            "clarification_reason": "",
        },
    )
    result = node.plan_patcher(state)
    assert result["sql_blueprint"]["aggregation"]["distinct"] is True
    assert result["sql_blueprint"]["aggregation"]["column"] == "task_code"
    assert result["sql_blueprint"]["aggregations"][0]["distinct"] is True


def test_patcher_allows_main_table_column_not_in_selected_columns(node):
    state = _state(
        sql_blueprint={
            "strategy": "simple_select",
            "main_table": "schema_a.fact_x",
            "where_conditions": [
                "report_dt >= '2026-02-01'::date",
                "report_dt < '2026-03-01'::date",
                "is_task = true",
            ],
            "aggregation": {"function": "COUNT", "column": "report_dt", "alias": "count_report_dt"},
            "group_by": [],
            "order_by": "count_report_dt DESC",
            "limit": None,
        },
        selected_columns={
            "schema_a.fact_x": {
                "select": ["report_dt"],
                "filter": ["report_dt"],
                "aggregate": ["report_dt"],
                "group_by": [],
            }
        },
        plan_edit_payload={
            "chosen_patch": [
                {"command": "replace_primary_metric", "function": "COUNT", "column": "inn", "distinct": False},
            ]
        },
        plan_edit_resolution={
            "edit_goal": "patch",
            "requested_changes": [],
            "candidate_targets": [],
            "chosen_patch": [
                {"command": "replace_primary_metric", "function": "COUNT", "column": "inn", "distinct": False},
            ],
            "confidence": 0.95,
            "clarification_reason": "",
        },
    )
    result = node.plan_patcher(state)
    assert result["sql_blueprint"]["aggregation"]["column"] == "inn"
    assert result["sql_blueprint"]["aggregation"]["alias"] == "count_inn"
    assert result["sql_blueprint"]["order_by"] == "count_inn DESC"
    assert result["user_hints"]["aggregation_preferences"]["column"] == "inn"
    assert result["user_hints"]["aggregation_preferences"]["function"] == "count"
    assert result["user_hints"]["aggregation_preferences"]["distinct"] is False
    assert result["user_hints"]["aggregation_preferences_list"][0]["column"] == "inn"


def test_patcher_adds_second_aggregation_and_persists_hints(node):
    state = _state(
        plan_edit_payload={
            "chosen_patch": [
                {"command": "add_metric", "function": "COUNT", "column": "inn", "distinct": True},
            ]
        },
        plan_edit_resolution={
            "edit_goal": "patch",
            "requested_changes": [],
            "candidate_targets": [],
            "chosen_patch": [
                {"command": "add_metric", "function": "COUNT", "column": "inn", "distinct": True},
            ],
            "confidence": 0.95,
            "clarification_reason": "",
        },
    )
    result = node.plan_patcher(state)
    assert len(result["sql_blueprint"]["aggregations"]) == 2
    assert result["sql_blueprint"]["aggregations"][1] == {
        "function": "COUNT",
        "column": "inn",
        "alias": "count_inn",
        "distinct": True,
    }
    assert result["sql_blueprint"]["aggregation"]["column"] == "task_code"
    assert result["user_hints"]["aggregation_preferences_list"] == [
        {"function": "count", "column": "task_code", "distinct": False},
        {"function": "count", "column": "inn", "distinct": True},
    ]


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
        plan_edit_payload={"chosen_patch": [{"command": "set_count_star"}]},
        plan_edit_resolution={
            "edit_goal": "patch",
            "requested_changes": [],
            "candidate_targets": [],
            "chosen_patch": [{"command": "set_count_star"}],
            "confidence": 0.95,
            "clarification_reason": "",
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


def test_query_spec_plan_edit_llm_remove_source_persists_exclusion(synthetic_loader):
    response = {
        "action": "remove_source",
        "tables": ["schema_a.dim_y"],
        "confidence": 0.93,
    }
    node = _Node(
        _DummyLLM([json.dumps(response, ensure_ascii=False)]),
        _DummyDB(),
        synthetic_loader,
        _DummyMemory(),
        _DummyValidator(),
        [],
    )
    state = _state(
        plan_edit_text="оставь только fact_x, dim_y не нужна",
        selected_tables=[("schema_a", "fact_x"), ("schema_a", "dim_y")],
        allowed_tables=["schema_a.fact_x", "schema_a.dim_y"],
        join_spec=[{"left": "schema_a.fact_x.task_code", "right": "schema_a.dim_y.task_code", "safe": True}],
        query_spec={
            "task": "answer_data",
            "metrics": [{"operation": "count", "target": "task_code", "distinct_policy": "all", "confidence": 0.9}],
            "dimensions": [],
            "filters": [],
            "source_constraints": [{"schema": "schema_a", "table": "fact_x", "required": True, "confidence": 0.9}],
            "join_constraints": [],
            "clarification_needed": False,
            "confidence": 0.9,
        },
    )

    routed = node.plan_edit_router(state)
    assert routed["plan_edit_kind"] == "rebind"
    assert routed["excluded_tables"] == ["schema_a.dim_y"]

    rebound = node.source_rebinder({**state, **routed})
    assert rebound["selected_tables"] == [("schema_a", "fact_x")]
    assert rebound["allowed_tables"] == ["schema_a.fact_x"]
    assert rebound["excluded_tables"] == ["schema_a.dim_y"]
    assert rebound["join_spec"] == []


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


def test_validator_accepts_catalog_column_not_in_selected_columns(node):
    state = _state(
        sql_blueprint={
            "main_table": "schema_a.fact_x",
            "aggregation": {"function": "COUNT", "column": "inn", "alias": "count_inn"},
            "where_conditions": [],
            "group_by": [],
            "order_by": "count_inn DESC",
            "limit": None,
        },
        selected_columns={
            "schema_a.fact_x": {
                "select": ["report_dt"],
                "filter": ["report_dt"],
                "aggregate": ["report_dt"],
                "group_by": [],
            }
        },
    )
    result = node.plan_edit_validator(state)
    assert result["needs_clarification"] is False


def test_validator_accepts_multiple_aggregations(node):
    state = _state(
        sql_blueprint={
            "main_table": "schema_a.fact_x",
            "aggregation": {"function": "COUNT", "column": "task_code", "alias": "count_task_code"},
            "aggregations": [
                {"function": "COUNT", "column": "task_code", "alias": "count_task_code"},
                {"function": "COUNT", "column": "inn", "alias": "count_inn", "distinct": True},
            ],
            "where_conditions": [],
            "group_by": [],
            "order_by": "count_inn DESC",
            "limit": None,
        },
        selected_columns={
            "schema_a.fact_x": {
                "select": ["task_code"],
                "filter": ["report_dt"],
                "aggregate": ["task_code"],
                "group_by": [],
            }
        },
    )
    result = node.plan_edit_validator(state)
    assert result["needs_clarification"] is False


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


def test_diff_renderer_reports_aggregations_change(node):
    state = _state(
        previous_sql_blueprint={
            "main_table": "schema_a.fact_x",
            "aggregation": {"function": "COUNT", "column": "task_code", "alias": "count_task_code"},
            "aggregations": [{"function": "COUNT", "column": "task_code", "alias": "count_task_code"}],
            "where_conditions": [],
            "group_by": [],
            "order_by": "count_task_code DESC",
            "limit": None,
        },
        sql_blueprint={
            "main_table": "schema_a.fact_x",
            "aggregation": {"function": "COUNT", "column": "task_code", "alias": "count_task_code"},
            "aggregations": [
                {"function": "COUNT", "column": "task_code", "alias": "count_task_code"},
                {"function": "COUNT", "column": "inn", "alias": "count_inn", "distinct": True},
            ],
            "where_conditions": [],
            "group_by": [],
            "order_by": "count_task_code DESC",
            "limit": None,
        },
    )
    result = node.plan_diff_renderer(state)
    changed_fields = {item["field"] for item in result["plan_diff"]["changed"]}
    assert "aggregations" in changed_fields


def test_query_spec_editor_adds_missing_tb_metric(synthetic_loader):
    response = {
        "task": "answer_data",
        "metrics": [
            {"operation": "count", "target": "old_gosb_id", "distinct_policy": "distinct", "confidence": 0.9},
            {"operation": "count", "target": "tb_id", "distinct_policy": "distinct", "confidence": 0.9},
        ],
        "dimensions": [],
        "filters": [],
        "source_constraints": [{"schema": "schema_a", "table": "dim_y", "required": True, "confidence": 0.9}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.9,
    }
    node = _Node(
        _DummyLLM([json.dumps(response, ensure_ascii=False)]),
        _DummyDB(),
        synthetic_loader,
        _DummyMemory(),
        _DummyValidator(),
        [],
    )
    state = _state(
        plan_edit_text="Ты посчитал только ГОСБ, а где ТБ?",
        query_spec={
            "task": "answer_data",
            "metrics": [
                {"operation": "count", "target": "old_gosb_id", "distinct_policy": "distinct", "confidence": 0.9}
            ],
            "dimensions": [],
            "filters": [],
            "source_constraints": [{"schema": "schema_a", "table": "dim_y", "required": True, "confidence": 0.9}],
            "join_constraints": [],
            "clarification_needed": False,
            "confidence": 0.9,
        },
    )

    result = node.plan_edit_router(state)

    assert result["plan_edit_kind"] == "query_spec"
    assert result["sql_blueprint"] == {}
    assert result["user_hints"]["aggregation_preferences_list"] == [
        {"function": "count", "column": "old_gosb_id", "distinct": True},
        {"function": "count", "column": "tb_id", "distinct": True},
    ]


def test_query_spec_editor_changes_sort_direction_without_losing_metric(synthetic_loader):
    response = {
        "task": "answer_data",
        "metrics": [{"operation": "count", "target": "inn", "distinct_policy": "all", "confidence": 0.9}],
        "dimensions": [],
        "filters": [],
        "order_by": {"target": "inn", "direction": "ASC", "confidence": 0.9},
        "source_constraints": [{"schema": "schema_a", "table": "fact_x", "required": True, "confidence": 0.9}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.9,
    }
    node = _Node(
        _DummyLLM([json.dumps(response, ensure_ascii=False)]),
        _DummyDB(),
        synthetic_loader,
        _DummyMemory(),
        _DummyValidator(),
        [],
    )
    state = _state(
        plan_edit_text="сделай сортировку asc",
        query_spec={
            "task": "answer_data",
            "metrics": [{"operation": "count", "target": "inn", "distinct_policy": "all", "confidence": 0.9}],
            "dimensions": [],
            "filters": [],
            "order_by": {"target": "inn", "direction": "DESC", "confidence": 0.9},
            "source_constraints": [{"schema": "schema_a", "table": "fact_x", "required": True, "confidence": 0.9}],
            "join_constraints": [],
            "clarification_needed": False,
            "confidence": 0.9,
        },
    )

    result = node.plan_edit_router(state)

    assert result["query_spec"]["order_by"] == {"target": "inn", "direction": "ASC", "confidence": 0.9}
    assert result["query_spec"]["metrics"] == [
        {"operation": "count", "target": "inn", "distinct_policy": "all", "confidence": 0.9}
    ]


def test_query_spec_editor_set_sources_only_rebinds_and_excludes_others(synthetic_loader):
    action = {
        "action": "set_sources_only",
        "tables": ["schema_a.dim_y"],
        "confidence": 0.95,
    }
    node = _Node(
        _DummyLLM([json.dumps(action, ensure_ascii=False)]),
        _DummyDB(),
        synthetic_loader,
        _DummyMemory(),
        _DummyValidator(),
        [],
    )
    state = _state(
        plan_edit_text="вообще не надо join, достаточно справочника dim_y",
        query_spec={
            "task": "answer_data",
            "metrics": [{"operation": "count", "target": "tb_id", "distinct_policy": "distinct", "confidence": 0.9}],
            "dimensions": [],
            "filters": [],
            "source_constraints": [],
            "join_constraints": [],
            "clarification_needed": False,
            "confidence": 0.9,
        },
        selected_tables=[("schema_a", "fact_x"), ("schema_a", "dim_y")],
        selected_columns={
            "schema_a.fact_x": {"select": ["task_code"], "aggregate": ["task_code"]},
            "schema_a.dim_y": {"select": ["tb_id"], "aggregate": ["tb_id"]},
        },
        join_spec=[{"left": "schema_a.fact_x.task_code", "right": "schema_a.dim_y.task_code"}],
    )

    routed = node.plan_edit_router(state)
    assert routed["plan_edit_kind"] == "rebind"

    rebased = node.source_rebinder({**state, **routed})

    assert rebased["selected_tables"] == [("schema_a", "dim_y")]
    assert rebased["allowed_tables"] == ["schema_a.dim_y"]
    assert "schema_a.fact_x" in rebased["excluded_tables"]
    assert all("schema_a.fact_x" not in str(item) for item in rebased.get("join_spec", []))


def test_plan_edit_validator_rejects_returned_excluded_source(node):
    result = node.plan_edit_validator(
        _state(
            excluded_tables=["schema_a.fact_x"],
            selected_tables=[("schema_a", "fact_x")],
        )
    )

    assert result["plan_edit_applied"] is False
    assert "Исключённый источник вернулся" in result["clarification_message"]


def test_table_explorer_does_not_fallback_when_explicit_sources_are_excluded(node):
    result = node.table_explorer(
        _state(
            selected_tables=[],
            allowed_tables=["schema_a.fact_x"],
            excluded_tables=["schema_a.fact_x"],
            user_input="покажи fact_x",
        )
    )

    assert result["table_structures"] == {}
    assert result["join_analysis_data"] == {}


def test_fallback_llm_emits_operations(synthetic_loader):
    llm = _DummyLLM([
        '{"edit_kind":"patch","confidence":0.4,"payload":{}}',
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
    assert result["plan_edit_resolution"]["chosen_patch"] == [{"command": "set_count_star"}]
    assert len(llm.calls) == 0
