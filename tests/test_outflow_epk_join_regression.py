from pathlib import Path
from unittest.mock import MagicMock

from core.column_selector_deterministic import select_columns
from core.join_analysis import detect_table_type
from core.schema_loader import SchemaLoader
from core.semantic_frame import derive_semantic_frame
from core.sql_planner_deterministic import build_blueprint
from core.user_hint_extractor import extract_user_hints
from core.where_resolver import resolve_where
from graph.graph import create_initial_state
from graph.nodes import GraphNodes


QUERY = "Посчитай сумму оттока по дате и сегменту (сегмент возьми в uzp_data_epk_consolidation по инн)"


def _loader() -> SchemaLoader:
    return SchemaLoader(data_dir=Path("data_for_agent"))


def _intent() -> dict:
    return {
        "intent": "analytics",
        "entities": ["отток", "дата", "сегмент", "uzp_data_epk_consolidation", "инн"],
        "date_filters": {"from": None, "to": None},
        "aggregation_hint": "sum",
        "needs_search": False,
        "complexity": "join",
        "required_output": ["дата", "сегмент"],
        "filter_conditions": [],
        "explicit_join": [{"table_hint": "epk_consolidation", "column_hint": "inn"}],
    }


def test_sum_outflow_metric_phrase_is_not_filter_even_with_value_candidate():
    loader = _loader()
    loader._rule_registry = {
        "rules": [
            {
                "rule_id": "text:schema.uzp_data_split_mzp_sale_funnel.task_subtype",
                "column_key": "schema.uzp_data_split_mzp_sale_funnel.task_subtype",
                "semantic_class": "enum_like",
                "match_kind": "text_search",
                "match_phrases": ["подтип задачи"],
                "value_candidates": ["фактический отток", "отток"],
            }
        ]
    }

    hints = extract_user_hints(QUERY, loader)
    frame = derive_semantic_frame(QUERY, _intent(), schema_loader=loader, user_hints=hints)

    assert frame["metric_intent"] == "sum"
    assert "date" in frame["output_dimensions"]
    assert any("сегмент" in dim or "сегмент" == dim for dim in frame["output_dimensions"])
    assert hints["dim_sources"]["segment"]["table"] == "schema.uzp_data_epk_consolidation"
    assert hints["join_fields"] == ["inn"]
    assert not any("outflow" in str(item).lower() or "отток" in str(item).lower() for item in frame["filter_intents"])


def test_confirmed_outflow_task_query_still_keeps_real_filter():
    loader = _loader()
    frame = derive_semantic_frame(
        "Сколько задач с подтвержденным оттоком",
        {
            "aggregation_hint": "count",
            "entities": ["задачи", "отток"],
            "required_output": [],
            "filter_conditions": [],
        },
        schema_loader=loader,
    )

    assert any(
        item.get("column_key") == "schema.uzp_data_split_mzp_sale_funnel.is_outflow"
        for item in frame["filter_intents"]
    )


class _TableResolverLLM:
    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        return (
            '{"tables": [{"schema": "schema", "table": "uzp_data_epk_consolidation", '
            '"reason": "сегмент явно указан пользователем"}], '
            '"plan_steps": ["Использовать schema.uzp_data_epk_consolidation"]}'
        )

    def invoke(self, prompt: str, temperature=None) -> str:
        return self.invoke_with_system("", prompt, temperature=temperature)


def _nodes(loader: SchemaLoader) -> GraphNodes:
    memory = MagicMock()
    memory.add_message.return_value = None
    memory.get_memory_list.return_value = []
    memory.get_all_memory.return_value = {}
    memory.get_sessions_context.return_value = ""
    memory.get_session_messages.return_value = []
    return GraphNodes(_TableResolverLLM(), MagicMock(), loader, memory, MagicMock(), [], debug_prompt=False)


def test_table_resolver_keeps_outflow_fact_when_epk_is_dim_source():
    loader = _loader()
    hints = extract_user_hints(QUERY, loader)
    frame = derive_semantic_frame(QUERY, _intent(), schema_loader=loader, user_hints=hints)
    frame["subject"] = "organization"
    frame["requested_grain"] = "organization"
    state = create_initial_state(QUERY)
    state["intent"] = _intent()
    state["user_hints"] = hints
    state["semantic_frame"] = frame

    result = _nodes(loader).table_resolver(state)

    assert ("schema", "uzp_dwh_fact_outflow") in result["selected_tables"]
    assert ("schema", "uzp_data_epk_consolidation") in result["selected_tables"]


def test_column_selector_and_planner_build_fact_dim_outflow_epk_join():
    loader = _loader()
    hints = extract_user_hints(QUERY, loader)
    intent = _intent()
    frame = derive_semantic_frame(QUERY, intent, schema_loader=loader, user_hints=hints)
    tables = ["schema.uzp_dwh_fact_outflow", "schema.uzp_data_epk_consolidation"]
    table_structures = {table: loader.get_table_info(*table.split(".", 1)) for table in tables}
    table_types = {
        table: detect_table_type(table.split(".", 1)[1], loader.get_table_columns(*table.split(".", 1)))
        for table in tables
    }

    selected = select_columns(
        intent,
        table_structures,
        table_types,
        {},
        loader,
        user_input=QUERY,
        user_hints=hints,
        semantic_frame=frame,
    )
    blueprint = build_blueprint(
        intent,
        selected["selected_columns"],
        selected["join_spec"],
        table_types,
        {},
        user_input=QUERY,
        user_hints=hints,
        schema_loader=loader,
        semantic_frame=frame,
    )

    assert selected["selected_columns"]["schema.uzp_dwh_fact_outflow"]["aggregate"] == ["outflow_qty"]
    assert selected["selected_columns"]["schema.uzp_dwh_fact_outflow"]["group_by"] == ["report_dt"]
    assert selected["selected_columns"]["schema.uzp_data_epk_consolidation"]["group_by"] == ["segment_name"]
    assert selected["join_spec"][0]["left"] == "schema.uzp_dwh_fact_outflow.inn"
    assert selected["join_spec"][0]["right"] == "schema.uzp_data_epk_consolidation.inn"
    assert blueprint["strategy"] == "fact_dim_join"
    assert blueprint["aggregation"]["function"] == "SUM"
    assert blueprint["aggregation"]["column"] == "outflow_qty"
    assert blueprint["group_by"] == ["report_dt", "segment_name"]
    assert blueprint["where_resolution"]["needs_clarification"] is False


def test_where_resolver_suppresses_stale_outflow_filter_when_aggregate_covers_it():
    loader = _loader()
    frame = {
        "metric_intent": "sum",
        "business_event": "outflow",
        "filter_intents": [
            {
                "request_id": "text:schema.uzp_data_split_mzp_sale_funnel.task_type",
                "kind": "text_search",
                "query_text": "фактический отток",
                "column_key": "schema.uzp_data_split_mzp_sale_funnel.task_type",
                "match_source": "value_candidate",
            }
        ],
    }

    result = resolve_where(
        user_input=QUERY,
        intent=_intent(),
        selected_columns={
            "schema.uzp_dwh_fact_outflow": {
                "select": ["report_dt", "outflow_qty"],
                "aggregate": ["outflow_qty"],
                "group_by": ["report_dt"],
            },
            "schema.uzp_data_epk_consolidation": {
                "select": ["segment_name"],
                "group_by": ["segment_name"],
            },
        },
        selected_tables=["schema.uzp_dwh_fact_outflow", "schema.uzp_data_epk_consolidation"],
        schema_loader=loader,
        semantic_frame=frame,
    )

    assert result["needs_clarification"] is False
    assert "aggregate_metric_covers_business_event" in result["reasoning"]
