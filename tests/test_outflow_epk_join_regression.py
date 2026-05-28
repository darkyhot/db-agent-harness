from pathlib import Path
from unittest.mock import MagicMock

from core.column_selector_deterministic import select_columns
from core.catalog_grounding import ground_query_spec
from core.join_analysis import detect_table_type
from core.query_ir import QuerySpec
from core.schema_loader import SchemaLoader
from core.semantic_frame import derive_semantic_frame
from core.sql_builder import SqlBuilder
from core.sql_planner_deterministic import build_blueprint
from core.user_hint_extractor import extract_user_hints
from graph.graph import create_initial_state
from graph.nodes import GraphNodes


SCHEMA = "s_grnplm_ld_salesntwrk_pcap_sn_uzp"
FACT_OUTFLOW = f"{SCHEMA}.uzp_dwh_fact_outflow"
EPK = f"{SCHEMA}.uzp_data_epk_consolidation"

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


class _TableResolverLLM:
    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        return (
            f'{{"tables": [{{"schema": "{SCHEMA}", "table": "uzp_data_epk_consolidation", '
            '"reason": "сегмент явно указан пользователем"}], '
            f'"plan_steps": ["Использовать {EPK}"]}}'
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

    assert (SCHEMA, "uzp_dwh_fact_outflow") in result["selected_tables"]
    assert (SCHEMA, "uzp_data_epk_consolidation") in result["selected_tables"]


def test_column_selector_and_planner_build_fact_dim_outflow_epk_join():
    loader = _loader()
    hints = extract_user_hints(QUERY, loader)
    intent = _intent()
    frame = derive_semantic_frame(QUERY, intent, schema_loader=loader, user_hints=hints)
    tables = [FACT_OUTFLOW, EPK]
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

    assert selected["selected_columns"][FACT_OUTFLOW]["aggregate"] == ["outflow_qty"]
    assert selected["selected_columns"][FACT_OUTFLOW]["group_by"] == ["report_dt"]
    assert selected["selected_columns"][EPK]["group_by"] == ["segment_name"]
    assert selected["join_spec"][0]["left"] == f"{FACT_OUTFLOW}.inn"
    assert selected["join_spec"][0]["right"] == f"{EPK}.inn"
    assert blueprint["strategy"] == "fact_dim_join"
    assert blueprint["aggregation"]["function"] == "SUM"
    assert blueprint["aggregation"]["column"] == "outflow_qty"
    assert blueprint["group_by"] == ["report_dt", "segment_name"]
    assert blueprint["where_resolution"]["needs_clarification"] is False


def test_implicit_outflow_segment_uses_more_complete_joinable_dimension_source():
    loader = _loader()
    user_input = "Посчитай сумму оттока по дате и сегменту"
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "sum", "target": "outflow", "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [
            {"target": "date", "confidence": 0.8},
            {"target": "segment", "confidence": 0.8},
        ],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.8,
    })
    assert spec is not None, errors

    grounding = ground_query_spec(query_spec=spec, schema_loader=loader, user_input=user_input)
    tables = [source.full_name for source in grounding.sources]
    assert FACT_OUTFLOW in tables
    assert EPK in tables

    intent = spec.to_legacy_intent()
    hints = spec.to_legacy_user_hints()
    frame = derive_semantic_frame(user_input, intent, schema_loader=loader, user_hints=hints)
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
        user_input=user_input,
        user_hints=hints,
        semantic_frame=frame,
    )
    blueprint = build_blueprint(
        intent,
        selected["selected_columns"],
        selected["join_spec"],
        table_types,
        {},
        user_input=user_input,
        user_hints=hints,
        schema_loader=loader,
        semantic_frame=frame,
    )
    sql = SqlBuilder().build(
        blueprint["strategy"],
        selected["selected_columns"],
        selected["join_spec"],
        blueprint,
        table_types,
    )

    assert selected["selected_columns"][FACT_OUTFLOW]["aggregate"] == ["outflow_qty"]
    assert selected["selected_columns"][FACT_OUTFLOW]["group_by"] == ["report_dt"]
    assert selected["selected_columns"][EPK]["group_by"] == ["segment_name"]
    assert selected["join_spec"][0]["left"] == f"{FACT_OUTFLOW}.inn"
    assert selected["join_spec"][0]["right"] == f"{EPK}.inn"
    assert set(blueprint["group_by"]) == {"report_dt", "segment_name"}
    assert sql is not None
    normalized = " ".join(sql.split()).upper()
    assert "SUM(" in normalized and "OUTFLOW_QTY" in normalized
    assert "GROUP BY" in normalized and "REPORT_DT" in normalized and "SEGMENT_NAME" in normalized
    assert "DISTINCT ON (INN)" in normalized
