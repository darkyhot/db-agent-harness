"""Тесты детерминированной коррекции table_resolver."""

from unittest.mock import MagicMock

import pandas as pd

from graph.graph import create_initial_state


class StubLLM:
    """Минимальный LLM-стаб для ответа table_resolver."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        if self._idx >= len(self._responses):
            return '{"tables": [], "plan_steps": []}'
        resp = self._responses[self._idx]
        self._idx += 1
        return resp

    def invoke(self, prompt: str, temperature=None) -> str:
        return self.invoke_with_system("", prompt, temperature=temperature)


def _make_nodes(schema_loader):
    from graph.nodes import GraphNodes

    memory = MagicMock()
    memory.add_message.return_value = None
    memory.get_memory_list.return_value = []
    memory.get_all_memory.return_value = {}
    memory.get_sessions_context.return_value = ""
    memory.get_session_messages.return_value = []

    db = MagicMock()
    validator = MagicMock()
    llm = StubLLM([
        (
            '{"tables": [{"schema": "dm", "table": "uzp_dwh_fact_outflow", '
            '"reason": "факт оттока"}], '
            '"plan_steps": ["Использовать dm.uzp_dwh_fact_outflow"]}'
        ),
    ])
    return GraphNodes(llm, db, schema_loader, memory, validator, [], debug_prompt=False)


def test_table_resolver_prefers_external_dim_when_fact_dimension_sparse(tmp_path):
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm", "dm"],
        "table_name": [
            "uzp_dwh_fact_outflow",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_epk_consolidation",
        ],
        "description": [
            "Фактовый отток клиентов",
            "Воронка продаж с сегментом и признаками оттока",
            "Консолидация клиентов и сегментов",
        ],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 11,
        "table_name": [
            "uzp_dwh_fact_outflow",
            "uzp_dwh_fact_outflow",
            "uzp_dwh_fact_outflow",
            "uzp_dwh_fact_outflow",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_epk_consolidation",
            "uzp_data_epk_consolidation",
            "uzp_data_epk_consolidation",
        ],
        "column_name": [
            "inn",
            "report_dt",
            "outflow_qty",
            "segment_name",
            "inn",
            "report_dt",
            "segment_name",
            "is_outflow",
            "inn",
            "segment_name",
            "segment_id",
        ],
        "dType": [
            "varchar",
            "date",
            "int4",
            "varchar",
            "varchar",
            "date",
            "varchar",
            "int4",
            "varchar",
            "varchar",
            "bigint",
        ],
        "description": [
            "ИНН клиента",
            "Отчётная дата",
            "Количество оттока клиентов",
            "Сегмент клиента",
            "ИНН клиента",
            "Отчётная дата",
            "Сегмент клиента",
            "Признак подтверждения оттока",
            "ИНН клиента",
            "Сегмент клиента",
            "Идентификатор сегмента",
        ],
        "is_primary_key": [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ],
        "unique_perc": [
            90.0,
            1.0,
            10.0,
            2.0,
            90.0,
            1.0,
            5.0,
            4.0,
            90.0,
            5.0,
            100.0,
        ],
        "not_null_perc": [
            95.0,
            99.0,
            95.0,
            1.61,
            95.0,
            99.0,
            99.99,
            100.0,
            95.0,
            99.98,
            100.0,
        ],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    nodes = _make_nodes(loader)

    state = create_initial_state("Посчитай сумму оттока по дате и сегменту")
    state["intent"] = {
        "intent": "analytics",
        "entities": ["отток", "дата", "сегмент"],
        "date_filters": {"from": None, "to": None},
        "aggregation_hint": "sum",
        "needs_search": False,
        "complexity": "single_table",
        "required_output": ["дата", "сегмент"],
        "filter_conditions": [],
    }

    result = nodes.table_resolver(state)
    selected = set(result["selected_tables"])
    allowed = set(result["allowed_tables"])

    assert ("dm", "uzp_dwh_fact_outflow") in selected
    assert ("dm", "uzp_data_epk_consolidation") in selected
    assert ("dm", "uzp_data_split_mzp_sale_funnel") not in selected
    assert "dm.uzp_data_epk_consolidation" in allowed


def test_table_resolver_uses_grain_to_prefer_task_table(tmp_path):
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["uzp_dwh_fact_outflow", "uzp_data_split_mzp_sale_funnel"],
        "description": [
            "Фактические события оттока",
            "Воронка продаж по задачам",
        ],
        "grain": ["event", "task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 6,
        "table_name": [
            "uzp_dwh_fact_outflow",
            "uzp_dwh_fact_outflow",
            "uzp_dwh_fact_outflow",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_split_mzp_sale_funnel",
        ],
        "column_name": [
            "saphr_id",
            "report_dt",
            "outflow_qty",
            "task_id",
            "report_dt",
            "is_outflow",
        ],
        "dType": ["varchar", "date", "int4", "bigint", "date", "int4"],
        "description": [
            "ID сотрудника",
            "Отчетная дата",
            "Количество оттока",
            "ID задачи",
            "Отчетная дата",
            "Признак подтверждения оттока",
        ],
        "is_primary_key": [False, False, False, False, False, False],
        "unique_perc": [80.0, 1.0, 5.0, 95.0, 1.0, 3.0],
        "not_null_perc": [95.0, 99.0, 99.0, 100.0, 99.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    nodes = _make_nodes(loader)

    state = create_initial_state("Посчитай количество задач с подтвержденным оттоком за февраль 2026")
    state["intent"] = {
        "intent": "analytics",
        "entities": ["задачи", "отток"],
        "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
        "aggregation_hint": "count",
        "needs_search": False,
        "complexity": "single_table",
        "required_output": [],
        "filter_conditions": [],
    }

    result = nodes.table_resolver(state)
    assert result["selected_tables"][0] == ("dm", "uzp_data_split_mzp_sale_funnel")
