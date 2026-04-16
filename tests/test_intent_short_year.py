from unittest.mock import MagicMock

from graph.graph import create_initial_state


class StubLLM:
    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        return (
            '{"intent": "clarification", '
            '"entities": ["задачи", "отток", "февраль"], '
            '"date_filters": {"from": null, "to": null}, '
            '"aggregation_hint": "count", '
            '"needs_search": false, '
            '"complexity": "single_table", '
            '"clarification_question": "За какой год считать данные за февраль?", '
            '"filter_conditions": [], '
            '"explicit_join": [], '
            '"required_output": [], '
            '"month_without_year": true}'
        )

    def invoke(self, prompt: str, temperature=None) -> str:
        return self.invoke_with_system("", prompt, temperature=temperature)


def test_intent_classifier_upgrades_short_year_to_analytics(tmp_path):
    import pandas as pd
    from core.schema_loader import SchemaLoader
    from graph.nodes import GraphNodes

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel"],
        "description": ["Воронка продаж по задачам"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel"],
        "column_name": ["report_dt"],
        "dType": ["date"],
        "description": ["Отчетная дата"],
        "is_primary_key": [False],
        "unique_perc": [1.0],
        "not_null_perc": [100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    memory = MagicMock()
    memory.add_message.return_value = None
    memory.get_memory_list.return_value = []
    memory.get_all_memory.return_value = {}
    memory.get_sessions_context.return_value = ""
    memory.get_session_messages.return_value = []
    db = MagicMock()
    validator = MagicMock()
    nodes = GraphNodes(StubLLM(), db, loader, memory, validator, [], debug_prompt=False)

    state = create_initial_state(
        "Посчитай количество задач по фактическому оттоку за февраль\nУточнение пользователя: 26"
    )
    result = nodes.intent_classifier(state)

    assert result["needs_clarification"] is False
    assert result["intent"]["intent"] == "analytics"
    assert result["intent"]["date_filters"] == {"from": "2026-02-01", "to": "2026-03-01"}
