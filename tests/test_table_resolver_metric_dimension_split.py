import pandas as pd

from graph.nodes.intent import IntentNodes


class _DummyMemory:
    def add_message(self, role: str, content: str) -> None:
        return


class _DummyLLM:
    def invoke_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # LLM явно упоминает только dimension-таблицу.
        return (
            '{"tables": ['
            '{"schema": "schema", "table": "dim_segments", "reason": "явно упомянута пользователем"}'
            '], "plan_steps": ["Использовать schema.dim_segments"]}'
        )


class _DummySchema:
    def __init__(self) -> None:
        self.tables_df = pd.DataFrame(
            [
                {"schema_name": "schema", "table_name": "fact_sales"},
                {"schema_name": "schema", "table_name": "dim_segments"},
            ]
        )
        self._columns = {
            ("schema", "fact_sales"): pd.DataFrame(
                [
                    {"column_name": "customer_id", "description": "id клиента", "not_null_perc": 100, "unique_perc": 10, "is_primary_key": False},
                    {"column_name": "outflow_qty", "description": "сумма оттока", "not_null_perc": 100, "unique_perc": 40, "is_primary_key": False},
                ]
            ),
            ("schema", "dim_segments"): pd.DataFrame(
                [
                    {"column_name": "customer_id", "description": "id клиента", "not_null_perc": 100, "unique_perc": 100, "is_primary_key": True},
                    {"column_name": "segment_name", "description": "сегмент клиента", "not_null_perc": 100, "unique_perc": 20, "is_primary_key": False},
                ]
            ),
        }

    def get_table_columns(self, schema_name: str, table_name: str) -> pd.DataFrame:
        return self._columns.get((schema_name, table_name), pd.DataFrame())

    def search_tables(self, query: str, top_n: int = 50) -> pd.DataFrame:
        return self.tables_df.head(top_n)


class _TestNode(IntentNodes):
    def __init__(self) -> None:
        self.llm = _DummyLLM()
        self.schema = _DummySchema()
        self.memory = _DummyMemory()
        self.debug_prompt = False

    def _get_schema_context(self, user_input: str) -> str:
        return "schema.fact_sales, schema.dim_segments"

    def _trim_to_budget(self, system_prompt: str, user_prompt: str):
        return system_prompt, user_prompt

    def _clean_llm_json(self, text: str) -> str:
        return text


def test_table_resolver_keeps_metric_table_and_adds_dimension_table() -> None:
    node = _TestNode()
    state = {
        "user_input": (
            "Посчитай сумму outflow_qty по segment_name, "
            "segment_name возьми из schema.dim_segments по customer_id"
        ),
        "intent": {
            "intent": "analytics",
            "entities": ["outflow"],
            "aggregation_hint": "sum",
            "complexity": "join",
            "required_output": ["segment_name"],
            "explicit_join": [{"table_hint": "dim_segments", "column_hint": "customer_id"}],
        },
        "messages": [],
    }

    result = node.table_resolver(state)
    selected = set(result["selected_tables"])

    assert ("schema", "fact_sales") in selected
    assert ("schema", "dim_segments") in selected
