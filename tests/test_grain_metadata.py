import pandas as pd

from core.schema_loader import SchemaLoader


class StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        self.calls += 1
        return self.response


def test_schema_loader_generates_missing_grains(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["sale_funnel", "fact_outflow"],
        "description": ["Воронка продаж по задачам", "Информация по фактическим оттокам"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 4,
        "table_name": ["sale_funnel", "sale_funnel", "fact_outflow", "fact_outflow"],
        "column_name": ["task_id", "task_subtipe", "client_id", "report_dt"],
        "dType": ["bigint", "varchar", "varchar", "date"],
        "description": ["ID задачи", "Подтип задачи", "ID клиента", "Отчетная дата"],
        "is_primary_key": [False, False, False, False],
        "unique_perc": [95.0, 10.0, 90.0, 1.0],
        "not_null_perc": [100.0, 100.0, 99.0, 99.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    llm = StubLLM(
        '{"tables": ['
        '{"schema": "dm", "table": "sale_funnel", "grain": "task"},'
        '{"schema": "dm", "table": "fact_outflow", "grain": "event"}'
        ']}'
    )

    assert loader.ensure_table_grains(llm) == 0
    assert llm.calls == 1

    persisted = pd.read_csv(tmp_path / "tables_list.csv")
    assert "grain" in persisted.columns
    assert persisted.loc[persisted["table_name"] == "sale_funnel", "grain"].iloc[0] == "task"
    assert persisted.loc[persisted["table_name"] == "fact_outflow", "grain"].iloc[0] == "event"


def test_schema_loader_skips_generation_when_grain_present(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel"],
        "column_name": ["task_id"],
        "dType": ["bigint"],
        "description": ["ID задачи"],
        "is_primary_key": [False],
        "unique_perc": [95.0],
        "not_null_perc": [100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    llm = StubLLM('{"tables": []}')

    assert loader.ensure_table_grains(llm) == 0
    assert llm.calls == 0
    assert loader.get_table_grain("dm", "sale_funnel") == "task"
