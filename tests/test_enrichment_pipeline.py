import pandas as pd

from core.enrichment_pipeline import EnrichmentPipeline
from core.schema_loader import SchemaLoader


class StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        self.calls += 1
        return self.response


class StubDB:
    def __init__(self):
        self.calls = 0

    def execute_query(self, sql: str, limit: int = 1000):
        self.calls += 1
        return pd.DataFrame({"val": ["новый", "фактический отток", "новый"]})


def test_enrichment_pipeline_builds_all_artifacts(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel"],
        "description": ["Воронка продаж по задачам"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 3,
        "table_name": ["sale_funnel"] * 3,
        "column_name": ["task_id", "task_subtype", "is_outflow"],
        "dType": ["bigint", "varchar", "int4"],
        "description": ["ID задачи", "Подтип задачи", "Признак подтверждения оттока"],
        "is_primary_key": [False, False, False],
        "unique_perc": [95.0, 10.0, 2.0],
        "not_null_perc": [100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    llm = StubLLM('{"tables": [{"schema": "dm", "table": "sale_funnel", "grain": "task"}]}')
    db = StubDB()

    EnrichmentPipeline(loader, llm=llm, db_manager=db).run()

    assert loader.get_table_grain("dm", "sale_funnel") == "task"
    assert loader.get_column_semantics("dm", "sale_funnel", "task_subtype")["semantic_class"] == "enum_like"
    assert loader.get_table_semantics("dm", "sale_funnel")["grain"] == "task"
    profile = loader.get_value_profile("dm", "sale_funnel", "task_subtype")
    assert "фактический отток" in profile.get("known_terms", [])
    assert db.calls >= 1


def test_enrichment_pipeline_reuses_existing_artifacts(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 2,
        "table_name": ["sale_funnel"] * 2,
        "column_name": ["task_id", "task_subtype"],
        "dType": ["bigint", "varchar"],
        "description": ["ID задачи", "Подтип задачи"],
        "is_primary_key": [True, False],
        "unique_perc": [100.0, 10.0],
        "not_null_perc": [100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    artifacts = {
        "column_semantics.json": '{"dm.sale_funnel.task_subtype": {"semantic_class": "enum_like"}}',
        "table_semantics.json": '{"dm.sale_funnel": {"grain": "task", "table_role": "fact"}}',
        "column_value_profiles.json": '{"dm.sale_funnel.task_subtype": {"known_terms": ["старое значение"]}}',
        "semantic_lexicon.json": '{"subjects": {"задачи": ["task"]}}',
        "rule_registry.json": '{"rules": [{"type": "metric", "name": "count"}]}',
    }
    before = {}
    for name, content in artifacts.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        before[name] = path.read_text(encoding="utf-8")

    loader = SchemaLoader(data_dir=tmp_path)
    llm = StubLLM('{"tables": [{"schema": "dm", "table": "sale_funnel", "grain": "task"}]}')
    db = StubDB()

    EnrichmentPipeline(loader, llm=llm, db_manager=db).run()

    assert llm.calls == 0
    assert db.calls == 0
    for name, content in before.items():
        assert (tmp_path / name).read_text(encoding="utf-8") == content


def test_enrichment_pipeline_regenerates_missing_artifact(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 2,
        "table_name": ["sale_funnel"] * 2,
        "column_name": ["task_id", "task_subtype"],
        "dType": ["bigint", "varchar"],
        "description": ["ID задачи", "Подтип задачи"],
        "is_primary_key": [True, False],
        "unique_perc": [100.0, 10.0],
        "not_null_perc": [100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    (tmp_path / "column_semantics.json").write_text(
        '{"dm.sale_funnel.task_subtype": {"semantic_class": "enum_like"}}',
        encoding="utf-8",
    )
    (tmp_path / "table_semantics.json").write_text(
        '{"dm.sale_funnel": {"grain": "task", "table_role": "fact"}}',
        encoding="utf-8",
    )
    (tmp_path / "semantic_lexicon.json").write_text(
        '{"subjects": {"задачи": ["task"]}}',
        encoding="utf-8",
    )
    (tmp_path / "rule_registry.json").write_text(
        '{"rules": [{"type": "metric", "name": "count"}]}',
        encoding="utf-8",
    )

    loader = SchemaLoader(data_dir=tmp_path)
    llm = StubLLM('{"tables": [{"schema": "dm", "table": "sale_funnel", "grain": "task"}]}')
    db = StubDB()

    EnrichmentPipeline(loader, llm=llm, db_manager=db).run()

    assert (tmp_path / "column_value_profiles.json").exists()
    assert db.calls >= 1
