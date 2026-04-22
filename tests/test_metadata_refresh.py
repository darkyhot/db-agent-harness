import pandas as pd

from core.metadata_refresh import (
    MetadataRefreshService,
    _find_candidate_primary_key,
    parse_table_refs,
)
from core.schema_loader import SchemaLoader


class StubLLM:
    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        if "один на строку" in user_prompt:
            return "идентификатор\nпризнак"
        return "1. Назначение\n2. Применение\n3. Ограничения\n4. Ключевые атрибуты"


class StubInspector:
    def get_columns(self, table, schema=None):
        if schema == "s_grnplm_as_salesntwrk_pcap_sn_view":
            return [
                {"name": "id", "type": "bigint", "nullable": False, "comment": "ID из view"},
                {"name": "flag", "type": "int4", "nullable": True, "comment": "Флаг из view"},
            ]
        return [
            {"name": "id", "type": "bigint", "nullable": False, "comment": ""},
            {"name": "flag", "type": "int4", "nullable": True, "comment": ""},
        ]

    def get_table_comment(self, table, schema=None):
        if schema == "s_grnplm_as_salesntwrk_pcap_sn_view":
            return {"text": "Описание из view"}
        return {"text": ""}

    def get_pk_constraint(self, table, schema=None):
        return {"constrained_columns": ["id"]}


class StubDB:
    def __init__(self):
        self.random_calls = 0
        self.existing_tables = None

    def get_engine(self):
        return object()

    def table_exists(self, schema, table):
        if self.existing_tables is None:
            return True
        return (schema, table) in self.existing_tables

    def get_random_sample(self, schema, table, n=100000, columns=None):
        self.random_calls += 1
        return pd.DataFrame({"id": [1, 2, 3], "flag": [1, 0, 1]})


def test_parse_table_refs_supports_commas_and_spaces():
    parsed = parse_table_refs("dm.sales, dm.clients support.tickets")
    assert parsed == [
        ("dm", "sales"),
        ("dm", "clients"),
        ("support", "tickets"),
    ]


def test_find_candidate_primary_key_prefers_dimensions_before_metrics():
    df = pd.DataFrame({
        "client_id": [1, 1, 2, 2],
        "report_dt": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "amount_amt": [100, 200, 100, 200],
    })

    result = _find_candidate_primary_key(df, max_columns=3)

    assert result == ["client_id", "report_dt"]


def test_find_candidate_primary_key_uses_metric_as_fallback_when_needed():
    df = pd.DataFrame({
        "client_id": [1, 1, 1, 1],
        "report_dt": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "amount_amt": [100, 200, 100, 200],
    })

    result = _find_candidate_primary_key(df, max_columns=3)

    assert result == ["report_dt", "amount_amt"]


def test_find_candidate_primary_key_reports_progress():
    df = pd.DataFrame({
        "client_id": [1, 1, 2, 2],
        "report_dt": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "status": ["a", "b", "a", "b"],
    })
    calls = []

    result = _find_candidate_primary_key(
        df,
        max_columns=3,
        progress_callback=lambda checked, total: calls.append((checked, total)),
    )

    assert result == ["client_id", "report_dt"]
    assert calls
    assert all(total >= checked >= 1 for checked, total in calls)


def test_metadata_service_add_targets_refreshes_catalog(tmp_path, monkeypatch):
    loader = SchemaLoader(data_dir=tmp_path)
    loader.replace_catalog(
        pd.DataFrame(columns=["schema_name", "table_name", "description", "grain"]),
        pd.DataFrame(columns=[
            "schema_name", "table_name", "column_name", "dType",
            "is_not_null", "description", "is_primary_key",
            "not_null_perc", "unique_perc",
            "foreign_key_target", "sample_values", "partition_key", "synonyms",
        ]),
    )

    monkeypatch.setattr("core.metadata_refresh.inspect", lambda engine: StubInspector())

    service = MetadataRefreshService(
        loader,
        StubDB(),
        StubLLM(),
        targets_path=tmp_path / "metadata_targets.yaml",
        examples_path=tmp_path / "examples.txt",
    )

    result = service.add_targets(["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"])

    assert result["added"] == ["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"]
    tables_df = pd.read_csv(tmp_path / "tables_list.csv")
    attrs_df = pd.read_csv(tmp_path / "attr_list.csv")
    assert tables_df.loc[0, "description"] == "Описание из view"
    assert set(attrs_df["description"]) == {"ID из view", "Флаг из view"}


def test_metadata_service_remove_targets_prunes_catalog(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["sales", "clients"],
        "description": ["Sales", "Clients"],
        "grain": ["", ""],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["sales", "clients"],
        "column_name": ["id", "id"],
        "dType": ["bigint", "bigint"],
        "is_not_null": [True, True],
        "description": ["ID", "ID"],
        "is_primary_key": [True, True],
        "not_null_perc": [100.0, 100.0],
        "unique_perc": [100.0, 100.0],
        "foreign_key_target": ["", ""],
        "sample_values": ["", ""],
        "partition_key": [False, False],
        "synonyms": ["", ""],
    })
    loader = SchemaLoader(data_dir=tmp_path)
    loader.replace_catalog(tables_df, attrs_df)
    (tmp_path / "metadata_targets.yaml").write_text(
        "tables:\n"
        "  - schema_name: dm\n"
        "    table_name: sales\n"
        "  - schema_name: dm\n"
        "    table_name: clients\n",
        encoding="utf-8",
    )

    service = MetadataRefreshService(
        loader,
        StubDB(),
        StubLLM(),
        targets_path=tmp_path / "metadata_targets.yaml",
    )

    result = service.remove_targets(["dm.sales"])

    assert result["removed"] == ["dm.sales"]
    persisted_tables = pd.read_csv(tmp_path / "tables_list.csv")
    persisted_attrs = pd.read_csv(tmp_path / "attr_list.csv")
    assert persisted_tables["table_name"].tolist() == ["clients"]
    assert persisted_attrs["table_name"].tolist() == ["clients"]


def test_metadata_service_add_targets_rejects_invalid_schema(tmp_path):
    loader = SchemaLoader(data_dir=tmp_path)
    loader.replace_catalog(
        pd.DataFrame(columns=["schema_name", "table_name", "description", "grain"]),
        pd.DataFrame(columns=[
            "schema_name", "table_name", "column_name", "dType",
            "is_not_null", "description", "is_primary_key",
            "not_null_perc", "unique_perc",
            "foreign_key_target", "sample_values", "partition_key", "synonyms",
        ]),
    )

    db = StubDB()
    service = MetadataRefreshService(
        loader,
        db,
        StubLLM(),
        targets_path=tmp_path / "metadata_targets.yaml",
    )

    result = service.add_targets(["unknown_schema.orders"])

    assert result["added"] == []
    assert result["invalid_schemas"] == ["unknown_schema.orders"]
    assert result["missing_tables"] == []


def test_metadata_service_add_targets_rejects_missing_table(tmp_path):
    loader = SchemaLoader(data_dir=tmp_path)
    loader.replace_catalog(
        pd.DataFrame(columns=["schema_name", "table_name", "description", "grain"]),
        pd.DataFrame(columns=[
            "schema_name", "table_name", "column_name", "dType",
            "is_not_null", "description", "is_primary_key",
            "not_null_perc", "unique_perc",
            "foreign_key_target", "sample_values", "partition_key", "synonyms",
        ]),
    )

    db = StubDB()
    db.existing_tables = set()
    service = MetadataRefreshService(
        loader,
        db,
        StubLLM(),
        targets_path=tmp_path / "metadata_targets.yaml",
    )

    result = service.add_targets([f"s_grnplm_ld_salesntwrk_pcap_sn_uzp.missing_table"])

    assert result["added"] == []
    assert result["invalid_schemas"] == []
    assert result["missing_tables"] == ["s_grnplm_ld_salesntwrk_pcap_sn_uzp.missing_table"]


def test_metadata_service_recreates_storage_when_data_dir_is_missing(tmp_path, monkeypatch):
    data_dir = tmp_path / "data_for_agent"
    loader = SchemaLoader(data_dir=data_dir)
    db = StubDB()

    monkeypatch.setattr("core.metadata_refresh.inspect", lambda engine: StubInspector())

    service = MetadataRefreshService(
        loader,
        db,
        StubLLM(),
        targets_path=data_dir / "metadata_targets.yaml",
        examples_path=tmp_path / "examples.txt",
    )

    # Имитируем ручную очистку каталога пользователем после инициализации.
    for path in data_dir.glob("*"):
        path.unlink()
    data_dir.rmdir()

    result = service.add_targets(["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"])

    assert result["added"] == ["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"]
    assert (data_dir / "metadata_targets.yaml").exists()
    assert (data_dir / "tables_list.csv").exists()
    assert (data_dir / "attr_list.csv").exists()
