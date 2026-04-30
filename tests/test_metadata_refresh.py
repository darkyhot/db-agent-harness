import pandas as pd
import yaml

from core.metadata_refresh import (
    MetadataRefreshService,
    _find_candidate_primary_key,
    parse_table_refs,
)
from core.schema_loader import SchemaLoader


class StubLLM:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        self.calls.append((system_prompt, user_prompt))
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
            "foreign_key_target", "sample_values", "partition_key",
        ]),
    )

    monkeypatch.setattr("core.metadata_refresh.inspect", lambda engine: StubInspector())

    service = MetadataRefreshService(
        loader,
        StubDB(),
        StubLLM(),
        targets_path=tmp_path / "metadata_targets.yaml",
    )

    result = service.add_targets(["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"])

    assert result["added"] == ["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"]
    tables_df = pd.read_csv(tmp_path / "tables_list.csv")
    attrs_df = pd.read_csv(tmp_path / "attr_list.csv")
    assert tables_df.loc[0, "description"] == "Описание из view"
    assert set(attrs_df["description"]) == {"ID из view", "Флаг из view"}
    table_few_shots = yaml.safe_load((tmp_path / "table_description_few_shots.yaml").read_text(encoding="utf-8"))
    column_few_shots = yaml.safe_load((tmp_path / "column_description_few_shots.yaml").read_text(encoding="utf-8"))
    assert table_few_shots["tables"][0]["description"] == "Описание из view"
    assert "schema_name" not in table_few_shots["tables"][0]
    assert any(item["column_name"] == "id" and item["description"] == "ID из view" for item in column_few_shots["columns"])
    assert "table_name" not in column_few_shots["columns"][0]


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
            "foreign_key_target", "sample_values", "partition_key",
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
            "foreign_key_target", "sample_values", "partition_key",
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


def test_metadata_service_add_targets_rolls_back_manifest_for_failed_refresh(tmp_path):
    loader = SchemaLoader(data_dir=tmp_path)
    loader.replace_catalog(
        pd.DataFrame(columns=["schema_name", "table_name", "description", "grain"]),
        pd.DataFrame(columns=[
            "schema_name", "table_name", "column_name", "dType",
            "is_not_null", "description", "is_primary_key",
            "not_null_perc", "unique_perc",
            "foreign_key_target", "sample_values", "partition_key",
        ]),
    )

    service = MetadataRefreshService(
        loader,
        StubDB(),
        StubLLM(),
        targets_path=tmp_path / "metadata_targets.yaml",
    )

    original_refresh_tables = service.refresh_tables

    def failing_refresh_tables(*args, **kwargs):
        result = original_refresh_tables(*args, **kwargs)
        result["failed"] = ["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"]
        result["refreshed"] = []
        return result

    service.refresh_tables = failing_refresh_tables  # type: ignore[method-assign]

    result = service.add_targets(["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"])

    assert result["added"] == []
    assert result["refresh"]["failed"] == ["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"]
    assert service.list_targets() == []


def test_metadata_service_add_targets_rolls_back_manifest_when_refresh_raises(tmp_path):
    loader = SchemaLoader(data_dir=tmp_path)
    loader.replace_catalog(
        pd.DataFrame(columns=["schema_name", "table_name", "description", "grain"]),
        pd.DataFrame(columns=[
            "schema_name", "table_name", "column_name", "dType",
            "is_not_null", "description", "is_primary_key",
            "not_null_perc", "unique_perc",
            "foreign_key_target", "sample_values", "partition_key",
        ]),
    )

    service = MetadataRefreshService(
        loader,
        StubDB(),
        StubLLM(),
        targets_path=tmp_path / "metadata_targets.yaml",
    )

    def crashing_refresh_tables(*args, **kwargs):
        raise RuntimeError("db is down")

    service.refresh_tables = crashing_refresh_tables  # type: ignore[method-assign]

    try:
        service.add_targets(["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"])
    except RuntimeError as exc:
        assert str(exc) == "db is down"
    else:
        raise AssertionError("Ожидалось исключение refresh_tables")

    assert service.list_targets() == []


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
    assert (data_dir / "table_description_few_shots.yaml").exists()
    assert (data_dir / "column_description_few_shots.yaml").exists()


def test_metadata_service_deduplicates_column_few_shots(tmp_path, monkeypatch):
    class DuplicateColumnInspector(StubInspector):
        def get_columns(self, table, schema=None):
            if table == "orders":
                return [
                    {"name": "id", "type": "bigint", "nullable": False, "comment": "ID заказа"},
                    {"name": "status", "type": "text", "nullable": True, "comment": "Статус заказа"},
                ]
            if table == "payments":
                return [
                    {"name": "id", "type": "bigint", "nullable": False, "comment": "ID платежа"},
                    {"name": "status", "type": "text", "nullable": True, "comment": "Статус платежа"},
                ]
            return super().get_columns(table, schema=schema)

        def get_table_comment(self, table, schema=None):
            if table == "orders":
                return {"text": "Заказы"}
            if table == "payments":
                return {"text": "Платежи"}
            return super().get_table_comment(table, schema=schema)

    loader = SchemaLoader(data_dir=tmp_path)
    loader.replace_catalog(
        pd.DataFrame(columns=["schema_name", "table_name", "description", "grain"]),
        pd.DataFrame(columns=[
            "schema_name", "table_name", "column_name", "dType",
            "is_not_null", "description", "is_primary_key",
            "not_null_perc", "unique_perc",
            "foreign_key_target", "sample_values", "partition_key",
        ]),
    )

    monkeypatch.setattr("core.metadata_refresh.inspect", lambda engine: DuplicateColumnInspector())

    service = MetadataRefreshService(
        loader,
        StubDB(),
        StubLLM(),
        targets_path=tmp_path / "metadata_targets.yaml",
    )

    service.add_targets([
        "s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders",
        "s_grnplm_ld_salesntwrk_pcap_sn_uzp.payments",
    ])

    column_few_shots = yaml.safe_load((tmp_path / "column_description_few_shots.yaml").read_text(encoding="utf-8"))
    assert sum(1 for item in column_few_shots["columns"] if item["column_name"] == "status") == 1
    assert "table_name" not in column_few_shots["columns"][0]


def test_metadata_service_builds_few_shots_before_generating_missing_descriptions(tmp_path, monkeypatch):
    class OrderedInspector(StubInspector):
        def get_columns(self, table, schema=None):
            if table == "commented":
                return [
                    {"name": "client_id", "type": "bigint", "nullable": False, "comment": "Идентификатор клиента"},
                ]
            if table == "generated":
                return [
                    {"name": "mystery_col", "type": "text", "nullable": True, "comment": ""},
                ]
            return super().get_columns(table, schema=schema)

        def get_table_comment(self, table, schema=None):
            if table == "commented":
                return {"text": "Клиентская таблица"}
            if table == "generated":
                return {"text": ""}
            return super().get_table_comment(table, schema=schema)

        def get_pk_constraint(self, table, schema=None):
            return {"constrained_columns": []}

    class OrderedDB(StubDB):
        def get_random_sample(self, schema, table, n=100000, columns=None):
            if table == "commented":
                return pd.DataFrame({"client_id": [1, 2, 3]})
            return pd.DataFrame({"mystery_col": ["a", "b", "c"]})

    llm = StubLLM()
    loader = SchemaLoader(data_dir=tmp_path)
    loader.replace_catalog(
        pd.DataFrame(columns=["schema_name", "table_name", "description", "grain"]),
        pd.DataFrame(columns=[
            "schema_name", "table_name", "column_name", "dType",
            "is_not_null", "description", "is_primary_key",
            "not_null_perc", "unique_perc",
            "foreign_key_target", "sample_values", "partition_key",
        ]),
    )
    monkeypatch.setattr("core.metadata_refresh.inspect", lambda engine: OrderedInspector())

    service = MetadataRefreshService(
        loader,
        OrderedDB(),
        llm,
        targets_path=tmp_path / "metadata_targets.yaml",
    )

    service.add_targets([
        "s_grnplm_ld_salesntwrk_pcap_sn_t_uzp.commented",
        "s_grnplm_ld_salesntwrk_pcap_sn_t_uzp.generated",
    ])

    assert llm.calls, "Ожидался хотя бы один LLM вызов для генерации описаний"
    prompt_text = "\n".join(user_prompt for _, user_prompt in llm.calls)
    assert "Клиентская таблица" in prompt_text or "Идентификатор клиента" in prompt_text


def test_metadata_service_add_targets_rebuilds_few_shots_without_scanning_manifest(tmp_path, monkeypatch):
    loader = SchemaLoader(data_dir=tmp_path)
    loader.replace_catalog(
        pd.DataFrame([
            {
                "schema_name": "dm",
                "table_name": "clients",
                "description": "Клиенты",
                "grain": "",
            }
        ]),
        pd.DataFrame([
            {
                "schema_name": "dm",
                "table_name": "clients",
                "column_name": "client_id",
                "dType": "bigint",
                "is_not_null": True,
                "description": "Идентификатор клиента",
                "is_primary_key": True,
                "not_null_perc": 100.0,
                "unique_perc": 100.0,
                "foreign_key_target": "",
                "sample_values": "",
                "partition_key": False,
            }
        ]),
    )
    (tmp_path / "metadata_targets.yaml").write_text(
        "tables:\n"
        "  - schema_name: dm\n"
        "    table_name: clients\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("core.metadata_refresh.inspect", lambda engine: StubInspector())

    service = MetadataRefreshService(
        loader,
        StubDB(),
        StubLLM(),
        targets_path=tmp_path / "metadata_targets.yaml",
    )
    progress_messages: list[str] = []

    result = service.add_targets(
        ["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"],
        progress_callback=progress_messages.append,
    )

    assert result["added"] == ["s_grnplm_ld_salesntwrk_pcap_sn_uzp.orders"]
    assert all("few-shots" not in message.lower() for message in progress_messages)
    table_few_shots = yaml.safe_load((tmp_path / "table_description_few_shots.yaml").read_text(encoding="utf-8"))
    column_few_shots = yaml.safe_load((tmp_path / "column_description_few_shots.yaml").read_text(encoding="utf-8"))
    assert any(item["table_name"] == "clients" and item["description"] == "Клиенты" for item in table_few_shots["tables"])
    assert any(item["table_name"] == "orders" and item["description"] == "Описание из view" for item in table_few_shots["tables"])
    assert any(item["column_name"] == "client_id" and item["description"] == "Идентификатор клиента" for item in column_few_shots["columns"])
    assert any(item["column_name"] == "id" and item["description"] == "ID из view" for item in column_few_shots["columns"])


def test_build_column_prompt_supports_batch_with_yaml_few_shots(tmp_path):
    loader = SchemaLoader(data_dir=tmp_path)
    service = MetadataRefreshService(
        loader,
        StubDB(),
        StubLLM(),
        targets_path=tmp_path / "metadata_targets.yaml",
    )
    (tmp_path / "column_description_few_shots.yaml").write_text(
        yaml.safe_dump(
            {
                "columns": [
                    {
                        "column_name": "client_id",
                        "description": "Идентификатор клиента",
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _, user_prompt = service._build_column_prompt(
        "schema_x",
        "target_table",
        ["first_col", "second_col", "third_col"],
    )

    assert "Пример 1:" in user_prompt
    assert "- first_col" in user_prompt
    assert "- second_col" in user_prompt
    assert "- third_col" in user_prompt
    examples_block = user_prompt.split("Примеры:\n", 1)[1].split("Расшифруй список атрибутов", 1)[0]
    assert "Таблица:" not in examples_block


def test_generate_column_descriptions_uses_exact_few_shot_without_llm(tmp_path):
    loader = SchemaLoader(data_dir=tmp_path)
    llm = StubLLM()
    service = MetadataRefreshService(
        loader,
        StubDB(),
        llm,
        targets_path=tmp_path / "metadata_targets.yaml",
    )
    (tmp_path / "column_description_few_shots.yaml").write_text(
        yaml.safe_dump(
            {
                "columns": [
                    {
                        "column_name": "client_id",
                        "description": "Идентификатор клиента",
                    },
                    {
                        "column_name": "status_cd",
                        "description": "Код статуса",
                    },
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = service._generate_column_descriptions(
        "schema_x",
        "target_table",
        ["client_id", "status_cd"],
    )

    assert result == {
        "client_id": "Идентификатор клиента",
        "status_cd": "Код статуса",
    }
    assert llm.calls == []


def test_generate_table_description_uses_exact_few_shot_without_llm(tmp_path):
    loader = SchemaLoader(data_dir=tmp_path)
    llm = StubLLM()
    service = MetadataRefreshService(
        loader,
        StubDB(),
        llm,
        targets_path=tmp_path / "metadata_targets.yaml",
    )
    (tmp_path / "table_description_few_shots.yaml").write_text(
        yaml.safe_dump(
            {
                "tables": [
                    {
                        "table_name": "known_table",
                        "description": "Известное описание таблицы",
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = service._generate_table_description(
        "schema_x",
        "known_table",
        pd.DataFrame([{"column_name": "id", "description": "Идентификатор"}]),
        pd.DataFrame({"id": [1, 2]}),
    )

    assert result == "Известное описание таблицы"
    assert llm.calls == []
