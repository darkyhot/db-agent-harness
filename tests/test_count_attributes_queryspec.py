import pandas as pd

from core.catalog_grounding import ground_query_spec
from core.column_binding import bind_columns
from core.join_analysis import detect_table_type
from core.query_ir import QuerySpec
from core.schema_loader import SchemaLoader
from core.sql_planner_deterministic import build_blueprint


def _loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["uzp_dim_gosb", "uzp_data_epk_consolidation"],
        "description": ["Справочник ТБ и ГОСБ", "Консолидация ЕПК клиентов"],
        "grain": ["dictionary", "client"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm", "dm", "dm", "dm"],
        "table_name": [
            "uzp_dim_gosb",
            "uzp_dim_gosb",
            "uzp_data_epk_consolidation",
            "uzp_data_epk_consolidation",
        ],
        "column_name": ["tb_id", "old_gosb_id", "epk_id", "tb_id"],
        "dType": ["int4", "int4", "int8", "int2"],
        "description": ["Номер ТБ", "Старый номер ГОСБ", "Идентификатор ЕПК", "Идентификатор ТБ"],
        "is_primary_key": [True, True, True, False],
        "unique_perc": [6.0, 98.0, 100.0, 0.1],
        "not_null_perc": [100.0, 100.0, 100.0, 45.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def _count_attributes_spec():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "strategy": "count_attributes",
        "entities": [{"name": "ТБ"}, {"name": "госб"}],
        "metrics": [],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.9,
    })
    assert spec is not None, errors
    return spec


def test_query_spec_count_attributes_contract():
    spec = _count_attributes_spec()

    assert spec.strategy == "count_attributes"
    assert [entity.name for entity in spec.entities] == ["ТБ", "госб"]
    assert spec.to_legacy_intent()["aggregation_hint"] == "count"
    assert spec.to_legacy_user_hints()["aggregation_preferences_list"] == [
        {"function": "count", "column": "ТБ", "distinct": True},
        {"function": "count", "column": "госб", "distinct": True},
    ]


def test_query_spec_rejects_count_attributes_without_targets():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "strategy": "count_attributes",
        "entities": [],
        "metrics": [],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.9,
    })

    assert spec is None
    assert any("count_attributes" in error for error in errors)


def test_count_attributes_grounding_prefers_single_dictionary(tmp_path):
    loader = _loader(tmp_path)
    spec = _count_attributes_spec()

    result = ground_query_spec(
        query_spec=spec,
        schema_loader=loader,
        user_input="Сколько есть ТБ и госб",
    )

    assert [source.full_name for source in result.sources] == ["dm.uzp_dim_gosb"]


def test_count_attributes_binding_and_blueprint(tmp_path):
    loader = _loader(tmp_path)
    spec = _count_attributes_spec()
    table_key = "dm.uzp_dim_gosb"
    table_structures = {table_key: loader.get_table_info("dm", "uzp_dim_gosb")}
    cols = loader.get_table_columns("dm", "uzp_dim_gosb")
    table_types = {table_key: detect_table_type("uzp_dim_gosb", cols)}

    bound = bind_columns(
        query_spec=spec,
        table_structures=table_structures,
        table_types=table_types,
        schema_loader=loader,
    )

    assert bound is not None
    assert bound["selected_columns"] == {
        table_key: {"select": ["tb_id", "old_gosb_id"], "aggregate": ["tb_id", "old_gosb_id"]}
    }
    assert "epk_id" not in str(bound["selected_columns"])

    blueprint = build_blueprint(
        spec.to_legacy_intent(),
        bound["selected_columns"],
        bound["join_spec"],
        table_types,
        {},
        user_hints=spec.to_legacy_user_hints(),
        schema_loader=loader,
        semantic_frame={},
    )

    assert blueprint["strategy"] == "simple_select"
    assert blueprint["group_by"] == []
    assert blueprint["aggregations"] == [
        {"function": "COUNT", "column": "tb_id", "alias": "count_tb_id", "distinct": True, "source_table": table_key},
        {
            "function": "COUNT",
            "column": "old_gosb_id",
            "alias": "count_old_gosb_id",
            "distinct": True,
            "source_table": table_key,
        },
    ]
