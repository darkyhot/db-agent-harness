"""Label-слоты в CatalogGrounder: fact-таблица без *_name не покрывает слот «название X»."""

import pandas as pd

from core.catalog_grounding import (
    _source_covers_query_slots,
    _source_has_label_column_for_term,
    _target_is_label_slot,
)
from core.query_ir import DimensionSpec, QuerySpec, SourceBinding
from core.schema_loader import SchemaLoader


def _two_tables_loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["fact_outflow", "dim_gosb"],
        "description": ["Факт оттока", "Справочник ГОСБ"],
        "grain": ["day", ""],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 4,
        "table_name": ["fact_outflow", "fact_outflow", "dim_gosb", "dim_gosb"],
        "column_name": ["gosb_id", "outflow_qty", "gosb_id", "gosb_name"],
        "dType": ["int4", "int4", "int4", "varchar"],
        "description": ["ID ГОСБ", "Отток", "ID ГОСБ", "Название ГОСБ"],
        "is_primary_key": [False, False, True, False],
        "unique_perc": [5.0, 80.0, 100.0, 90.0],
        "not_null_perc": [100.0, 100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_target_is_label_slot_recognizes_russian_and_english():
    assert _target_is_label_slot("название ГОСБ")
    assert _target_is_label_slot("имя сотрудника")
    assert _target_is_label_slot("gosb_name")
    assert _target_is_label_slot("region_label")
    assert not _target_is_label_slot("gosb_id")
    assert not _target_is_label_slot("date")


def test_source_has_label_column_for_term(tmp_path):
    loader = _two_tables_loader(tmp_path)
    # fact_outflow has gosb_id but no gosb_name → no label match
    assert not _source_has_label_column_for_term(loader, "dm", "fact_outflow", "ГОСБ")
    # dim_gosb has gosb_name → match
    assert _source_has_label_column_for_term(loader, "dm", "dim_gosb", "ГОСБ")


def test_fact_table_does_not_cover_label_slot(tmp_path):
    loader = _two_tables_loader(tmp_path)
    fact_source = SourceBinding(schema="dm", table="fact_outflow", confidence=0.9)
    spec = QuerySpec(
        dimensions=[DimensionSpec(target="название ГОСБ")],
    )
    assert not _source_covers_query_slots(
        fact_source, query_spec=spec, schema_loader=loader,
    )


def test_dim_table_covers_label_slot(tmp_path):
    loader = _two_tables_loader(tmp_path)
    dim_source = SourceBinding(schema="dm", table="dim_gosb", confidence=0.9)
    spec = QuerySpec(
        dimensions=[DimensionSpec(target="название ГОСБ")],
    )
    assert _source_covers_query_slots(
        dim_source, query_spec=spec, schema_loader=loader,
    )
