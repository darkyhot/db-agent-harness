"""Юнит-тесты apply_plan_edits: валидация по каталогу + применение whitelisted ops."""

import pandas as pd

from core.plan_edit_applier import apply_plan_edits
from core.plan_verifier_models import PlanEdit
from core.schema_loader import SchemaLoader


def _loader_with_two_tables(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["fact_outflow", "dim_gosb"],
        "description": ["Факт оттока", "Справочник ГОСБ"],
        "grain": ["day", ""],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": ["fact_outflow", "fact_outflow", "fact_outflow", "dim_gosb", "dim_gosb"],
        "column_name": ["report_dt", "gosb_id", "outflow_qty", "gosb_id", "gosb_name"],
        "dType": ["date", "int4", "int4", "int4", "varchar"],
        "description": ["Отчетная дата", "ID ГОСБ", "Метрика оттока", "ID ГОСБ", "Название ГОСБ"],
        "is_primary_key": [False, False, False, True, False],
        "unique_perc": [0.5, 5.0, 80.0, 100.0, 90.0],
        "not_null_perc": [99.0, 100.0, 100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_replace_column_swaps_id_for_name(tmp_path):
    loader = _loader_with_two_tables(tmp_path)
    selected = {
        "dm.fact_outflow": {
            "select": ["gosb_id", "outflow_qty"],
            "group_by": ["gosb_id"],
            "aggregate": ["outflow_qty"],
        }
    }
    edit = PlanEdit(
        op="replace_column",
        target_role="group_by",
        from_ref="dm.fact_outflow.gosb_id",
        to_ref="dm.dim_gosb.gosb_name",
        reason="label slot",
    )
    result = apply_plan_edits(
        selected_columns=selected,
        where_conditions=[],
        edits=[edit],
        schema_loader=loader,
    )
    assert len(result["applied"]) == 1
    assert result["selected_columns"]["dm.dim_gosb"]["group_by"] == ["gosb_name"]
    assert "gosb_id" not in result["selected_columns"]["dm.fact_outflow"].get("group_by", [])


def test_replace_column_rejects_nonexistent_target(tmp_path):
    loader = _loader_with_two_tables(tmp_path)
    selected = {"dm.fact_outflow": {"group_by": ["gosb_id"]}}
    edit = PlanEdit(
        op="replace_column",
        target_role="group_by",
        from_ref="dm.fact_outflow.gosb_id",
        to_ref="dm.dim_gosb.does_not_exist",
    )
    result = apply_plan_edits(
        selected_columns=selected,
        where_conditions=[],
        edits=[edit],
        schema_loader=loader,
    )
    assert result["applied"] == []
    assert len(result["rejected"]) == 1
    assert "not in catalog" in result["rejected"][0]["reason"]


def test_drop_filter_removes_matching_condition(tmp_path):
    loader = _loader_with_two_tables(tmp_path)
    edit = PlanEdit(
        op="drop_filter",
        from_ref="inserted_dttm",
    )
    result = apply_plan_edits(
        selected_columns={"dm.fact_outflow": {}},
        where_conditions=["inserted_dttm >= '2026-02-01'", "report_dt >= '2026-02-01'"],
        edits=[edit],
        schema_loader=loader,
    )
    assert len(result["applied"]) == 1
    assert result["where_conditions"] == ["report_dt >= '2026-02-01'"]


def test_add_table_validates_table_exists(tmp_path):
    loader = _loader_with_two_tables(tmp_path)
    selected = {"dm.fact_outflow": {"group_by": ["gosb_id"]}}
    bad = PlanEdit(op="add_table", to_ref="dm.no_such_table")
    good = PlanEdit(op="add_table", to_ref="dm.dim_gosb")
    result = apply_plan_edits(
        selected_columns=selected,
        where_conditions=[],
        edits=[bad, good],
        schema_loader=loader,
    )
    assert len(result["applied"]) == 1
    assert len(result["rejected"]) == 1
    assert "dm.dim_gosb" in result["selected_columns"]


def test_swap_aggregation_only_for_aggregate_role(tmp_path):
    loader = _loader_with_two_tables(tmp_path)
    selected = {"dm.fact_outflow": {"aggregate": ["outflow_qty"]}}
    edit = PlanEdit(
        op="swap_aggregation",
        from_ref="dm.fact_outflow.outflow_qty",
        to_ref="AVG",
    )
    result = apply_plan_edits(
        selected_columns=selected,
        where_conditions=[],
        edits=[edit],
        schema_loader=loader,
    )
    assert len(result["applied"]) == 1
    assert result["selected_columns"]["dm.fact_outflow"]["_aggregation_overrides"] == {
        "outflow_qty": "AVG",
    }
