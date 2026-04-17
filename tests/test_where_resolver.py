import pandas as pd

from core.schema_loader import SchemaLoader
from core.semantic_frame import derive_semantic_frame
from core.where_resolver import resolve_where


def _loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_data_split_mzp_sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 4,
        "table_name": ["uzp_data_split_mzp_sale_funnel"] * 4,
        "column_name": ["report_dt", "task_code", "task_subtype", "is_outflow"],
        "dType": ["date", "text", "text", "int4"],
        "description": ["Отчетная дата", "Код задачи", "Подтип задачи", "Признак подтверждения оттока"],
        "is_primary_key": [False, False, False, False],
        "unique_perc": [0.5, 90.0, 10.0, 2.0],
        "not_null_perc": [99.0, 100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_where_resolver_adds_confirmed_outflow_flag(tmp_path):
    loader = _loader(tmp_path)
    loader.ensure_value_profiles()
    frame = derive_semantic_frame(
        "Посчитай количество задач с подтвержденным оттоком",
        schema_loader=loader,
    )
    result = resolve_where(
        user_input="Посчитай количество задач с подтвержденным оттоком",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert any("is_outflow = 1" in cond for cond in result["conditions"])
    assert result["applied_rules"]
    assert next(iter(result["filter_candidates"].values()))[0]["column"] == "is_outflow"
    assert not any("task_subtype" in cond for cond in result["conditions"])


def test_where_resolver_adds_factual_outflow_task_subtype(tmp_path):
    loader = _loader(tmp_path)
    loader.ensure_value_profiles()
    frame = derive_semantic_frame(
        "Посчитай количество задач по фактическому оттоку",
        schema_loader=loader,
    )
    result = resolve_where(
        user_input="Посчитай количество задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert any("task_subtype ILIKE '%фактическому%'" in cond for cond in result["conditions"])
    assert any("task_subtype ILIKE '%оттоку%'" in cond for cond in result["conditions"])
    assert result["applied_rules"]
    assert any(cands[0]["column"] == "task_subtype" for cands in result["filter_candidates"].values() if cands)


def test_where_resolver_respects_explicit_column_clarification(tmp_path):
    loader = _loader(tmp_path)
    loader.ensure_value_profiles()
    frame = derive_semantic_frame(
        "Посчитай количество задач по фактическому оттоку",
        schema_loader=loader,
    )
    result = resolve_where(
        user_input="Посчитай количество задач по фактическому оттоку\nУточнение пользователя: task_subtype",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert result["needs_clarification"] is False
    assert any("task_subtype" in cond for cond in result["conditions"])
