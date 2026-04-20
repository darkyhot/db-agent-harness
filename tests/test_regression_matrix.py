"""Regression matrix by failure class."""

import pandas as pd

from core.join_governor import decide_join_plan
from core.schema_loader import SchemaLoader
from core.semantic_frame import derive_semantic_frame
from core.where_resolver import resolve_where


def _loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["sale_funnel", "fact_outflow"],
        "description": ["Воронка продаж по задачам", "Фактические события оттока"],
        "grain": ["task", "event"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 6,
        "table_name": [
            "sale_funnel", "sale_funnel", "sale_funnel",
            "fact_outflow", "fact_outflow", "fact_outflow",
        ],
        "column_name": [
            "task_id", "is_outflow", "task_subtype",
            "saphr_id", "report_dt", "outflow_qty",
        ],
        "dType": ["bigint", "int4", "text", "text", "date", "int4"],
        "description": [
            "ID задачи", "Признак подтверждения оттока", "Подтип задачи",
            "ID сотрудника", "Отчетная дата", "Количество оттока",
        ],
        "is_primary_key": [False, False, False, False, False, False],
        "unique_perc": [95.0, 2.0, 10.0, 80.0, 1.0, 5.0],
        "not_null_perc": [100.0, 100.0, 99.0, 95.0, 99.0, 99.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    loader.ensure_value_profiles()
    return loader


def test_regression_wrong_filter_class_confirmed_outflow(tmp_path):
    loader = _loader(tmp_path)
    frame = derive_semantic_frame(
        "Посчитай количество задач с подтвержденным оттоком",
        schema_loader=loader,
    )
    result = resolve_where(
        user_input="Посчитай количество задач с подтвержденным оттоком",
        intent={"filter_conditions": []},
        selected_columns={"dm.sale_funnel": {"select": ["task_id"], "aggregate": ["task_id"]}},
        selected_tables=["dm.sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert any("is_outflow = 1" in cond for cond in result["conditions"])
    assert not any("task_subtype" in cond for cond in result["conditions"])


def test_regression_unnecessary_join_pruned(tmp_path):
    loader = _loader(tmp_path)
    decision = decide_join_plan(
        selected_tables=[("dm", "sale_funnel"), ("dm", "fact_outflow")],
        main_table=("dm", "sale_funnel"),
        locked_tables=[],
        join_requested=False,
        semantic_frame={"subject": "task", "qualifier": "confirmed_outflow"},
        requested_grain="task",
        dimension_slots=["date"],
        slot_scores={
            "dm.sale_funnel": {"date": 500.0},
            "dm.fact_outflow": {"date": 10.0},
        },
        schema_loader=loader,
    )
    assert decision["selected_tables"] == [("dm", "sale_funnel")]
