"""Тесты для core/column_selector_deterministic.py."""

import pandas as pd


def test_choose_best_metric_prefers_outflow_qty_over_is_outflow_for_sum(tmp_path):
    from core.column_selector_deterministic import _choose_best_column
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["uzp_dwh_fact_outflow", "uzp_data_split_mzp_sale_funnel"],
        "description": [
            "Фактический отток клиентов",
            "Воронка продаж и признаки оттока",
        ],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": [
            "uzp_dwh_fact_outflow",
            "uzp_dwh_fact_outflow",
            "uzp_dwh_fact_outflow",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_split_mzp_sale_funnel",
        ],
        "column_name": [
            "inn",
            "report_dt",
            "outflow_qty",
            "segment_name",
            "is_outflow",
        ],
        "dType": [
            "varchar",
            "date",
            "int4",
            "varchar",
            "int4",
        ],
        "description": [
            "ИНН клиента",
            "Отчётная дата",
            "Кол-во ФЛ переставших быть ЗП клиентами",
            "Сегмент клиента",
            "Признак подтверждения оттока",
        ],
        "is_primary_key": [
            False,
            False,
            False,
            False,
            False,
        ],
        "unique_perc": [
            85.0,
            1.0,
            10.0,
            5.0,
            2.0,
        ],
        "not_null_perc": [
            95.0,
            99.0,
            95.0,
            99.99,
            100.0,
        ],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    table_structures = {
        "dm.uzp_dwh_fact_outflow": "outflow fact with inn, report_dt, outflow_qty",
        "dm.uzp_data_split_mzp_sale_funnel": "sale funnel with segment_name, is_outflow",
    }
    table_types = {
        "dm.uzp_dwh_fact_outflow": "fact",
        "dm.uzp_data_split_mzp_sale_funnel": "fact",
    }

    best = _choose_best_column(
        table_structures=table_structures,
        table_types=table_types,
        schema_loader=loader,
        slot="отток_ottok",
        require_numeric=True,
        agg_hint="sum",
    )

    assert best == ("dm.uzp_dwh_fact_outflow", "outflow_qty")


def test_select_columns_single_entity_count_prefers_pk_over_aux_code(tmp_path):
    from core.column_selector_deterministic import select_columns
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_data_split_mzp_sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": ["uzp_data_split_mzp_sale_funnel"] * 5,
        "column_name": ["report_dt", "task_code", "uzp_task_code", "task_subtype", "task_category"],
        "dType": ["date", "text", "text", "text", "text"],
        "description": [
            "Отчетная дата",
            "Код задачи",
            "Код задачи УЗП",
            "Подтип задачи",
            "Категория задачи",
        ],
        "is_primary_key": [False, True, False, False, False],
        "unique_perc": [0.5, 100.0, 36.68, 2.62, 0.02],
        "not_null_perc": [99.0, 100.0, 36.68, 37.06, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    result = select_columns(
        intent={
            "aggregation_hint": "count",
            "entities": ["задачи", "отток"],
            "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
        },
        table_structures={"dm.uzp_data_split_mzp_sale_funnel": "Воронка продаж по задачам"},
        table_types={"dm.uzp_data_split_mzp_sale_funnel": "fact"},
        join_analysis_data={},
        schema_loader=loader,
        user_input="Сколько задач по фактическому оттоку за февраль 2026",
        semantic_frame={
            "subject": "task",
            "requires_single_entity_count": True,
            "output_dimensions": [],
        },
    )

    roles = result["selected_columns"]["dm.uzp_data_split_mzp_sale_funnel"]
    assert roles["aggregate"] == ["task_code"]
    assert roles["select"] == ["task_code"]
    assert "group_by" not in roles or not roles["group_by"]
    assert "uzp_task_code" not in roles.get("select", [])


def test_select_columns_scalar_count_without_semantic_flag_still_avoids_group_by(tmp_path):
    from core.column_selector_deterministic import select_columns
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_data_split_mzp_sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": ["uzp_data_split_mzp_sale_funnel"] * 5,
        "column_name": ["report_dt", "task_code", "uzp_task_code", "task_subtype", "task_category"],
        "dType": ["date", "text", "text", "text", "text"],
        "description": [
            "Отчетная дата",
            "Код задачи",
            "Код задачи УЗП",
            "Подтип задачи",
            "Категория задачи",
        ],
        "is_primary_key": [False, True, False, False, False],
        "unique_perc": [0.5, 100.0, 36.68, 2.62, 0.02],
        "not_null_perc": [99.0, 100.0, 36.68, 37.06, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    result = select_columns(
        intent={
            "aggregation_hint": "count",
            "entities": ["задачи", "отток"],
            "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
        },
        table_structures={"dm.uzp_data_split_mzp_sale_funnel": "Воронка продаж по задачам"},
        table_types={"dm.uzp_data_split_mzp_sale_funnel": "fact"},
        join_analysis_data={},
        schema_loader=loader,
        user_input="Сколько задач по фактическому оттоку за февраль 2026",
        semantic_frame={"subject": "task", "output_dimensions": []},
    )

    roles = result["selected_columns"]["dm.uzp_data_split_mzp_sale_funnel"]
    assert roles["aggregate"] == ["task_code"]
    assert roles["select"] == ["task_code"]
    assert "group_by" not in roles or not roles["group_by"]
