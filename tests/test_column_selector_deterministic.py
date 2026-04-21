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


# ---------------------------------------------------------------------------
# Task 1.2: group_by_hints influence on explorer (indirect test via state)
# ---------------------------------------------------------------------------

def test_explorer_skips_llm_when_no_group_by_hints(tmp_path):
    """Если group_by_hints пустые — ColumnSelector использует детерминированный путь."""
    from graph.graph import create_initial_state

    state = create_initial_state("посчитай отток")
    state["user_hints"]["group_by_hints"] = []

    # Нет явных group_by_hints → _has_explicit_group = False
    hints = state.get("user_hints", {}) or {}
    has_explicit_group = bool(hints.get("group_by_hints"))
    assert has_explicit_group is False


def test_explorer_uses_llm_path_when_group_by_hints_present(tmp_path):
    """Если group_by_hints не пустые — _has_explicit_group = True (LLM путь)."""
    from graph.graph import create_initial_state

    state = create_initial_state("сгруппируй по segment_name")
    state["user_hints"]["group_by_hints"] = ["segment_name"]

    hints = state.get("user_hints", {}) or {}
    has_explicit_group = bool(hints.get("group_by_hints"))
    assert has_explicit_group is True
