import pandas as pd

from core.join_governor import decide_join_plan
from core.schema_loader import SchemaLoader


def _loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["sale_funnel", "epk_consolidation"],
        "description": ["Воронка продаж по задачам", "Клиентский справочник с сегментами"],
        "grain": ["task", "client"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 6,
        "table_name": [
            "sale_funnel", "sale_funnel", "sale_funnel",
            "epk_consolidation", "epk_consolidation", "epk_consolidation",
        ],
        "column_name": ["task_id", "report_dt", "is_outflow", "inn", "segment_name", "segment_id"],
        "dType": ["bigint", "date", "int4", "text", "text", "bigint"],
        "description": [
            "ID задачи",
            "Отчетная дата",
            "Признак подтверждения оттока",
            "ИНН клиента",
            "Сегмент клиента",
            "ID сегмента",
        ],
        "is_primary_key": [False, False, False, False, False, True],
        "unique_perc": [95.0, 1.0, 2.0, 90.0, 5.0, 100.0],
        "not_null_perc": [100.0, 99.0, 100.0, 95.0, 99.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_join_governor_prunes_unnecessary_join(tmp_path):
    loader = _loader(tmp_path)
    decision = decide_join_plan(
        selected_tables=[("dm", "sale_funnel"), ("dm", "epk_consolidation")],
        main_table=("dm", "sale_funnel"),
        locked_tables=[],
        join_requested=False,
        semantic_frame={"subject": "task", "qualifier": "confirmed_outflow"},
        requested_grain="task",
        dimension_slots=["date"],
        slot_scores={
            "dm.sale_funnel": {"date": 500.0},
            "dm.epk_consolidation": {"date": 0.0},
        },
        schema_loader=loader,
    )

    assert decision["allow_join"] is False
    assert decision["selected_tables"] == [("dm", "sale_funnel")]


def test_join_governor_keeps_external_table_when_dimension_coverage_is_better(tmp_path):
    loader = _loader(tmp_path)
    decision = decide_join_plan(
        selected_tables=[("dm", "sale_funnel"), ("dm", "epk_consolidation")],
        main_table=("dm", "sale_funnel"),
        locked_tables=[],
        join_requested=False,
        semantic_frame={"subject": "task"},
        requested_grain="task",
        dimension_slots=["segment"],
        slot_scores={
            "dm.sale_funnel": {"segment": 10.0},
            "dm.epk_consolidation": {"segment": 220.0},
        },
        schema_loader=loader,
    )

    assert decision["allow_join"] is True
    assert ("dm", "epk_consolidation") in decision["selected_tables"]
