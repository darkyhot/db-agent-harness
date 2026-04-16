import pandas as pd

from core.schema_loader import SchemaLoader
from core.semantic_frame import derive_semantic_frame


def _loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 4,
        "table_name": ["sale_funnel"] * 4,
        "column_name": ["report_dt", "task_code", "task_subtype", "is_outflow"],
        "dType": ["date", "text", "text", "int4"],
        "description": ["Отчетная дата", "Код задачи", "Подтип задачи", "Признак подтверждения оттока"],
        "is_primary_key": [False, False, False, False],
        "unique_perc": [0.5, 90.0, 10.0, 2.0],
        "not_null_perc": [99.0, 100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    loader.ensure_semantic_registry()
    return loader


def test_semantic_frame_extracts_metric_and_output_dimensions(tmp_path):
    loader = _loader(tmp_path)
    frame = derive_semantic_frame(
        "Посчитай количество задач с подтвержденным оттоком за февраль 2026 по дате и сегменту",
        {
            "aggregation_hint": "count",
            "entities": ["задачи", "отток", "дата", "сегмент"],
            "required_output": ["дата", "сегмент"],
        },
        schema_loader=loader,
    )

    assert frame["subject"] == "task"
    assert frame["metric_intent"] == "count"
    assert frame["requested_grain"] == "task"
    assert "дата" in frame["output_dimensions"]
    assert "сегмент" in frame["output_dimensions"]
    assert frame["requires_single_entity_count"] is False
    assert frame["period_kind"] == "calendar"
    assert frame["filter_intents"]
    assert any(item["kind"] == "boolean_true" for item in frame["filter_intents"])


def test_semantic_frame_detects_listing_and_ambiguity():
    frame = derive_semantic_frame(
        "Покажи заказы",
        {
            "aggregation_hint": None,
            "entities": ["заказы"],
            "required_output": [],
        },
    )

    assert frame["requires_listing"] is False
    assert "metric_intent" in frame["ambiguities"]
