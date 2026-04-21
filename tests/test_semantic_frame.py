import pandas as pd

from core.schema_loader import SchemaLoader
from core.semantic_frame import (
    _extract_freeform_phrases,
    derive_semantic_frame,
    sanitize_user_input_for_semantics,
)


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


def test_extract_freeform_phrases_stops_before_date_tail():
    phrases = _extract_freeform_phrases("Покажи сумму оттока по сегментам за январь 2024")
    assert "сегментам за январь 2024" not in phrases


def test_output_dimension_not_turned_into_filter_intent(tmp_path):
    loader = _loader(tmp_path)
    frame = derive_semantic_frame(
        "Покажи сумму оттока по сегментам за январь 2024",
        {
            "aggregation_hint": "sum",
            "entities": ["отток", "сегмент"],
            "required_output": ["сегмент"],
        },
        schema_loader=loader,
    )
    assert "сегмент" in frame["output_dimensions"]
    assert not any("segment" in str(item.get("column_key", "")) for item in frame["filter_intents"])


def test_single_word_grouping_phrase_not_turned_into_filter_intent(tmp_path):
    loader = _loader(tmp_path)
    frame = derive_semantic_frame(
        "Покажи сумму оттока по сегментам за январь 2024",
        {
            "aggregation_hint": "sum",
            "entities": ["отток"],
            "required_output": [],
        },
        schema_loader=loader,
    )
    assert not frame["filter_intents"]


def test_explicit_projection_column_not_turned_into_filter_intent(tmp_path):
    loader = _loader(tmp_path)
    frame = derive_semantic_frame(
        "Покажи количество задач по task_subtype",
        {
            "aggregation_hint": "count",
            "entities": ["task_subtype"],
            "required_output": [],
        },
        schema_loader=loader,
    )
    assert not any("task_subtype" in str(item.get("column_key", "")) for item in frame["filter_intents"])


def test_explicit_column_after_po_becomes_output_dimension_not_filter(tmp_path):
    loader = _loader(tmp_path)
    frame = derive_semantic_frame(
        "Покажи количество задач по task_subtype за февраль 2026",
        {
            "aggregation_hint": "count",
            "entities": ["задачи"],
            "required_output": [],
        },
        schema_loader=loader,
    )
    assert "task_subtype" in frame["output_dimensions"]
    assert frame["qualifier"] is None
    assert not frame["filter_intents"]


def test_value_candidate_after_po_stays_filter_not_output_dimension(tmp_path):
    loader = _loader(tmp_path)
    loader._rule_registry = {
        "rules": [
            {
                "rule_id": "text:dm.sale_funnel.task_subtype",
                "column_key": "dm.sale_funnel.task_subtype",
                "semantic_class": "enum_like",
                "match_kind": "text_search",
                "match_phrases": ["подтип задачи"],
                "value_candidates": ["Фактический отток"],
            }
        ]
    }
    frame = derive_semantic_frame(
        "Сколько задач по фактическому оттоку за февраль 2026",
        {
            "aggregation_hint": "count",
            "entities": ["задачи", "отток"],
            "required_output": [],
        },
        schema_loader=loader,
    )
    assert frame["requires_single_entity_count"] is True
    assert any(
        item.get("column_key") == "dm.sale_funnel.task_subtype"
        for item in frame["filter_intents"]
    )


def test_projection_phrase_does_not_bleed_into_freeform_filter():
    phrases = _extract_freeform_phrases(
        "Покажи сумму outflow_qty по region_name, подтяни region_name из uzp_dim_gosb по gosb_id"
    )
    assert "region_name подтяни region_name из" not in phrases


def test_sanitize_user_input_removes_table_choice_suffix():
    cleaned = sanitize_user_input_for_semantics(
        "Сколько задач по фактическому оттоку поставили в феврале 26 "
        "(использовать таблицу dm.uzp_data_split_mzp_sale_funnel)"
    )
    assert "использовать таблицу" not in cleaned
    assert "Сколько задач по фактическому оттоку поставили в феврале 26" in cleaned


def test_semantic_frame_ignores_table_choice_suffix_and_value_after_po(tmp_path):
    loader = _loader(tmp_path)
    loader._rule_registry = {
        "rules": [
            {
                "rule_id": "text:dm.sale_funnel.task_subtype",
                "column_key": "dm.sale_funnel.task_subtype",
                "semantic_class": "enum_like",
                "match_kind": "text_search",
                "match_phrases": ["подтип задачи"],
                "value_candidates": ["Фактический отток"],
            },
            {
                "rule_id": "text:dm.sale_funnel.task_category",
                "column_key": "dm.sale_funnel.task_category",
                "semantic_class": "enum_like",
                "match_kind": "text_search",
                "match_phrases": ["категория задачи"],
                "value_candidates": ["Задача"],
            },
        ]
    }
    loader._semantic_lexicon = {
        "subjects": {
            "task": {"aliases": ["задача", "задачи"]},
            "snapshot": {"aliases": ["срез", "отчет"]},
        },
        "dimensions": {
            "фактическая дата закрытия задачи": {
                "aliases": ["фактическая дата закрытия задачи"],
            },
            "тип_задачи": {
                "aliases": ["тип задачи"],
            },
        },
    }
    frame = derive_semantic_frame(
        "Сколько задач по фактическому оттоку поставили в феврале 26 "
        "(использовать таблицу dm.uzp_data_split_mzp_sale_funnel)",
        {
            "aggregation_hint": "count",
            "entities": ["задачи", "отток"],
            "required_output": [],
        },
        schema_loader=loader,
    )
    assert frame["subject"] == "task"
    assert frame["requested_grain"] == "task"
    assert frame["requires_single_entity_count"] is True
    assert "фактическая дата закрытия задачи" not in frame["output_dimensions"]
    assert "тип_задачи" not in frame["output_dimensions"]
