import pandas as pd

from core.filter_ranking import rank_filter_candidates
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
        "schema_name": ["dm"] * 7,
        "table_name": ["sale_funnel"] * 7,
        "column_name": ["report_dt", "task_code", "task_subtype", "is_outflow", "segment_name", "task_category", "task_type"],
        "dType": ["date", "text", "text", "int4", "text", "text", "text"],
        "description": [
            "Отчетная дата",
            "Код задачи",
            "Подтип задачи",
            "Признак подтверждения оттока",
            "Сегмент клиента",
            "Категория задачи",
            "Тип задачи",
        ],
        "is_primary_key": [False, False, False, False, False, False, False],
        "unique_perc": [0.5, 90.0, 10.0, 2.0, 5.0, 0.02, 0.11],
        "not_null_perc": [99.0, 100.0, 100.0, 100.0, 99.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    loader.ensure_value_profiles()
    return loader


def test_filter_ranking_finds_confirmed_outflow_flag(tmp_path):
    loader = _loader(tmp_path)
    frame = derive_semantic_frame(
        "Посчитай количество задач с подтвержденным оттоком",
        schema_loader=loader,
    )
    ranked = rank_filter_candidates(
        user_input="Посчитай количество задач с подтвержденным оттоком",
        intent={"filter_conditions": []},
        selected_tables=["dm.sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
    )

    top = next(iter(ranked.values()))[0]
    assert top["column"] == "is_outflow"
    assert top["condition"] == "is_outflow = 1"
    assert top["confidence"] in {"high", "medium"}


def test_filter_ranking_supports_generic_explicit_filter(tmp_path):
    loader = _loader(tmp_path)
    frame = derive_semantic_frame("Посчитай количество задач по сегменту retail", schema_loader=loader)
    ranked = rank_filter_candidates(
        user_input="Посчитай количество задач по сегменту retail",
        intent={"filter_conditions": [{"column_hint": "сегмент", "operator": "=", "value": "retail"}]},
        selected_tables=["dm.sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
    )

    top = ranked["explicit:0"][0]
    assert top["column"] == "segment_name"
    assert "retail" in top["condition"].lower()


def test_filter_ranking_prefers_task_subtype_over_dense_task_fields(tmp_path):
    loader = _loader(tmp_path)
    frame = derive_semantic_frame(
        "Посчитай количество задач по фактическому оттоку",
        schema_loader=loader,
    )
    ranked = rank_filter_candidates(
        user_input="Посчитай количество задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_tables=["dm.sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
    )

    top = next(iter(ranked.values()))[0]
    assert top["column"] == "task_subtype"


def test_filter_ranking_prefers_explicit_column_reference(tmp_path):
    loader = _loader(tmp_path)
    frame = derive_semantic_frame(
        "Посчитай количество задач по фактическому оттоку",
        schema_loader=loader,
    )
    ranked = rank_filter_candidates(
        user_input="Посчитай количество задач по фактическому оттоку Уточнение пользователя: task_subtype",
        intent={"filter_conditions": []},
        selected_tables=["dm.sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
    )

    top = next(iter(ranked.values()))[0]
    assert top["column"] == "task_subtype"
    assert top["confidence"] == "high"


def test_filter_ranking_matches_known_terms_semantically(tmp_path):
    loader = _loader(tmp_path)
    loader._value_profiles = {
        "dm.sale_funnel.task_subtype": {
            "known_terms": ["фактический отток"],
            "top_values": [],
            "value_mode": "enum_like",
        }
    }
    frame = derive_semantic_frame(
        "Посчитай количество задач по фактическому оттоку",
        schema_loader=loader,
    )
    ranked = rank_filter_candidates(
        user_input="Посчитай количество задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_tables=["dm.sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
    )

    top = next(iter(ranked.values()))[0]
    assert top["column"] == "task_subtype"
    assert any("value_match=фактический отток" == ev for ev in top["evidence"])


def test_filter_ranking_phrase_bonus_breaks_tie_between_similar_columns(tmp_path):
    """Полная фраза «фактический отток» должна склонить выбор к task_subtype
    даже рядом с очень похожим task_type, у которого в known_terms только
    «отток» (одно слово)."""
    loader = _loader(tmp_path)
    loader._value_profiles = {
        "dm.sale_funnel.task_subtype": {
            "known_terms": ["фактический отток"],
            "top_values": [],
            "value_mode": "enum_like",
        },
        "dm.sale_funnel.task_type": {
            "known_terms": ["отток"],
            "top_values": [],
            "value_mode": "enum_like",
        },
    }
    frame = derive_semantic_frame(
        "Посчитай количество задач по фактическому оттоку",
        schema_loader=loader,
    )
    ranked = rank_filter_candidates(
        user_input="Посчитай количество задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_tables=["dm.sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
    )
    candidates = next(iter(ranked.values()))
    columns = [c["column"] for c in candidates]
    assert columns.index("task_subtype") < columns.index("task_type"), columns
    top = candidates[0]
    assert top["column"] == "task_subtype"
    assert top["matched_example"] == "фактический отток"
    # score gap достаточно большой, чтобы where_resolver не ушёл в ambiguity.
    subtype_score = next(c["score"] for c in candidates if c["column"] == "task_subtype")
    type_score = next(c["score"] for c in candidates if c["column"] == "task_type")
    assert subtype_score - type_score >= 40.0


def test_filter_ranking_records_example_values_for_fallback_label(tmp_path):
    loader = _loader(tmp_path)
    loader._value_profiles = {
        "dm.sale_funnel.task_subtype": {
            "known_terms": ["фактический отток", "новый"],
            "top_values": [],
            "value_mode": "enum_like",
        },
    }
    frame = derive_semantic_frame(
        "Посчитай количество задач по фактическому оттоку",
        schema_loader=loader,
    )
    ranked = rank_filter_candidates(
        user_input="Посчитай количество задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_tables=["dm.sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
    )
    top = next(c for c in next(iter(ranked.values())) if c["column"] == "task_subtype")
    assert top["example_values"][:2] == ["фактический отток", "новый"]
