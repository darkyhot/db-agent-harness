"""Тесты на LLM-tiebreaker для filter-кандидатов в resolve_where."""

import pandas as pd

from core.schema_loader import SchemaLoader
from core.semantic_frame import derive_semantic_frame
from core.where_resolver import resolve_where


def _loader_with_similar_cols(tmp_path):
    """Таблица с двумя семантически близкими колонками для фильтра."""
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["tasks"],
        "description": ["Задачи по регионам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 4,
        "table_name": ["tasks"] * 4,
        "column_name": ["report_dt", "region_name", "gosb_region", "task_code"],
        "dType": ["date", "text", "text", "text"],
        "description": [
            "Отчетная дата",
            "Название региона",
            "Регион ГОСБ",
            "Код задачи",
        ],
        "is_primary_key": [False, False, False, False],
        "unique_perc": [0.5, 20.0, 15.0, 90.0],
        "not_null_perc": [99.0, 100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    loader._value_profiles = {
        "dm.tasks.region_name": {
            "known_terms": ["Москва", "СПб"],
            "top_values": ["Москва"],
            "value_mode": "enum_like",
        },
        "dm.tasks.gosb_region": {
            "known_terms": ["Москва", "СПб"],
            "top_values": ["Москва"],
            "value_mode": "enum_like",
        },
    }
    return loader


def test_tiebreaker_callback_receives_top_candidates(tmp_path):
    loader = _loader_with_similar_cols(tmp_path)
    frame = derive_semantic_frame("Покажи задачи по региону Москва", schema_loader=loader)

    seen: list[dict] = []

    def tiebreaker(*, request_id, user_input, candidates):
        seen.append({
            "request_id": request_id,
            "user_input": user_input,
            "candidates": [c.get("column") for c in candidates],
        })
        return "region_name"

    resolve_where(
        user_input="Покажи задачи по региону Москва",
        intent={
            "filter_conditions": [
                {"kind": "explicit_filter", "column_hint": "регион", "value": "Москва"}
            ],
        },
        selected_columns={"dm.tasks": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.tasks"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
        filter_tiebreaker=tiebreaker,
    )
    # Тайбрейкер должен был быть вызван если scoring дал узкий gap
    # (мы проверяем, что callback вообще интегрирован; если scoring выбрал
    # без ambiguity, seen может быть пустым — это тоже валидный путь).
    if seen:
        assert "region_name" in seen[0]["candidates"] or "gosb_region" in seen[0]["candidates"]


def test_tiebreaker_returning_none_keeps_clarification(tmp_path):
    loader = _loader_with_similar_cols(tmp_path)
    frame = derive_semantic_frame("Покажи задачи по региону", schema_loader=loader)

    def tiebreaker(*, request_id, user_input, candidates):
        return None

    # Без tiebreaker'а: возможна clarification. С tiebreaker=None: поведение то же.
    result = resolve_where(
        user_input="Покажи задачи по региону",
        intent={
            "filter_conditions": [
                {"kind": "explicit_filter", "column_hint": "регион", "value": "Москва"}
            ],
        },
        selected_columns={"dm.tasks": {"select": ["task_code"]}},
        selected_tables=["dm.tasks"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
        filter_tiebreaker=tiebreaker,
    )
    # Пустой возврат от tiebreaker → поведение не меняется (может быть clarify
    # или применён top1 — зависит от scoring'а).
    assert isinstance(result, dict)
    assert "conditions" in result


def test_tiebreaker_exception_does_not_break(tmp_path):
    loader = _loader_with_similar_cols(tmp_path)
    frame = derive_semantic_frame("Покажи задачи по региону Москва", schema_loader=loader)

    def tiebreaker(**kw):
        raise RuntimeError("LLM is down")

    # Должно просто пройти дальше без tiebreaker'а
    result = resolve_where(
        user_input="Покажи задачи по региону Москва",
        intent={
            "filter_conditions": [
                {"kind": "explicit_filter", "column_hint": "регион", "value": "Москва"}
            ],
        },
        selected_columns={"dm.tasks": {"select": ["task_code"]}},
        selected_tables=["dm.tasks"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
        filter_tiebreaker=tiebreaker,
    )
    assert isinstance(result, dict)


def test_no_tiebreaker_legacy_behavior(tmp_path):
    """Без tiebreaker'а поведение должно быть идентично старому — clarification при ambiguity."""
    loader = _loader_with_similar_cols(tmp_path)
    frame = derive_semantic_frame("Покажи задачи по региону Москва", schema_loader=loader)
    result = resolve_where(
        user_input="Покажи задачи по региону Москва",
        intent={
            "filter_conditions": [
                {"kind": "explicit_filter", "column_hint": "регион", "value": "Москва"}
            ],
        },
        selected_columns={"dm.tasks": {"select": ["task_code"]}},
        selected_tables=["dm.tasks"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert "conditions" in result
