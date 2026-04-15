"""Тесты для core/column_selector_deterministic.py."""

from core.column_selector_deterministic import _derive_requested_slots


def test_extracts_multiple_dimensions_with_russian_separators():
    intent = {"aggregation_hint": "sum", "entities": []}
    result = _derive_requested_slots(
        "Покажи продажи по дате, сегменту и региону; в разбивке по каналу",
        intent,
    )

    assert "date" in result["dimensions"]
    assert "segment_name" in result["dimensions"]
    assert "region_name" in result["dimensions"]
    assert "channel_name" in result["dimensions"]


def test_join_key_hints_support_cyrillic_tokens():
    intent = {"aggregation_hint": "list", "entities": []}
    result = _derive_requested_slots(
        "Сопоставь данные через ИНН и по ключу КПП",
        intent,
    )

    assert "инн" in result["join_key_hints"]
    assert "inn" in result["join_key_hints"]
    assert "кпп" in result["join_key_hints"]
    assert "kpp" in result["join_key_hints"]


def test_normalizes_morphology_for_base_dimensions():
    intent = {"aggregation_hint": "count", "entities": []}
    result = _derive_requested_slots(
        "Сколько клиентов по дате и сегменту",
        intent,
    )

    assert "date" in result["dimensions"]
    assert "segment_name" in result["dimensions"]
