"""Регресс: после ответа пользователя на clarification CLI не должен
запустить граф ещё раз с тем же вопросом."""

from cli.interface import (
    _interpret_filter_clarification,
    _match_clarification_to_choice,
    _normalize_clarification_options,
    _parse_clarification_choice,
)


def test_match_clarification_by_column_name():
    filter_candidates = {
        "phrase:0": [
            {"column": "task_subtype", "description": "Подтип задачи"},
            {"column": "task_type", "description": "Тип задачи"},
        ]
    }
    result = _match_clarification_to_choice("task_subtype", filter_candidates)
    assert result == {"phrase:0": "task_subtype"}


def test_match_clarification_by_description_fragment():
    filter_candidates = {
        "phrase:0": [
            {"column": "task_subtype", "description": "Подтип задачи"},
            {"column": "task_type", "description": "Тип задачи"},
        ]
    }
    result = _match_clarification_to_choice("подтип задачи", filter_candidates)
    assert result == {"phrase:0": "task_subtype"}


def test_match_clarification_skips_already_resolved_request_id():
    filter_candidates = {
        "phrase:0": [
            {"column": "task_subtype", "description": "Подтип задачи"},
            {"column": "task_type", "description": "Тип задачи"},
        ],
        "phrase:1": [
            {"column": "segment_name", "description": "Сегмент"},
            {"column": "region", "description": "Регион"},
        ],
    }
    result = _match_clarification_to_choice(
        "регион",
        filter_candidates,
        already_resolved={"phrase:0": "task_subtype"},
    )
    assert result == {"phrase:1": "region"}


def test_match_clarification_returns_empty_when_no_hit():
    filter_candidates = {
        "phrase:0": [
            {"column": "task_subtype", "description": "Подтип задачи"},
            {"column": "task_type", "description": "Тип задачи"},
        ]
    }
    result = _match_clarification_to_choice("нечто непонятное", filter_candidates)
    assert result == {}


def test_interpret_confirm_accepts_yes_reply():
    where_resolution = {
        "clarification_spec": {
            "type": "confirm",
            "request_id": "text:dm.t.task_subtype",
            "options": [{"column": "task_subtype", "label": "Подтип задачи"}],
        }
    }
    accepted, rejected = _interpret_filter_clarification("Да", where_resolution)
    assert accepted == {"text:dm.t.task_subtype": "task_subtype"}
    assert rejected == {}


def test_interpret_confirm_rejects_no_reply():
    where_resolution = {
        "clarification_spec": {
            "type": "confirm",
            "request_id": "text:dm.t.task_subtype",
            "options": [{"column": "task_subtype", "label": "Подтип задачи"}],
        }
    }
    accepted, rejected = _interpret_filter_clarification("нет", where_resolution)
    assert accepted == {}
    assert rejected == {"text:dm.t.task_subtype": ["task_subtype"]}


# ---------------------------------------------------------------------------
# Iteration 3: Step K — option normalisation + numbered rendering helpers.
# ---------------------------------------------------------------------------


def test_normalize_h2_string_options():
    """H2 (catalog_grounding) emits `options: list[str]` shaped as
    "name — description". The normalizer splits value/label correctly.
    """
    spec = {"options": [
        "dm.sale_funnel_task — Воронка продаж по задачам",
        "dm.fact_outflow — Информация по фактическим оттокам",
    ]}
    normalized = _normalize_clarification_options(spec)
    assert normalized == [
        ("dm.sale_funnel_task", "dm.sale_funnel_task — Воронка продаж по задачам"),
        ("dm.fact_outflow", "dm.fact_outflow — Информация по фактическим оттокам"),
    ]


def test_normalize_h4_dict_options():
    """H4 (where_resolver fallback) emits `options: list[dict]` with
    `value`/`label` keys. The normalizer handles both shapes uniformly.
    """
    spec = {"options": [
        {"value": "dm.fact_outflow", "label": "dm.fact_outflow — Факты"},
        {"column": "task_type", "label": "task_type (Тип задачи)"},
    ]}
    normalized = _normalize_clarification_options(spec)
    assert normalized == [
        ("dm.fact_outflow", "dm.fact_outflow — Факты"),
        ("task_type", "task_type (Тип задачи)"),
    ]


def test_parse_clarification_choice_by_number():
    options = [
        ("dm.foo", "dm.foo — desc 1"),
        ("dm.bar", "dm.bar — desc 2"),
        ("dm.baz", "dm.baz — desc 3"),
    ]
    assert _parse_clarification_choice("2", options) == "dm.bar"


def test_parse_clarification_choice_by_substring():
    options = [
        ("dm.sale_funnel_task", "Воронка задач"),
        ("dm.fact_outflow", "Отток"),
    ]
    # Unique substring on the value → matched.
    assert _parse_clarification_choice("outflow", options) == "dm.fact_outflow"
    # Unique substring on the label → matched.
    assert _parse_clarification_choice("воронка", options) == "dm.sale_funnel_task"


def test_parse_clarification_choice_ambiguous_returns_none():
    options = [
        ("dm.fact_outflow", "Отток (fact)"),
        ("dm.fact_outflow_history", "Отток история"),
    ]
    # Both contain "отток" → ambiguous → None (the CLI falls back to the
    # free-form clarification flow).
    assert _parse_clarification_choice("отток", options) is None


def test_parse_clarification_choice_empty_or_unknown():
    options = [("dm.foo", "dm.foo — desc")]
    assert _parse_clarification_choice("", options) is None
    assert _parse_clarification_choice("nonexistent", options) is None
    # Out-of-range number → None.
    assert _parse_clarification_choice("99", options) is None
