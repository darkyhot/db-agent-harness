"""Регресс: после ответа пользователя на clarification CLI не должен
запустить граф ещё раз с тем же вопросом."""

from cli.interface import _match_clarification_to_choice


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
