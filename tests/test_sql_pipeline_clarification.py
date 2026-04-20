from graph.nodes.sql_pipeline import _build_specific_clarification


def test_build_specific_clarification_prefers_candidate_details():
    where_resolution = {
        "filter_candidates": {
            "phrase:0": [
                {"column": "task_category", "description": "Категория задачи"},
                {"column": "task_type", "description": "Тип задачи"},
            ]
        }
    }

    msg = _build_specific_clarification(where_resolution)

    assert "task_category" in msg
    assert "Категория задачи" in msg
    assert "task_type" in msg
    assert "Тип задачи" in msg
    assert "Нужна короткая уточняющая деталь" not in msg


def test_build_specific_clarification_returns_empty_when_all_resolved():
    """Если все request_id закрыты через user_filter_choices — задавать
    нечего, возвращаем пустую строку. sql_planner интерпретирует это как
    «уточнение не требуется» и идёт к выполнению. Регресс на зависание
    второго круга clarification после ответа пользователя."""
    where_resolution = {
        "filter_candidates": {
            "phrase:0": [
                {"column": "task_subtype", "description": "Подтип задачи"},
                {"column": "task_type", "description": "Тип задачи"},
            ],
            "text:dm.t.task_type": [
                {"column": "task_subtype", "description": "Подтип задачи"},
                {"column": "task_category", "description": "Категория задачи"},
            ],
        },
        "user_filter_choices": {
            "phrase:0": "task_subtype",
            "text:dm.t.task_type": "task_subtype",
        },
    }
    msg = _build_specific_clarification(where_resolution)
    assert msg == ""


def test_build_specific_clarification_skips_resolved_and_asks_remaining():
    """Закрытые request_id пропускаются, но если остался незакрытый —
    спрашиваем именно о нём (а не возвращаем пустоту)."""
    where_resolution = {
        "filter_candidates": {
            "phrase:0": [
                {"column": "task_subtype", "description": "Подтип задачи"},
            ],
            "text:dm.t.segment": [
                {"column": "segment_name", "description": "Сегмент клиента"},
                {"column": "region", "description": "Регион"},
            ],
        },
        "user_filter_choices": {"phrase:0": "task_subtype"},
    }
    msg = _build_specific_clarification(where_resolution)
    assert "segment_name" in msg
    assert "region" in msg
    assert "task_subtype" not in msg
