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
