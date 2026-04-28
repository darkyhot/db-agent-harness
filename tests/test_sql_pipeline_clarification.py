from graph.nodes.sql_pipeline import (
    _build_specific_clarification,
    _build_specific_clarification_spec,
    _collect_required_tables,
)


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


def test_build_specific_clarification_spec_builds_confirm_for_single_candidate():
    where_resolution = {
        "filter_candidates": {
            "text:dm.t.segment": [
                {"column": "segment_name", "description": "Сегмент клиента", "evidence": []},
            ],
        }
    }
    spec = _build_specific_clarification_spec(where_resolution)
    assert spec["type"] == "confirm"
    assert spec["request_id"] == "text:dm.t.segment"
    assert spec["options"][0]["column"] == "segment_name"


def test_build_specific_clarification_spec_skips_semantic_exact_confirm():
    where_resolution = {
        "filter_candidates": {
            "text:dm.t.task_subtype": [
                {
                    "column": "task_subtype",
                    "description": "Подтип задачи",
                    "matched_example": "фактический отток",
                    "evidence": ["known_term_phrase=фактический отток"],
                },
            ],
        }
    }
    spec = _build_specific_clarification_spec(where_resolution)
    assert spec == {}


def test_build_specific_clarification_skips_when_table_context_covers_business_event():
    where_resolution = {
        "reasoning": ["table_context_covers_business_event"],
        "filter_candidates": {
            "text:dm.sale_funnel.task_subtype": [
                {"column": "task_subtype", "description": "Подтип задачи"},
                {"column": "task_type", "description": "Тип задачи"},
            ]
        },
    }
    msg = _build_specific_clarification(where_resolution)
    assert msg == ""


def test_build_specific_clarification_spec_skips_when_table_context_covers_business_event():
    where_resolution = {
        "reasoning": ["table_context_covers_business_event"],
        "filter_candidates": {
            "text:dm.sale_funnel.task_subtype": [
                {"column": "task_subtype", "description": "Подтип задачи"},
                {"column": "task_type", "description": "Тип задачи"},
            ]
        },
    }
    spec = _build_specific_clarification_spec(where_resolution)
    assert spec == {}


def test_join_analysis_table_is_not_required_without_blueprint_or_query_need():
    required = _collect_required_tables(
        join_spec=[],
        blueprint={
            "strategy": "simple_select",
            "main_table": "dm.gosb_dim",
            "aggregations": [
                {"function": "COUNT", "column": "tb_id", "source_table": "dm.gosb_dim"}
            ],
        },
        query_spec={
            "task": "answer_data",
            "metrics": [{"operation": "count", "target": "tb_id"}],
            "source_constraints": [],
        },
    )

    assert required == {"dm.gosb_dim"}
    assert "dm.unused_join_neighbor" not in required


def test_required_tables_include_join_spec_and_required_source_constraints():
    required = _collect_required_tables(
        join_spec=[
            {"left": "dm.fact.client_id", "right": "dm.clients.client_id"},
        ],
        blueprint={"strategy": "fact_dim_join", "main_table": "dm.fact"},
        query_spec={
            "source_constraints": [
                {"schema": "dm", "table": "calendar", "required": True},
            ],
            "dimensions": [{"target": "region", "source_table": "dm.clients"}],
        },
    )

    assert {"dm.fact", "dm.clients", "dm.calendar"} <= required
