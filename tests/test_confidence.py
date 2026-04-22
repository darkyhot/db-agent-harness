from core.confidence import (
    build_fallback_policy,
    build_planning_confidence,
    evaluate_filter_confidence,
    evaluate_join_confidence,
    evaluate_table_confidence,
)


def test_table_confidence_high_for_strong_candidates():
    result = evaluate_table_confidence({"dm.sale_funnel": 90, "dm.dim": 80})
    assert result["level"] == "high"


def test_filter_confidence_low_when_clarification_needed():
    result = evaluate_filter_confidence(
        {"needs_clarification": True, "filter_candidates": {}},
        semantic_frame={"qualifier": "factual_outflow"},
        intent={"filter_conditions": []},
    )
    assert result["level"] == "low"


def test_planning_confidence_uses_worst_component():
    # Weighted formula: 0.4*0.9 + 0.3*0.52 + 0.3*0.86 = 0.36 + 0.156 + 0.258 = 0.774 → high
    # Поведение изменилось: взвешенная сумма, а не min — medium filter больше не блокирует
    planning = build_planning_confidence(
        table_confidence={"score": 0.9, "level": "high", "evidence": []},
        filter_confidence={"score": 0.52, "level": "medium", "evidence": []},
        join_confidence={"score": 0.86, "level": "high", "evidence": []},
    )
    assert planning["level"] == "high"
    assert planning["action"] == "execute"


def test_fallback_policy_allows_llm_only_for_high_confidence():
    planning = {
        "score": 0.88,
        "level": "high",
        "action": "execute",
    }
    policy = build_fallback_policy(
        planning_confidence=planning,
        deterministic_sql_valid=False,
        has_template_sql=True,
    )
    assert policy["allow_llm_fallback"] is True
    assert policy["action"] == "llm_fallback"


def test_fallback_policy_blocks_medium_confidence():
    planning = {
        "score": 0.55,
        "level": "medium",
        "action": "clarify",
    }
    policy = build_fallback_policy(
        planning_confidence=planning,
        deterministic_sql_valid=False,
        has_template_sql=True,
    )
    assert policy["allow_llm_fallback"] is False
    assert policy["action"] == "clarify"


def test_filter_confidence_treats_user_choices_as_resolved():
    """Когда пользователь явно ответил на clarification (user_filter_choices),
    score соответствующего request_id должен подниматься до 1.0 — иначе
    planning_confidence остаётся «medium» и пайплайн зацикливается на
    повторных уточнениях, даже если выбор уже сделан."""
    where_resolution = {
        "needs_clarification": False,
        "filter_candidates": {
            "phrase:0": [
                {"column": "task_subtype", "score": 50.0, "confidence": "medium"},
                {"column": "task_type", "score": 42.0, "confidence": "low"},
            ],
            "text:dm.t.task_type": [
                {"column": "task_subtype", "score": 50.0, "confidence": "medium"},
                {"column": "task_type", "score": 50.0, "confidence": "medium"},
            ],
        },
        "user_filter_choices": {
            "phrase:0": "task_subtype",
            "text:dm.t.task_type": "task_subtype",
        },
    }
    result = evaluate_filter_confidence(
        where_resolution,
        semantic_frame={"qualifier": "factual_outflow"},
        intent={"filter_conditions": []},
    )
    assert result["level"] == "high"
    assert result["score"] >= 0.95
    assert all("user_choice" in ev for ev in result["evidence"])


# ---------------------------------------------------------------------------
# Task 1.3: Weighted confidence with hint-boost
# ---------------------------------------------------------------------------

def test_hint_boost_must_keep_tables_elevates_low_table_score():
    """must_keep_tables → table_score boosted to 0.9 → overall high."""
    planning = build_planning_confidence(
        table_confidence={"score": 0.5, "level": "medium", "evidence": []},
        filter_confidence={"score": 0.9, "level": "high", "evidence": []},
        join_confidence={"score": 0.9, "level": "high", "evidence": []},
        user_hints={"must_keep_tables": [("dm", "fact_churn")]},
    )
    # 0.4*0.9 + 0.3*0.9 + 0.3*0.9 = 0.9 → high
    assert planning["level"] == "high"


def test_hint_boost_group_by_hints_elevates_filter_score():
    """group_by_hints → filter_score boosted to 0.8."""
    planning = build_planning_confidence(
        table_confidence={"score": 0.9, "level": "high", "evidence": []},
        filter_confidence={"score": 0.3, "level": "low", "evidence": []},
        join_confidence={"score": 0.9, "level": "high", "evidence": []},
        user_hints={"group_by_hints": ["task_code"]},
    )
    # filter_score = max(0.3, 0.8) = 0.8
    # 0.4*0.9 + 0.3*0.8 + 0.3*0.9 = 0.36 + 0.24 + 0.27 = 0.87 → high
    assert planning["level"] == "high"


def test_no_hints_low_table_score_remains_low():
    """Без hints, table=0.3 и все компоненты низкие → level == 'low'."""
    planning = build_planning_confidence(
        table_confidence={"score": 0.3, "level": "low", "evidence": []},
        filter_confidence={"score": 0.3, "level": "low", "evidence": []},
        join_confidence={"score": 0.3, "level": "low", "evidence": []},
    )
    assert planning["level"] == "low"
    assert planning["action"] == "stop"


def test_all_high_scores_remain_high():
    """Все компоненты высокие → high."""
    planning = build_planning_confidence(
        table_confidence={"score": 0.9, "level": "high", "evidence": []},
        filter_confidence={"score": 0.9, "level": "high", "evidence": []},
        join_confidence={"score": 0.9, "level": "high", "evidence": []},
    )
    assert planning["level"] == "high"


def test_filter_confidence_treats_semantic_exact_match_as_high():
    where_resolution = {
        "needs_clarification": False,
        "applied_rules": [],
        "filter_candidates": {
            "text:dm.t.task_subtype": [
                {
                    "column": "task_subtype",
                    "score": 61.0,
                    "confidence": "medium",
                    "matched_example": "фактический отток",
                    "evidence": ["known_term_phrase=фактический отток"],
                },
            ],
        },
    }
    result = evaluate_filter_confidence(
        where_resolution,
        semantic_frame={"qualifier": "factual_outflow"},
        intent={"filter_conditions": []},
    )
    assert result["level"] == "high"
    assert result["score"] >= 0.95
    assert any("semantic_exact" in ev for ev in result["evidence"])


def test_filter_confidence_treats_table_context_business_event_as_high():
    where_resolution = {
        "needs_clarification": False,
        "applied_rules": [],
        "reasoning": ["table_context_covers_business_event"],
        "filter_candidates": {
            "text:dm.sale_funnel.task_subtype": [
                {
                    "column": "task_subtype",
                    "score": 61.0,
                    "confidence": "medium",
                },
                {
                    "column": "task_category",
                    "score": 60.0,
                    "confidence": "medium",
                },
            ],
        },
    }
    result = evaluate_filter_confidence(
        where_resolution,
        semantic_frame={"qualifier": "factual_outflow"},
        intent={"filter_conditions": []},
    )
    assert result["level"] == "high"
    assert result["score"] >= 0.95
    assert result["evidence"] == ["table_context_covers_business_event"]
