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
    planning = build_planning_confidence(
        table_confidence={"score": 0.9, "level": "high", "evidence": []},
        filter_confidence={"score": 0.52, "level": "medium", "evidence": []},
        join_confidence={"score": 0.86, "level": "high", "evidence": []},
    )
    assert planning["level"] == "medium"
    assert planning["action"] == "clarify"


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
