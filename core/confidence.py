"""Confidence model and controlled fallback policy for planning/execution."""

from __future__ import annotations

from typing import Any


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _level(score: float) -> str:
    score = _clamp(score)
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def evaluate_table_confidence(
    table_confidences: dict[str, int] | None,
    *,
    disambiguation_options: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    values = [max(0, min(100, int(v))) for v in (table_confidences or {}).values()]
    if not values:
        score = 0.25
        evidence = ["no_table_confidences"]
    else:
        score = sum(values) / (len(values) * 100.0)
        evidence = [f"avg_table_confidence={round(score, 3)}"]
    if disambiguation_options:
        score = min(score, 0.58)
        evidence.append("disambiguation_pending")
    return {
        "score": round(_clamp(score), 3),
        "level": _level(score),
        "evidence": evidence,
    }


def evaluate_filter_confidence(
    where_resolution: dict[str, Any] | None,
    *,
    semantic_frame: dict[str, Any] | None = None,
    intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    where_resolution = where_resolution or {}
    reasoning = [str(item) for item in (where_resolution.get("reasoning", []) or [])]
    qualifier = str((semantic_frame or {}).get("qualifier") or "")
    explicit_filters = list((intent or {}).get("filter_conditions") or [])
    filter_candidates = where_resolution.get("filter_candidates", {}) or {}
    user_choices = where_resolution.get("user_filter_choices", {}) or {}
    applied_rules = {str(v) for v in (where_resolution.get("applied_rules", []) or [])}

    if where_resolution.get("needs_clarification"):
        return {
            "score": 0.35,
            "level": "low",
            "evidence": ["where_needs_clarification"],
        }

    if not qualifier and not explicit_filters:
        return {
            "score": 0.95,
            "level": "high",
            "evidence": ["no_extra_filters_requested"],
        }

    if "table_context_covers_business_event" in reasoning:
        return {
            "score": 0.95,
            "level": "high",
            "evidence": ["table_context_covers_business_event"],
        }

    candidate_scores: list[float] = []
    evidence: list[str] = []
    for request_id, candidates in filter_candidates.items():
        if not candidates:
            continue
        top = candidates[0]
        if str(request_id) in applied_rules:
            candidate_scores.append(1.0)
            evidence.append(f"{request_id}:{top.get('column')}:applied_rule")
            continue
        # Если пользователь явно выбрал колонку для этого request_id —
        # считаем фильтр полностью разрешённым (score=1.0). Без этого
        # candidate-scores остаются на уровне medium и planning_confidence
        # продолжает требовать clarification, хотя выбор уже сделан.
        if str(request_id) in user_choices:
            chosen_column = str(user_choices[str(request_id)])
            candidate_scores.append(1.0)
            evidence.append(f"{request_id}:{chosen_column}:user_choice")
            continue
        top_evidence = {str(ev) for ev in (top.get("evidence") or [])}
        second = candidates[1] if len(candidates) > 1 else None
        if (
            second is None
            and (
                top.get("matched_example")
                or any(
                    ev.startswith("known_term_phrase=") or ev.startswith("value_match=")
                    for ev in top_evidence
                )
            )
        ):
            candidate_scores.append(1.0)
            evidence.append(f"{request_id}:{top.get('column')}:semantic_exact")
            continue
        if applied_rules and str(top.get("confidence") or "").lower() == "low":
            evidence.append(f"{request_id}:{top.get('column')}:ignored_low_confidence")
            continue
        top_score = float(top.get("score", 0.0) or 0.0) / 100.0
        candidate_scores.append(top_score)
        evidence.append(f"{request_id}:{top.get('column')}:{top.get('confidence')}")

    if candidate_scores:
        score = sum(candidate_scores) / len(candidate_scores)
        return {
            "score": round(_clamp(score), 3),
            "level": _level(score),
            "evidence": evidence,
        }

    return {
        "score": 0.3,
        "level": "low",
        "evidence": ["required_filters_without_candidates"],
    }


def evaluate_join_confidence(join_decision: dict[str, Any] | None) -> dict[str, Any]:
    join_decision = join_decision or {}
    selected_tables = list(join_decision.get("selected_tables") or [])
    risk_level = str(join_decision.get("risk_level") or "low")
    reason = str(join_decision.get("reason") or "unknown")

    if len(selected_tables) <= 1:
        score = 0.96
    elif risk_level == "low":
        score = 0.85
    elif risk_level == "medium":
        score = 0.62
    else:
        score = 0.35
    return {
        "score": round(_clamp(score), 3),
        "level": _level(score),
        "evidence": [f"risk={risk_level}", f"reason={reason}"],
    }


def build_planning_confidence(
    *,
    table_confidence: dict[str, Any] | None,
    filter_confidence: dict[str, Any] | None,
    join_confidence: dict[str, Any] | None,
    user_hints: dict[str, Any] | None = None,
    explicit_mode: bool = False,
) -> dict[str, Any]:
    components = {
        "table_confidence": table_confidence or {"score": 0.25, "level": "low", "evidence": ["missing"]},
        "filter_confidence": filter_confidence or {"score": 0.95, "level": "high", "evidence": ["default"]},
        "join_confidence": join_confidence or {"score": 0.95, "level": "high", "evidence": ["default"]},
    }
    table_score = float(components["table_confidence"]["score"])
    filter_score = float(components["filter_confidence"]["score"])
    join_score = float(components["join_confidence"]["score"])

    # Hint-boost: явные хинты пользователя повышают соответствующие компоненты.
    # В explicit_mode (≥2 явных хинта) применяется строже: max(score, 0.95) вместо 0.9/0.8.
    hints = user_hints or {}
    _boost_table = 0.95 if explicit_mode else 0.9
    _boost_filter = 0.95 if explicit_mode else 0.8
    _boost_join = 0.95 if explicit_mode else 0.9

    if hints.get("must_keep_tables"):
        table_score = max(table_score, _boost_table)
    if hints.get("group_by_hints") or hints.get("aggregate_hints"):
        filter_score = max(filter_score, _boost_filter)
    if hints.get("join_fields"):
        join_score = max(join_score, _boost_join)

    # Взвешенная сумма (table доминирует, т.к. выбор таблиц — критический шаг)
    score = 0.4 * table_score + 0.3 * filter_score + 0.3 * join_score

    level = _level(score)
    action = "execute" if level == "high" else ("clarify" if level == "medium" else "stop")
    return {
        "score": round(_clamp(score), 3),
        "level": level,
        "action": action,
        "components": components,
    }


def build_fallback_policy(
    *,
    planning_confidence: dict[str, Any] | None,
    deterministic_sql_valid: bool,
    has_template_sql: bool,
) -> dict[str, Any]:
    planning_confidence = planning_confidence or {}
    level = str(planning_confidence.get("level") or "low")
    action = str(planning_confidence.get("action") or ("execute" if level == "high" else "clarify"))

    if deterministic_sql_valid:
        return {
            "allow_llm_fallback": False,
            "action": "use_deterministic",
            "reason": "deterministic_sql_valid",
            "message": "",
        }

    if not has_template_sql:
        return {
            "allow_llm_fallback": level == "high",
            "action": "llm_fallback" if level == "high" else action,
            "reason": "no_deterministic_template",
            "message": (
                "" if level == "high"
                else "Не хватает уверенности, чтобы безопасно строить SQL через LLM без детерминированной основы."
            ),
        }

    if level == "high":
        return {
            "allow_llm_fallback": True,
            "action": "llm_fallback",
            "reason": "deterministic_failed_but_plan_high_confidence",
            "message": "",
        }

    if level == "medium":
        return {
            "allow_llm_fallback": False,
            "action": "clarify",
            "reason": "medium_confidence_requires_clarification",
            "message": "Есть несколько правдоподобных интерпретаций запроса. Нужна короткая уточняющая деталь перед генерацией SQL.",
        }

    return {
        "allow_llm_fallback": False,
        "action": "stop",
        "reason": "low_confidence_blocks_llm_fallback",
        "message": "Недостаточно уверенности в выборе источника или фильтров, поэтому генерация SQL остановлена до уточнения.",
    }
