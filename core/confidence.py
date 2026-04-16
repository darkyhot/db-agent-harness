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
    qualifier = str((semantic_frame or {}).get("qualifier") or "")
    explicit_filters = list((intent or {}).get("filter_conditions") or [])
    filter_candidates = where_resolution.get("filter_candidates", {}) or {}

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

    candidate_scores: list[float] = []
    evidence: list[str] = []
    for request_id, candidates in filter_candidates.items():
        if not candidates:
            continue
        top = candidates[0]
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
) -> dict[str, Any]:
    components = {
        "table_confidence": table_confidence or {"score": 0.25, "level": "low", "evidence": ["missing"]},
        "filter_confidence": filter_confidence or {"score": 0.95, "level": "high", "evidence": ["default"]},
        "join_confidence": join_confidence or {"score": 0.95, "level": "high", "evidence": ["default"]},
    }
    score = min(float(components["table_confidence"]["score"]), float(components["filter_confidence"]["score"]), float(components["join_confidence"]["score"]))
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
