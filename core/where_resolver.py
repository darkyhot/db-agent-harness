"""WHERE resolver using ranked metadata-driven filter candidates."""

from __future__ import annotations

import logging
from typing import Any

from core.domain_rules import table_can_satisfy_frame
from core.filter_ranking import rank_filter_candidates

logger = logging.getLogger(__name__)


def _add_unique(conditions: list[str], condition: str) -> None:
    if condition not in conditions:
        conditions.append(condition)


def resolve_where(
    *,
    user_input: str,
    intent: dict[str, Any],
    selected_columns: dict[str, dict[str, Any]],
    selected_tables: list[str],
    schema_loader,
    semantic_frame: dict[str, Any] | None,
    base_conditions: list[str] | None = None,
) -> dict[str, Any]:
    """Достроить WHERE доменными правилами и value profiles."""
    conditions: list[str] = list(base_conditions or [])
    reasoning: list[str] = []
    applied_rules: list[str] = []
    clarification_message = ""
    ranked_candidates = rank_filter_candidates(
        user_input=user_input,
        intent=intent,
        selected_tables=selected_tables,
        schema_loader=schema_loader,
        semantic_frame=semantic_frame,
    )

    for request_id, candidates in ranked_candidates.items():
        if not candidates:
            continue
        filtered = []
        for candidate in candidates:
            schema = candidate.get("schema")
            table = candidate.get("table")
            if schema and table and not table_can_satisfy_frame(schema_loader, schema, table, semantic_frame):
                continue
            filtered.append(candidate)
        candidates = filtered or candidates
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        score_gap = float(best.get("score", 0.0)) - float((second or {}).get("score", 0.0))
        if best.get("confidence") == "low":
            continue
        if second and best.get("confidence") == "medium" and score_gap < 15.0:
            clarification_message = (
                "Найдено несколько близких вариантов фильтра. "
                f"Уточните, пожалуйста, по какому признаку фильтровать: "
                f"`{best.get('column')}` или `{second.get('column')}`?"
            )
            reasoning.append(f"ambiguity:{request_id}:gap={score_gap:.1f}")
            break

        _add_unique(conditions, str(best.get("condition") or ""))
        applied_rules.append(str(request_id))
        evidence = ", ".join(best.get("evidence", []) or [])
        reasoning.append(
            f"{best.get('table_key')}: {best.get('column')} -> {best.get('condition')}"
            + (f" ({evidence})" if evidence else "")
        )

    filter_intents = list((semantic_frame or {}).get("filter_intents") or [])
    if filter_intents and not applied_rules and not clarification_message:
        clarification_message = (
            "Нашёл несколько возможных способов применить фильтр, "
            "но уверенности недостаточно. Уточните, пожалуйста, желаемый признак или значение."
        )
        logger.info("WhereResolver: не удалось привязать filter_intents автоматически")

    logger.info(
        "WhereResolver: filter_intents=%s, applied_rules=%s, conditions=%s, candidates=%s",
        filter_intents, applied_rules, conditions,
        {k: [c.get('column') for c in v[:3]] for k, v in ranked_candidates.items()},
    )
    return {
        "conditions": conditions,
        "applied_rules": applied_rules,
        "reasoning": reasoning,
        "filter_candidates": ranked_candidates,
        "needs_clarification": bool(clarification_message),
        "clarification_message": clarification_message,
    }
